# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:57:25 2021

@author: Cristiano


Implements different bandwidth selection methods, including:
- Scott's rule of thumb
- Silverman's rule of thumb
- Sheather-Jones estimator

Class of distribution
"""

import collections
import numpy as np
import pylab as pp
import scipy.special as special
from scipy import stats
from scipy.integrate import quad
import numpy as np


__all__ = ['wmean',
           'wvar',
           'dnorm',
           'hsilverman',
           'hscott',
           'hnorm',
           'isj',
           'proportional',
           'gamma',
           'beta',
           'exponential',
           'poisson',
           'normal',
           'multivariate_normal',
           'lognormal',
           'logitnormal',
           'uniform',
           'uniform_nd',
           'multivariate_mixture']



def wmean(x, w):
    '''
    Weighted mean
    '''
    return sum(x * w) / float(sum(w))


def wvar(x, w):
    '''
    Weighted variance
    '''
    return sum(w * (x - wmean(x, w)) ** 2) / float(sum(w) - 1)


def dnorm(x):
    return distr.normal.pdf(x, 0.0, 1.0)


def bowman(x):
    pass
    # TODO: implement?
    #hx = median(abs(x - median(x))) / 0.6745 * (4 / 3 / r.n) ^ 0.2
    #hy = median(abs(y - median(y))) / 0.6745 * (4 / 3 / r.n) ^ 0.2
    #h = sqrt(hy * hx)


def hsilverman(x, weights=None):
    IQR = np.percentile(x, 75) - np.percentile(x, 25)
    A = min(np.std(x, ddof=1), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = float(sum(weights))

    return 0.9 * A * n ** (-0.2)


def hscott(x, weights=None):

    IQR = np.percentile(x, 75) - np.percentile(x, 25)
    A = min(np.std(x, ddof=1), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = float(sum(weights))

    return 1.059 * A * n ** (-0.2)


def hnorm(x, weights=None):
    '''
    Bandwidth estimate assuming f is normal. See paragraph 2.4.2 of
    Bowman and Azzalini[1]_ for details.
    References
    ----------
    .. [1] Applied Smoothing Techniques for Data Analysis: the
        Kernel Approach with S-Plus Illustrations.
        Bowman, A.W. and Azzalini, A. (1997).
        Oxford University Press, Oxford
    '''

    x = np.asarray(x)

    if weights is None:
        weights = np.ones(len(x))

    n = float(sum(weights))

    if len(x.shape) == 1:
        sd = np.sqrt(wvar(x, weights))
        return sd * (4 / (3 * n)) ** (1 / 5.0)

    # TODO: make this work for more dimensions
    # ((4 / (p + 2) * n)^(1 / (p+4)) * sigma_i
    if len(x.shape) == 2:
        ndim = x.shape[1]
        sd = np.sqrt(np.apply_along_axis(wvar, 1, x, weights))
        return (4.0 / ((ndim + 2.0) * n) ** (1.0 / (ndim + 4.0))) * sd


def isj(x, weights=None):
    '''
    Sheather-Jones bandwidth estimator [1]_.
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    '''

    h0 = hnorm(x)
    v0 = sj(x, h0)

    if v0 > 0:
        hstep = 1.1
    else:
        hstep = 0.9

    h1 = h0 * hstep
    v1 = sj(x, h1)

    while v1 * v0 > 0:
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = sj(x, h1)

    return h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))


def sj(x, h):
    '''
    Equation 12 of Sheather and Jones [1]_
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    '''
    phi6 = lambda x: (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * dnorm(x)
    phi4 = lambda x: (x ** 4 - 6 * x ** 2 + 3) * dnorm(x)

    n = len(x)
    one = np.ones((1, n))

    lam = np.percentile(x, 75) - np.percentile(x, 25)
    a = 0.92 * lam * n ** (-1 / 7.0)
    b = 0.912 * lam * n ** (-1 / 9.0)

    W = np.tile(x, (n, 1))
    W = W - W.T

    W1 = phi6(W / b)
    tdb = np.dot(np.dot(one, W1), one.T)
    tdb = -tdb / (n * (n - 1) * b ** 7)

    W1 = phi4(W / a)
    sda = np.dot(np.dot(one, W1), one.T)
    sda = sda / (n * (n - 1) * a ** 5)

    alpha2 = 1.357 * (abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0)

    W1 = phi4(W / alpha2)
    sdalpha2 = np.dot(np.dot(one, W1), one.T)
    sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

    return (distr.normal.pdf(0, 0, np.sqrt(2)) /
            (n * abs(sdalpha2[0, 0]))) ** 0.2 - h


class proportional(object):

    '''
    Distribution class that computes the pdf and cdf values using a function
    that is proportional to the true probability density.
    Works only for 1D densities.
    '''

    def __init__(self, proportional_function, lower_limit=-np.inf, upper_limit=np.inf):
        '''
        Initialize this distribution class by precalculating the normalization
        constant.
        Arguments
        ---------
        proportional : function
            Function that is proportional to the computed distribution
        '''

        self.proportional_function = proportional_function
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.normalization, _ = quad(
            self.proportional_function,
            lower_limit,
            upper_limit)

    def pdf(self, x):
        return self.proportional_function(x) / self.normalization

    def cdf(self, x):
        if isinstance(x, collections.Iterable):
            val = np.array([quad(self.pdf, self.lower_limit, np.clip(i, self.lower_limit, self.upper_limit))[0] for i in x])
        else:
            val = quad(self.proportional_function, self.lower_limit, np.clip(x, self.lower_limit, self.upper_limit))[0] / self.normalization
        return val


class gamma(object):

    '''
    The Gamma distribution.
    '''

    @staticmethod
    def pdf(x, alpha, beta):
        return np.exp(gamma.logpdf(x, alpha, beta))

    @staticmethod
    def logpdf(x, alpha, beta):
        if beta > 0:
            return alpha * np.log(beta) - \
                special.gammaln(alpha) + (alpha - 1) * np.log(x) - beta * x
        else:
            assert False, "Beta is zero"

    @staticmethod
    def cdf(x, alpha, beta):
        return special.gammainc(alpha, x * beta)

    @staticmethod
    def cdf(x, alpha, beta):
        g = stats.gamma(alpha, 0, 1.0 / beta)
        return g.cdf(x)

    @staticmethod
    def icdf(x, alpha, beta):
        g = stats.gamma(alpha, 0, 1.0 / beta)
        return g.ppf(x)

    @staticmethod
    def rvs(alpha, beta, N=1):
        return np.random.gamma(alpha, 1.0 / beta, N)


class beta(object):

    '''
    The Beta distribution.
    '''

    @staticmethod
    def pdf(x, a, b):
        return np.exp(beta.logpdf(x, a, b))

    @staticmethod
    def logpdf(x, alpha, beta):
        return (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x) + \
            special.gammaln(alpha + beta) - special.gammaln(alpha) - \
            special.gammaln(beta)

    @staticmethod
    def cdf(x, alpha, beta):
        g = stats.beta(alpha, beta)
        return g.cdf(x)

    @staticmethod
    def icdf(x, alpha, beta):
        g = stats.beta(alpha, beta)
        return g.ppf(x)

    @staticmethod
    def rvs(alpha, beta, N=1):
        return np.random.beta(alpha, beta, N)


class exponential(object):

    '''
    The exponential distribution.
    '''

    @staticmethod
    def pdf(x, beta):
        return np.exp(exponential.logpdf(x, beta))

    @staticmethod
    def logpdf(x, beta):
        if beta > 0:
            return np.log(beta) - beta * x
        else:
            assert False, "Beta is zero"

    @staticmethod
    def rvs(beta, N=1):
        return np.random.exponential(1.0 / beta, N)


class lognormal(object):

    '''
    The log-normal distribution. When X is normally distributed,
    then Y = exp(X) is log-normally distributed.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(lognormal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        if type(x) == np.ndarray:
            if sigma > 0:
                small = np.log(0.5 + 0.5 * special.erf((np.log(1e-6) - mu) /
                               (np.sqrt(2.0) * sigma)))
                I = pp.find(x > 1e-6)
                log_x = np.log(x[I])
                lp = small * np.ones(x.shape)
                lp[I] = -log_x - 0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
                    0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
            else:
                I = pp.find(x == mu)
                lp = -np.inf * np.ones(x.shape)
                lp[I] = 0
        else:
            if sigma > 0:
                if x > 1e-6:
                    log_x = np.log(x)
                    lp = - log_x - 0.5 * np.log(2.0 * np.pi) - \
                        np.log(sigma) - \
                        0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
                else:
                    lp = np.log(0.5 + 0.5 * special.erf((np.log(1e-6) - mu) /
                                (np.sqrt(2.0) * sigma)))
            else:
                if x == mu:
                    lp = 0
                else:
                    lp = -np.inf
        return lp

    @staticmethod
    def rvs(mu, sigma):
        return np.array([np.random.lognormal(mu, sigma)])


class logitnormal(object):

    '''
    The logit-normal distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(logitnormal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        return - np.log(x) - np.log(1 - x) - \
            0.5 * np.log(2.0 * np.pi * sigma ** 2) - \
            0.5 * ((np.log(x) - np.log(1 - x) - mu) / sigma) ** 2

    @staticmethod
    def rvs(mu, sigma):
        # TODO: Implement
        pass

class poisson(object):

    '''
    The Poisson distribution.
    '''

    @staticmethod
    def pdf(x, mu):
        return np.exp(poisson.logpdf(x, mu))

    @staticmethod
    def logpdf(x, mu):
        return (x - 1) * np.log(mu) - special.gammaln(x - 1) - mu

    @staticmethod
    def rvs(mu, N=1):
        return np.random.poisson(mu, N)


class uniform(object):

    '''
    The 1D uniform distribution.
    '''

    @staticmethod
    def pdf(x, a=0, b=1):
        #if a <= x <= b:
        return (np.all([a <= x, x <= b], axis=0)) / (b - a)
        #return 0.0

    @staticmethod
    def logpdf(x, a=0, b=1):
        if a <= x <= b:
            return -np.log(b - a)
        return - np.inf

    @staticmethod
    def rvs(a=0, b=1, N=1):
        return np.random.uniform(a, b, N)

#TODO: make naming consistent
class uniform_nd(object):

    '''
    The n-dimensional uniform distribution.
    '''

    @staticmethod
    def pdf(x, p1, p2):
        float(np.all(x > p1) and np.all(x < p2)) / np.prod(p2 - p1)

    @staticmethod
    def logpdf(x, p1, p2):
        if np.all(x > p1) and np.all(x < p2):
            return -np.sum(np.log(p2 - p1))
        return -np.inf

    @staticmethod
    def rvs(p1, p2, N=1):
        if N == 1:
            return np.random.uniform(p1, p2)
        return np.random.uniform(p1, p2, (N, len(p1)))


class normal(object):

    '''
    The 1D normal (or Gaussian) distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(normal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
            0.5 * ((x - mu) ** 2) / (sigma ** 2)

    @staticmethod
    def rvs(mu, sigma, N=1):
        return np.random.normal(mu, sigma, N)


class multivariate_normal(object):

    '''
    The multivariate normal (or Gaussian) distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(multivariate_normal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        m = np.matrix(x - mu)
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        try:
            k = x.shape[-1]
        except AttributeError:
            # Floats have no shape attribute
            k = 1
        return -0.5 * (m * inv * m.T + k * np.log(2 * np.pi) + np.log(det))

    @staticmethod
    def rvs(mu, sigma, N=1):
        return np.random.multivariate_normal(mu, sigma, N).ravel()


class multivariate_mixture(object):

    '''
    A multivariate distribution that consists of the product of multiple
    one-dimensional distributions.
    '''

    def __init__(self, distributions, arguments):
        '''
        Arguments
        ---------
        distributions : list
            The list of the distributions this is a product of.
        arguments : list of lists
            The list of arguments for the different distributions.
        '''
        self.distrs = distributions
        self.args = arguments

    def pdf(self, thetas):
        return np.exp(self.logpdf(thetas))

    def logpdf(self, thetas):
        logpdf = 0.0
        for theta, distribution, arguments in zip(thetas, self.distrs, self.args):
            logpdf += distribution.logpdf(theta, *arguments)
        return logpdf

    def rvs(self, N=1):
        D = len(self.distrs)
        thetas = np.zeros((D, N))
        for t, theta, distribution, arguments in zip(range(D), thetas, self.distrs, self.args):
            thetas[t, :] = distribution.rvs(*arguments, N=N)
        if N == 1:
            return thetas.ravel()
        return thetas