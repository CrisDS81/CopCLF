'''TEST KDE'''
from csv import reader
from random import seed
from random import randrange
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
#from copulas.multivariate import GaussianMultivariate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
import seaborn as sns
from KDEpy import FFTKDE
from scipy.stats import gaussian_kde
from KDE_estimation import *


def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(500)
#data = np.random.randn(2**10)
data=x
seed(1)
x1=np.random.normal(loc=0.0, scale=1.0, size=500)
data1=x1

kde=KDE_estimation('scipy_kde', option_kde)
kde.fit(data)
epdf=kde.eval_pdf(data1)

grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': 10**np.linspace(-1, 1, 100)}, cv=20) # 20-fold cross-validation
grid.fit(data[:,None])
#print (grid.best_params_)
kde = grid.best_estimator_
data=np.sort(data)
kde=kde.score_samples(data.reshape(-1,1))
density=np.exp(kde)
plt.plot(np.sort(data), density, alpha=0.75)
#=============================================================================
kdeuni= KDEUnivariate(x)
kdeuni.fit(bw=0.5,fft=True)
density_fft=kdeuni.evaluate(np.sort(x))
#sns.distplot(x, hist = False, kde = True, rug = True,color = 'darkblue', kde_kws={'linewidth': 3},rug_kws={'color': 'black'})
p=plt.hist(x,bins=50)
#plt.show(p)
p1=plt.plot(np.sort(ex),density_fft)
plt.show(p1)
p=plt.hist(x,bins=50).show
#==============================================================================

data = np.random.randn(2**10)

# Notice how bw (standard deviation), kernel, weights and grid points are set



'''x, y = FFTKDE(bw=0.5, kernel='gaussian').fit(data, weights=None).evaluate(2**8)

plt.plot(x, y); plt.tight_layout()
plt.hist(data, bins=30,density=True)
#=============================================================================
#interpolation
import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt
x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)
print (x,y)
plt.plot(x, y,'o')
plt.show()
f1 = interp.UnivariateSpline(x, y)
f2 = interp.interp1d(x, y, kind = 'cubic')'''
