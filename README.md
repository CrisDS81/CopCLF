# CopCLF
Copula Based classification
CopCLF is a copula-based classifier, it can be used with single scripts or by installing the package. 

The data have to be rearranged as matrix, where the rows represent the observation and the columns represent the features plus the column of the labels.

To use it requires the "copulae" package and "openturns".
The code allows the choice of different copulas in the learning phase, Gaussian, t-Student, Frank, Gumbel, Clayton, Bernstein.
If the choice is a single copula write the name "". If the choise is multiple, write the list of copula, i.e. ["Gaussian", "Frank",...].

The marginals can be fitted both empirically with different kernels or with a normal distribution.
It can also possible to use different methods for dimensionality reduction, setting them at the beginning of the code.
Documentation: Cooming soon

