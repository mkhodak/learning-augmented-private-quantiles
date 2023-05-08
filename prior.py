import pdb
import numpy as np
import torch
from scipy import optimize


GLOBAL_MAX = 1E6
GLOBAL_MIN = -1E6
TOLERANCE = 1E-14


class Prior:
    def cdf(self, x, a=GLOBAL_MIN, b=GLOBAL_MAX):
        xgta = x > a
        altxltb = np.logical_and(xgta, x < b)
        if np.any(altxltb):
            cdfa = self._cdf(a)
            return np.where(altxltb,
                            (self._cdf(x) - cdfa) / (self._cdf(b) - cdfa),
                            xgta)
        return xgta.astype(x.dtype)
    def its(self, a, b):
        p = np.random.uniform(self._cdf(a), self._cdf(b))
        return optimize.root_scalar(lambda x: self._cdf(x)-p, 
                                    bracket=[a, b], 
                                    method='brentq', 
                                    xtol=TOLERANCE).root


class Uniform(Prior):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def _cdf(self, x):
        return np.maximum(np.minimum((x-self.a) / (self.b-self.a), 1.), 0.)
    def its(self, a, b):
        return np.random.uniform(max(a, self.a), min(b, self.b))


ROBUST_COEF = 2. / (GLOBAL_MAX - GLOBAL_MIN)
ROBUST_PRIOR = Uniform(GLOBAL_MIN, GLOBAL_MAX)


class Cauchy(Prior):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def _cdf(self, x):
        return np.arctan((x-self.loc)/self.scale) / np.pi + .5
    def its(self, a, b):
        p = np.random.uniform(self._cdf(a), self._cdf(b))
        return self.loc + self.scale * np.tan(np.pi*(p-.5))


class HalfCauchy(Prior):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def _cdf(self, x):
        return np.maximum(2./np.pi * np.arctan((x-self.loc)/self.scale), 0.)
    def its(self, a, b):
        p = np.random.uniform(self._cdf(a), self._cdf(b))
        return self.loc + self.scale * np.tan(.5*np.pi*p)


class Laplace(Prior):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def _cdf(self, x):
        x = (x - self.loc) / self.scale
        with np.errstate(under='ignore', over='ignore'):
            return np.where(x>0., 1.-.5*np.exp(-x), .5*np.exp(x))


class Mixture(Prior):
    def __init__(self, prior1, prior2, lambda2):
        self.prior1 = prior1
        self.prior2 = prior2
        self.lambda1 = 1. - lambda2
        self.lambda2 = lambda2
    def _cdf(self, x):
        with np.errstate(under='ignore', over='ignore'):
            return self.lambda1 * self.prior1._cdf(x) + self.lambda2 * self.prior2._cdf(x)
