import numpy as np
from scipy.stats import norm, expon


class Normal:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return norm.logpdf(x, loc=self.mu, scale=self.sigma)


class Exp:
    def __init__(self, mu=1):
        self.mu = mu

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return expon.logpdf(np.sign(self.mu) * x, scale=np.sign(self.mu) * self.mu)


class Flat:
    def __call__(self, x):
        return 0.0
