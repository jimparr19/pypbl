import numpy as np
from scipy.stats import norm, expon


class Normal:
    """
    Normal prior. This prior is useful if you have a good guess for what the weight should be, and an understanding of
    how much you expect to differ from that guess.

    """

    def __init__(self, mu=0, sigma=1):
        """
        Args:
            mu (float): mean value of normal distribution
            sigma (float): standard deviation of normal distribution
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        """
        Args:
            x (float): weight

        Returns:
            Log probability for weight

        """
        return norm.logpdf(x, loc=self.mu, scale=self.sigma)


class Exponential:
    """
    Exponential prior. This prior is particularly useful if you deterministically know the sign of the weight, and have
    a guess for the value of the weight. The mean may be negative.

    """

    def __init__(self, mu=1):
        """
        Args:
            mu (float): mean value of exponential distribution
        """
        self.mu = mu

    def __call__(self, x):
        """
        Args:
            x (float): weight

        Returns:
            Log probability for weight

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return expon.logpdf(np.sign(self.mu) * x, scale=np.sign(self.mu) * self.mu)


class Flat:
    """
    Flat prior. This is useful when you are completely agnostic to the weights on a particular weight. You may wish to
    remove this parameter from the analysis instead.

    """
    def __call__(self, x):
        """
        Args:
            x (float): weight

        Returns:
            Log probability for weight

        """
        return 0.0
