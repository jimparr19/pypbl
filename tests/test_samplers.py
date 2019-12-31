import numpy as np

from scipy.stats import multivariate_normal

from pypbl.samplers import simple_sampler, ensemble_sampler


def example_distribution(x):
    mus = np.array([5, 5])
    sigmas = np.array([[1, .9], [.9, 1]])
    return multivariate_normal.logpdf([x[0], x[1]], mean=mus, cov=sigmas)


def test_simple_sampler():
    start = [0, 0]
    iterations = 50
    samples = simple_sampler(example_distribution, start=start, sigma=1, iterations=iterations)
    assert len(samples) == iterations


def test_ensemble_sampler():
    start = [0, 0]
    iterations = 50
    samples = ensemble_sampler(example_distribution, start=[0, 0], sigma=1, iterations=iterations)
    assert len(samples) == (2 * len(start) * iterations)
