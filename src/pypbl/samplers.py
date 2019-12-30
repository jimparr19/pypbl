import emcee

import numpy as np

from multiprocessing import Pool


def simple_sampler(fun, start, sigma=1, iterations=1000):
    """
    Simple sampler based on the metropolis hastings algorithm for Markov chain Monte Carlo

    """
    mean = np.zeros(len(start))
    cov = np.eye(len(start)) * sigma

    if isinstance(start, np.ndarray):
        previous = start
    else:
        previous = np.array(start)

    f_previous = fun(previous)

    samples = np.zeros((iterations, len(start)))
    acceptance = 0
    for i in range(iterations):
        proposal = previous + np.random.multivariate_normal(mean=mean, cov=cov)
        f_proposal = fun(proposal)
        fun(previous)
        if (np.log(np.random.rand())) < (f_proposal - f_previous):
            previous = proposal
            acceptance += 1
        samples[i] = np.array(previous)

    print('sampler acceptance = {0:.3f}'.format(acceptance / iterations))

    return samples


def ensemble_sampler(fun, start, sigma, iterations=500):
    """
    Sampler based on the affine-invariant ensemble sampler for Markov chain Monte Carlo

    """
    n_dim = len(start)
    n_walkers = 2 * n_dim
    initial_state = np.array([start + np.random.normal(size=n_dim, scale=sigma) for _ in range(n_walkers)])

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=n_dim, log_prob_fn=fun, pool=pool)
        sampler.run_mcmc(initial_state=initial_state, nsteps=iterations)

    samples = sampler.get_chain(flat=True)

    print('sampler acceptance = {0:.3f}'.format(np.mean(sampler.acceptance_fraction)))

    return samples


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import seaborn as sns
    from scipy.stats import multivariate_normal

    mus = np.array([5, 5])
    sigmas = np.array([[1, .9], [.9, 1]])


    def pgauss(x):
        return multivariate_normal.logpdf([x[0], x[1]], mean=mus, cov=sigmas)


    # samples = simple_sampler(pgauss, start=[0, 0], sigma=1, iterations=1000)
    samples = ensemble_sampler(pgauss, start=[0, 0], sigma=sigmas, iterations=5000)
    print(samples)
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()
