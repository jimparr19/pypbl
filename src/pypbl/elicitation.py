import copy
import itertools
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import gaussian_kde

from pypbl.priors import Normal, Exponential
from pypbl.samplers import ensemble_sampler


class BayesPreference:
    """
    Class for preference based learning using Bayesian inference

    """

    def __init__(self, data):
        self.data = data
        self.items = self.data.index.values
        self.sigma = 0.1
        self.Sigma = self.sigma * np.eye(len(self.data.columns))
        self.strict_preferences = []
        self.indifferent_preferences = []
        self.priors = None
        self.weights = None
        self.samples = None
        self.lb = None
        self.ub = None

    def set_priors(self, priors):
        if len(priors) != len(self.data.columns):
            raise AttributeError('The number of priors should match the number of parameters')
        self.priors = priors

    def add_strict_preference(self, a, b):
        """"
        Adds a preference based on a > b

        """
        if a not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(a))
        if b not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(b))

        self.strict_preferences.append([a, b])

    def remove_last_strict_preference(self):
        """"
        Removed the most recently added strict preference

        """
        del self.strict_preferences[-1]

    def add_indifferent_preference(self, a, b):
        """"
        Adds a preference based on a == b

        """
        if a not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(a))
        if b not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(b))

        self.indifferent_preferences.append([a, b])

    def strict_log_probability(self, preference, weights):
        """"
        Computes the log probability for a strict preference

        """
        delta = self.data.loc[preference[0], :].values - self.data.loc[preference[1], :].values
        variance = delta.dot(self.Sigma).dot(delta)
        mean = weights.dot(delta)
        return norm.logcdf(mean, loc=0, scale=np.sqrt(variance))

    def indifferent_log_probability(self, preference, weights):
        """"
        Computes the log probability for an indifferent preference

        """
        delta = self.data.loc[preference[0], :].values - self.data.loc[preference[1], :].values
        variance = delta.dot(self.Sigma).dot(delta)
        sd = np.sqrt(variance)
        mean = weights.dot(delta)
        return np.log(norm.cdf(mean + 0.5, loc=0, scale=sd) - norm.cdf(mean - 0.5, loc=0, scale=sd))

    def log_probability(self, weights):
        """"
        Computes the log posterior probability based on the sum of each strict and indifferent preference probability

        """
        strict_log_probability = sum(
            [self.strict_log_probability(preference, weights) for preference in self.strict_preferences])
        indifferent_log_probability = sum(
            [self.indifferent_log_probability(preference, weights) for preference in self.indifferent_preferences])
        log_prior = sum([self.priors[i](weight) for i, weight in enumerate(weights)])
        return strict_log_probability + indifferent_log_probability + log_prior

    def negative_log_probability(self, weights):
        return -1.0 * self.log_probability(weights)

    def probability(self, weights):
        return np.exp(self.log_probability(weights))

    def infer_weights(self, method='MAP', iterations=500):
        """"
        Infer weights for each attribute based on preferences

        """
        if self.priors is None:
            raise AttributeError('No priors have been specified.')

        if any([isinstance(prior, Exponential) for prior in self.priors]):
            warnings.warn("It is recommended to use method='mean' when using exponential priors.")

        self.lb = [0 if prior(-1) == -np.inf else -np.inf for prior in self.priors]
        self.ub = [0 if prior(1) == -np.inf else np.inf for prior in self.priors]

        bounds = list(zip(self.lb, self.ub))
        x0 = np.zeros(len(self.data.columns))
        x0 = np.array([x - 1e-8 if self.ub[i] == 0 else x for i, x in enumerate(x0)])
        results = minimize(self.negative_log_probability, x0=x0, bounds=bounds)
        map_estimate = results['x']

        print('map estimate = {}'.format(self.negative_log_probability(map_estimate)))
        print('map estimate weights {}'.format(map_estimate))
        if method == 'MAP':
            self.weights = map_estimate
            return map_estimate

        self.samples = ensemble_sampler(fun=self.log_probability, start=map_estimate, sigma=self.sigma,
                                        iterations=iterations)

        mean_estimate = np.mean(self.samples, axis=0)
        print('mean estimate = {}'.format(self.negative_log_probability(mean_estimate)))
        print('mean estimate weights {}'.format(mean_estimate))

        self.weights = mean_estimate
        return mean_estimate

    def rank(self):
        """"
        Rank items based on the inferred weights

        """
        if self.weights is None:
            self.infer_weights()

        utilities = [self.weights.dot(row.values) for i, row in self.data.iterrows()]
        rank_df = pd.DataFrame(utilities, index=self.data.index.values, columns=['utility'])
        return rank_df.sort_values(by='utility', ascending=False)

    def compute_entropy(self, x):
        """"
        Compute entropy of a new preference

        """
        if self.samples is None:
            self.infer_weights(method='mean')

        p_new = copy.deepcopy(self)
        p_new.add_strict_preference(x[0], x[1])

        ab_prob = p_new.probability(self.weights)
        p_new.infer_weights(method='mean')
        ab_samples = p_new.samples

        p_new.remove_last_strict_preference()
        p_new.add_strict_preference(x[1], x[0])

        p_new.infer_weights(method='mean')
        ba_samples = p_new.samples

        bins = 100
        a_entropy = 0
        b_entropy = 0
        for weight in range(len(self.weights)):
            a_samples = ab_samples[:, weight]
            b_samples = ba_samples[:, weight]
            a_kernel = gaussian_kde(a_samples)
            b_kernel = gaussian_kde(b_samples)
            a_density = a_kernel.evaluate(np.linspace(min(a_samples), max(a_samples), bins))
            b_density = b_kernel.evaluate(np.linspace(min(b_samples), max(b_samples), bins))
            a_entropy += a_density.dot(np.log(a_density))
            b_entropy += b_density.dot(np.log(b_density))

        ab_entropy = -(ab_prob * a_entropy + (1 - ab_prob) * b_entropy)
        print('entropy for {} and {} is {}'.format(x[0], x[1], ab_entropy))
        return ab_entropy

    def suggest_new_pair(self):
        """"
        Suggest a new pair of items with minimum entropy

        """
        possible_combinations = [tuple(sorted(p)) for p in itertools.combinations(self.data.index, 2)]
        existing_combinations = [tuple(sorted(p)) for p in self.strict_preferences] + \
                                [tuple(sorted(p)) for p in self.indifferent_preferences]
        new_combinations = list(set(possible_combinations) - set(existing_combinations))
        entropy = [self.compute_entropy(x) for x in new_combinations]
        index = np.argmin(entropy)
        return new_combinations[int(index)]

    def suggest(self):
        """"
        Suggest a new item to compare with the most preferred item

        """
        best = self.rank().index.values[0]
        possible_combinations = [tuple(sorted(p)) for p in itertools.combinations(self.data.index, 2) if best in p]
        existing_combinations = [tuple(sorted(p)) for p in self.strict_preferences if best in p] + \
                                [tuple(sorted(p)) for p in self.indifferent_preferences if best in p]
        new_combinations = list(set(possible_combinations) - set(existing_combinations))
        if len(new_combinations) == 0:
            warnings.warn(
                '''
                All pairs that include the highest ranked item have been suggested, suggesting a fresh new pair instead.
                ''')
            return self.suggest_new_pair()
        entropy = [self.compute_entropy(x) for x in new_combinations]
        index = np.argmin(entropy)
        return new_combinations[int(index)]


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('data/mtcars.csv')
    print(data)
    data.set_index('model', inplace=True)
    p = BayesPreference(data=data)
    # p.priors = [
    #     Exponential(1),  # MPG
    #     Normal(),  # Number of cylinders (Normal() = Normal(0, 1))
    #     Normal(),  # displacement
    #     Exponential(2),  # horsepower
    #     Normal(),  # real axle ratio
    #     Normal(),  # weight
    #     Exponential(-3),  # quarter mile time
    #     Normal(),  # Engine type
    #     Normal(),  # transmission type
    #     Normal(),  # number of gears
    #     Normal()  # number of carburetors
    # ]
    p.priors = [
        Normal(),  # MPG
        Normal(),  # Number of cylinders (Normal() = Normal(0, 1))
        Normal(),  # displacement
        Normal(),  # horsepower
        Normal(),  # real axle ratio
        Normal(),  # weight
        Normal(),  # quarter mile time
        Normal(),  # Engine type
        Normal(),  # transmission type
        Normal(),  # number of gears
        Normal()  # number of carburetors
    ]
    p.add_strict_preference('Pontiac Firebird', 'Fiat 128')
    p.add_strict_preference('Mazda RX4', 'Mazda RX4 Wag')
    p.add_indifferent_preference('Merc 280', 'Merc 280C')
    # p.add_strict_preference('Mazda RX4', 'Cadillac Fleetwood')
    # p.add_strict_preference('Chrysler Imperial', 'Mazda RX4')
    p.infer_weights()
    print(p.weights)
    print(p.rank().head(5))
    # p.compute_entropy(['Pontiac Firebird', 'Fiat 128'])
    # p.compute_entropy(['Cadillac Fleetwood', 'Merc 450SL'])
    # print(p.suggest())
    # print(p.suggest_new_pair())
    # import matplotlib.pyplot as plt
    # for i in range(len(p.priors)):
    #     plt.hist(p.samples[:, i], 100, color="k", histtype="step")
    #     plt.show()
