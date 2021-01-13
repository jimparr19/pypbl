import copy
import itertools
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import gaussian_kde

from pypbl.priors import Exponential
from pypbl.samplers import ensemble_sampler


class BayesPreference:
    """
    Class for preference based learning using Bayesian inference

    """

    def __init__(self, data, normalise=True):
        """
        Args:
            data (object): Pandas DataFrame with columns as features and index as item names
            normalise (bool): Normalise data using unit normalisation
        """
        self.original_data = data.copy()
        self.data = (data - data.min()) / (data.max() - data.min()) if normalise else data
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
        """
        Set priors

        Args:
            priors (list): List of priors

        Raises:
            AttributeError
        """
        if len(priors) != len(self.data.columns):
            raise AttributeError('The number of priors should match the number of parameters')
        self.priors = priors

    def add_strict_preference(self, a, b):
        """
        Adds a preference based on a > b

        Args:
            a (str): name of item a that is preferred over item b
            b (str): name of item b

        Raises:
            ValueError

        """
        if a not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(a))
        if b not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(b))

        self.strict_preferences.append([a, b])

    def remove_last_strict_preference(self):
        """
        Removed the most recently added strict preference

        """
        del self.strict_preferences[-1]

    def add_indifferent_preference(self, a, b):
        """
        Adds a preference based on a == b

        Args:
            a (str): name of item a that has equal preference to item b
            b (str): name of item b

        Raises:
            ValueError

        """
        if a not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(a))
        if b not in self.items:
            raise ValueError('Parameter {} not a valid item'.format(b))

        self.indifferent_preferences.append([a, b])

    def strict_log_probability(self, preference, weights):
        """
        Computes the log probability for a strict preference

        Args:
            preference (tuple): strict preference relationship (a > b)
            weights (Numpy Array): weights for each feature

        Returns:
            Log probability of strict preference given weights

        """
        delta = self.data.loc[preference[0], :].values - self.data.loc[preference[1], :].values
        variance = delta.dot(self.Sigma).dot(delta)
        sd = np.sqrt(variance)
        mean = weights.dot(delta)
        return norm.logcdf(mean, loc=0, scale=sd)

    def indifferent_log_probability(self, preference, weights):
        """
        Computes the log probability for an indifferent preference

        Args:
            preference (tuple): indifference preference relationship (a == b)
            weights (Numpy Array): weights for each feature

        Returns:
            Log probability of indifferent preference given weights

        """
        delta = self.data.loc[preference[0], :].values - self.data.loc[preference[1], :].values
        variance = delta.dot(self.Sigma).dot(delta)
        sd = np.sqrt(variance)
        mean = weights.dot(delta)
        return np.log(norm.cdf(mean + 0.5, loc=0, scale=sd) - norm.cdf(mean - 0.5, loc=0, scale=sd))

    def log_probability(self, weights):
        """
        Computes the log posterior probability based on the sum of each strict and indifferent preference probability

        Args:
            weights (Numpy Array): weights for each feature

        Returns:
            Log posterior probability given preference data and weights

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

    def infer_weights(self, method='MAP', iterations=100):
        """
        Infer weights for each attribute based on preferences.
        method='MAP' uses gradient based optimisation to compute the maximum a posteriori,
        method='mean' uses sampling to compute a better estimate of the weights when using non-normal priors

        Args:
            method (str): method used for inference
            iterations (int): number of iterations to use when method='mean'

        Returns:
            Estimated weights using Bayesian inference

        Raises:
            AttributeError
            ValueError

        """
        available_methods = ['MAP', 'mean']
        if method not in available_methods:
            raise ValueError("'method' most be one of the available methods {}".format(available_methods))

        if self.priors is None:
            raise AttributeError('No priors have been specified.')

        if method == 'MAP':
            if any([isinstance(prior, Exponential) for prior in self.priors]):
                warnings.warn("It is recommended to use method='mean' when using exponential priors.")

        self.lb = [0 if prior(-1) == -np.inf else -np.inf for prior in self.priors]
        self.ub = [0 if prior(1) == -np.inf else np.inf for prior in self.priors]

        bounds = list(zip(self.lb, self.ub))
        x0 = np.zeros(len(self.data.columns))
        x0 = np.array([x - 1e-8 if self.ub[i] == 0 else x for i, x in enumerate(x0)])
        results = minimize(self.negative_log_probability, x0=x0, bounds=bounds)
        map_estimate = results['x']

        if method == 'MAP':
            self.weights = map_estimate
            return map_estimate

        self.samples = ensemble_sampler(fun=self.log_probability, start=map_estimate, sigma=self.sigma,
                                        iterations=iterations)

        mean_estimate = np.mean(self.samples, axis=0)

        self.weights = mean_estimate
        return mean_estimate

    def rank(self):
        """
        Rank items based on the inferred weights

        Returns:
            Pandas DataFrame of ordered items and utility values

        """
        if self.weights is None:
            self.infer_weights()

        utilities = [self.weights.dot(row.values) for i, row in self.data.iterrows()]
        rank_df = pd.DataFrame(utilities, index=self.data.index.values, columns=['utility'])
        return rank_df.sort_values(by='utility', ascending=False)

    def compute_entropy(self, x):
        """
        Compute entropy of a new preference

        Args:
            x (list): pair of items

        Returns:
            Entropy for the pair of items

        """
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
        return ab_entropy

    def suggest_new_pair(self, method='random'):
        """
        Suggest a new pair of items with minimum entropy
        method='random' chooses a new pair at random,
        method='min_entropy' chooses a new pair that minimises expected entropy

        Args:
            method (str): suggestion method one of 'random' or 'min_entropy'

        Returns:
            Pair of items that based on the selected method

        Raises:
            ValueError

        """
        available_methods = ['random', 'min_entropy']
        if method not in available_methods:
            raise ValueError("'method' most be one of the available methods {}".format(available_methods))

        possible_combinations = [tuple(sorted(p)) for p in itertools.combinations(self.data.index, 2)]
        existing_combinations = [tuple(sorted(p)) for p in self.strict_preferences] + \
                                [tuple(sorted(p)) for p in self.indifferent_preferences]
        new_combinations = list(set(possible_combinations) - set(existing_combinations))

        if method == 'random':
            index = np.random.choice(range(len(new_combinations)))
            suggestion = new_combinations[int(index)]
        elif method == 'min_entropy':
            entropy = []
            for i, x in enumerate(new_combinations):
                print('Computing entropy for {} of {} combinations'.format(i, len(new_combinations)))
                entropy.append(self.compute_entropy(x))

            index = np.argmin(entropy)
            suggestion = new_combinations[int(index)]
        return suggestion

    def suggest(self, method='random'):
        """
        Suggest a new item to compare with the most preferred item
        method='random' chooses a new pair that includes the top ranked item at random,
        method='max_variance' chooses a pair that includes the top ranked item and the item with greatest uncertainty,
        method='min_entropy' chooses a pair that includes the top ranked item and minimises expected entropy

        Args:
            method (str): suggestion method one of 'random', 'max_variance' or 'min_entropy'

        Returns:
            Pair of items that includes the top ranked item and another item based on the selected method

        Raises:
            ValueError

        """
        available_methods = ['random', 'min_entropy', 'max_variance']
        if method not in available_methods:
            raise ValueError("'method' most be one of the available methods {}".format(available_methods))

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
            return self.suggest_new_pair(method=method)

        if method == 'random':
            index = np.random.choice(range(len(new_combinations)))
            suggestion = new_combinations[int(index)]
        elif method == 'max_variance':
            self.infer_weights(method='mean')
            utility_std = []
            for items in new_combinations:
                item = [i for i in items if i != best][0]
                item_values = self.data.loc[item, :].values
                utility_std.append(np.std([sample.dot(item_values) for sample in self.samples]))
            index = np.argmax(utility_std)
            suggestion = new_combinations[int(index)]
        elif method == 'min_entropy':
            entropy = []
            for i, x in enumerate(new_combinations):
                print('Computing entropy for {} of {} combinations'.format(i, len(new_combinations)))
                entropy.append(self.compute_entropy(x))
            index = np.argmin(entropy)
            suggestion = new_combinations[int(index)]
        return suggestion
