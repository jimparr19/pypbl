import pytest

import numpy as np
import pandas as pd

from pypbl.elicitation import BayesPreference
from pypbl.priors import Normal, Exponential


@pytest.fixture
def basic_model():
    data = pd.DataFrame({'x': [1, 0, 1], 'y': [0, 1, 1]}, index=['item 0', 'item 1', 'item 2'])
    model = BayesPreference(data=data)
    return model


def test_set_priors(basic_model):
    assert basic_model.priors is None
    basic_model.set_priors([Normal(), Normal()])
    for prior in basic_model.priors:
        assert isinstance(prior, Normal)


def test_incorrect_set_priors(basic_model):
    assert basic_model.priors is None
    with pytest.raises(AttributeError):
        basic_model.set_priors([Normal()])


def test_set_strict_preference(basic_model):
    assert len(basic_model.strict_preferences) == 0
    basic_model.add_strict_preference('item 0', 'item 1')
    assert len(basic_model.strict_preferences) == 1


def test_set_strict_preference_invalid_items(basic_model):
    with pytest.raises(ValueError):
        basic_model.add_strict_preference(0, 1)


def test_set_indifferent_preference(basic_model):
    assert len(basic_model.indifferent_preferences) == 0
    basic_model.add_indifferent_preference('item 0', 'item 1')
    assert len(basic_model.indifferent_preferences) == 1


def test_set_invalid_preference_invalid_items(basic_model):
    with pytest.raises(ValueError):
        basic_model.add_indifferent_preference(0, 1)


def test_remove_last_strict_preference(basic_model):
    assert len(basic_model.strict_preferences) == 0
    basic_model.add_strict_preference('item 0', 'item 1')
    assert len(basic_model.strict_preferences) == 1
    basic_model.remove_last_strict_preference()
    assert len(basic_model.strict_preferences) == 0


def test_strict_log_probability(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Exponential(0.5)])
    x = np.array([1.0, 0.5])
    assert basic_model.strict_log_probability(('item 0', 'item 1'), x) == pytest.approx(-0.1413058, 0.001)
    assert basic_model.strict_log_probability(('item 1', 'item 0'), x) == pytest.approx(-2.026650, 0.001)


def test_indifferent_log_probability(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Exponential(0.5)])
    x = np.array([1.0, 0.5])
    assert basic_model.indifferent_log_probability(('item 0', 'item 1'), x) == pytest.approx(-0.7188213, 0.001)


def test_log_probability(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Exponential(0.5)])
    x = np.array([1.0, 0.5])
    assert basic_model.log_probability(x) == basic_model.priors[0](x[0]) + basic_model.priors[1](x[1])


def test_negative_log_probability(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Exponential(0.5)])
    x = np.array([1.0, 0.5])
    assert basic_model.negative_log_probability(x) == - basic_model.log_probability(x)


def test_probability(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Exponential(0.5)])
    x = np.array([1.0, 0.5])
    assert basic_model.probability(x) == np.exp(basic_model.log_probability(x))


def test_inference_raises_error(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    with pytest.raises(ValueError):
        basic_model.infer_weights(method='test')


def test_inference_with_normal_priors(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Normal(2, 0.5)])
    assert basic_model.weights is None
    basic_model.infer_weights()
    assert all(a - b < 1e-4 for a, b in zip(basic_model.weights.tolist(), [1, 2]))


def test_inference_with_normal_priors_parsing_method(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Normal(2, 0.5)])
    assert basic_model.weights is None
    basic_model.infer_weights(method='MAP')
    assert all(a - b < 1e-4 for a, b in zip(basic_model.weights.tolist(), [1, 2]))


def test_inference_with_normal_priors_parsing_mean_method(basic_model):
    basic_model.set_priors([Normal(1, 0.5), Normal(2, 0.5)])
    assert basic_model.weights is None
    basic_model.infer_weights(method='mean', iterations=500)
    assert all(a - b < 0.5 for a, b in zip(basic_model.weights.tolist(), [1, 2]))


def test_inference_with_different_priors(basic_model):
    basic_model.set_priors([Normal(1, 1), Exponential(-0.5)])
    assert basic_model.weights is None
    with pytest.warns(UserWarning):
        basic_model.infer_weights()
    assert all(a - b < 1e-4 for a, b in zip(basic_model.weights.tolist(), [1, 0]))


def test_inference_with_strict_preferences(basic_model):
    basic_model.set_priors([Normal(0, 1), Normal(0, 1)])
    basic_model.add_strict_preference('item 0', 'item 2')
    assert basic_model.weights is None
    basic_model.infer_weights()
    assert basic_model.weights is not None
    assert basic_model.weights[0] > basic_model.weights[1]
    basic_model.add_strict_preference('item 1', 'item 2')
    basic_model.infer_weights()
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights()
    assert basic_model.weights[0] > basic_model.weights[1]


def test_inference_with_indifferent_preferences(basic_model):
    basic_model.set_priors([Normal(0, 1), Normal(0, 1)])
    basic_model.add_indifferent_preference('item 0', 'item 2')
    basic_model.infer_weights()
    assert basic_model.weights[0] == basic_model.weights[1]
    basic_model.add_indifferent_preference('item 1', 'item 2')
    basic_model.infer_weights()
    assert basic_model.weights[0] == basic_model.weights[1]


def test_inference_with_strict_and_indifferent_preferences(basic_model):
    basic_model.set_priors([Normal(0, 1), Normal(0, 1)])
    basic_model.add_strict_preference('item 0', 'item 2')
    assert basic_model.weights is None
    basic_model.infer_weights()
    assert basic_model.weights is not None
    assert basic_model.weights[0] > basic_model.weights[1]
    basic_model.add_strict_preference('item 1', 'item 2')
    basic_model.infer_weights()
    assert basic_model.weights[0] == basic_model.weights[1]
    basic_model.add_indifferent_preference('item 0', 'item 1')
    basic_model.infer_weights()
    assert basic_model.weights[0] == basic_model.weights[1]
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights()
    assert basic_model.weights[0] > basic_model.weights[1]


def test_inference_with_strict_and_indifferent_preferences_with_mean_method(basic_model):
    basic_model.set_priors([Normal(0, 1), Normal(0, 1)])
    basic_model.add_strict_preference('item 0', 'item 2')
    basic_model.add_strict_preference('item 1', 'item 2')
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights(method='mean')
    assert basic_model.weights[0] > basic_model.weights[1]


def test_suggest_new_pair_random_method(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 2')
    basic_model.add_strict_preference('item 1', 'item 2')
    basic_model.infer_weights()
    new_pair = basic_model.suggest_new_pair(method='random')
    for item in new_pair:
        assert item in ['item 0', 'item 1']


def test_suggest_random_method(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights()
    pair = basic_model.suggest(method='random')
    assert 'item 0' in pair


def test_suggest_new_pair_entropy_method(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 2')
    basic_model.add_strict_preference('item 1', 'item 2')
    basic_model.infer_weights()
    new_pair = basic_model.suggest_new_pair(method='min_entropy')
    for item in new_pair:
        assert item in ['item 0', 'item 1']


def test_suggest_entropy_method(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights()
    pair = basic_model.suggest(method='min_entropy')
    assert 'item 0' in pair


def test_suggest_variance_method(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights()
    pair = basic_model.suggest(method='max_variance')
    assert 'item 0' in pair


def test_suggest_all_suggested_pairs(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.add_strict_preference('item 0', 'item 2')
    basic_model.infer_weights()
    with pytest.warns(UserWarning):
        pair = basic_model.suggest()
    assert 'item 0' not in pair


def test_suggest_raises_error(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.add_strict_preference('item 0', 'item 2')
    with pytest.raises(ValueError):
        basic_model.suggest(method='test')
    with pytest.raises(ValueError):
        basic_model.suggest_new_pair(method='test')


def test_rank(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 2', 'item 0')
    basic_model.add_strict_preference('item 2', 'item 1')
    basic_model.add_strict_preference('item 1', 'item 0')
    basic_model.infer_weights()
    rank = basic_model.rank()
    assert rank.index[0] == 'item 2'
    assert rank.index[1] == 'item 1'
    assert rank.index[2] == 'item 0'


def test_rank_automatic_inference(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 2', 'item 0')
    basic_model.add_strict_preference('item 2', 'item 1')
    basic_model.add_strict_preference('item 1', 'item 0')
    rank = basic_model.rank()
    assert rank.index[0] == 'item 2'
    assert rank.index[1] == 'item 1'
    assert rank.index[2] == 'item 0'


def test_compute_entropy(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    basic_model.infer_weights()
    low_entropy = basic_model.compute_entropy(['item 0', 'item 2'])
    high_entropy = basic_model.compute_entropy(['item 0', 'item 1'])
    assert high_entropy
    assert low_entropy


def test_compute_entropy_automatic_inference(basic_model):
    basic_model.set_priors([Normal(), Normal()])
    basic_model.add_strict_preference('item 0', 'item 1')
    low_entropy = basic_model.compute_entropy(['item 0', 'item 2'])
    high_entropy = basic_model.compute_entropy(['item 0', 'item 1'])
    assert high_entropy
    assert low_entropy


if __name__ == '__main__':
    pytest.main()
