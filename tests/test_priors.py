import pytest

import numpy as np
from pypbl.priors import Normal, Exponential, Flat


def test_default_normal_prior():
    prior = Normal()
    assert prior.mu == 0
    assert prior.sigma == 1
    assert prior(0) > prior(-1)
    assert prior(0) > prior(1)
    assert prior(np.array([0])) > prior(np.array([-1]))


def test_normal_prior():
    prior = Normal(mu=2, sigma=5)
    assert prior.mu == 2
    assert prior.sigma == 5
    assert prior(2) > prior(-3)
    assert prior(2) > prior(7)


def test_normal_prior_with_negative_weights():
    prior = Normal(mu=-2, sigma=5)
    assert prior.mu == -2
    assert prior.sigma == 5
    assert prior(-2) > prior(3)
    assert prior(-2) > prior(-7)


def test_default_exponential_prior():
    prior = Exponential()
    assert prior.mu == 1
    assert prior(0) > prior(-1)
    assert prior(0) > prior(1)
    assert prior(np.array([0])) > prior(np.array([-1]))


def test_exponential_prior():
    prior = Exponential(mu=0.5)
    assert prior.mu == 0.5
    assert prior(0.5) > prior(-1)
    assert prior(0.5) > prior(1)


def test_exponential_prior_with_negative_weights():
    prior = Exponential(mu=-0.5)
    assert prior.mu == -0.5
    assert prior(-0.5) > prior(-1)
    assert prior(-0.5) > prior(1)


def test_flat_prior():
    prior = Flat()
    assert prior(0) == 0
    assert prior(-1) == 0
    assert prior(1) == 0
    assert prior(np.array([0])) == 0


if __name__ == '__main__':
    pytest.main()
