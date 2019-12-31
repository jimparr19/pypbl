# pypbl

A python library for preference based learning using pairwise comparisons.

We want to recommend a personalised list of items to an individual. 

There are three approaches we could take:

1. Ask the individual manually rank all items.
2. Ask the individual to provide weights for the desirability of different item features (size, cost, weight etc), and calculate the weighted value of each item.
3. Find similar people and base recommendations on what these people also like.
3. Ask the individual compare a small number of alternatives, and derive feature weights from those comparisons.

Option 1 quickly becomes an enormous burden on the user as the number of items increases. 

Option 2 is difficult for the user to do and replicate. What exactly does it mean if the weight assigned to one feature is double the weight assigned to another?

Option 3 requires lots of data, a way to determine similarity between individuals and may not be fully personalised

=======
An example use case is below:

If we wanted to recommend a personalised list of preferred items, there are three approaches we could take:

1. Have each individual manually rank all options.
2. Ask each individual tp provide weights for the desirability of different item features, and calculate the weighted value of each item.
3. Find similar people and based recommendations on what these people also like.
3. Have the individual compare a small number of alternatives, and derive feature weights from those comparisons.

Option 1 quickly becomes an enormous burden on the user as the number of items increases. 
Option 2 is difficult for the user to do and replicate. What exactly does it mean if the weight assigned to one feature is double the weight assigned to another?
Option 3 requires lots of data, a way to determine similarity between individuals and may not be fully personalised
Option 4 is enabled by preference based learning using pairwise comparisons.

### Installing

```
pip install pypbl
```

## Development

Dependencies and packaging are managed using [Poetry](https://github.com/python-poetry/poetry). 

Install poetry and clone the repository


To create a virtual environment and install dependencies
```
poetry install
```

To run tests
```
poetry run pytest --cov=src --cov-branch --cov-fail-under=90 tests/
```

To run linting
```
poetry run flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Early versions of this package is heavily based on the [PrefeR](https://cran.r-project.org/web/packages/prefeR/index.html) library by John Lepird. 
* [PreferenceElicitation.jl](https://github.com/sisl/PreferenceElicitation.jl) by Mykel Kochenderfer.
* [Interactive Bayesian Optimisation](https://github.com/misterwindupbird/IBO) by Eric Brochu.


## TODO
* Improve suggestion engine as using entropy is expensive and required a hammer to compute accurately 
* Include [dynasty](https://github.com/joshspeagle/dynesty) sampling algorithm and [PyMC3](https://docs.pymc.io/) or perhaps [pyMC4](https://github.com/pymc-devs/pymc4)
* Include preference elicitation using Gaussian Processes see [gp_pref_elicit](https://github.com/lmzintgraf/gp_pref_elicit)
