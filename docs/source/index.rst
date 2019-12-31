pypbl
=====

**pypbl** is a python library for preference based learning using pairwise comparisons.


.. image:: https://img.shields.io/badge/GitHub-jimparr19%2Fpypbl-blue.svg?style=flat
    :target: https://github.com/jimparr19/pypbl
.. image:: https://github.com/jimparr19/pypbl/workflows/pythonpackage/badge.svg?style=flat
    :target: https://github.com/jimparr19/pypbl/actions

Basic Usage
-----------

If we want to recommend a personalised list of items to an individual.

There are three approaches we could take:

1. Ask the individual to manually rank all items.
2. Ask the individual to provide weights based on their preferences of different features (size, cost, weight etc), and calculate the weighted value of each item.
3. Find similar people and base recommendations on what these people also like.
4. Ask the individual compare a small number of alternatives, and derive feature weights from those comparisons.

Option 1 quickly becomes an enormous burden on the user as the number of items increases.

Option 2 is difficult for the user to do and replicate. What exactly does it mean if the weight assigned to one feature is double the weight assigned to another?

Option 3 requires lots of data, a way to determine similarity between individuals and may not be fully personalised.

Option 4 is enabled by preference based learning using pairwise comparisons.

Below is an example of using pypbl to rank top choices of cars using very few pairwise comparisons

.. literalinclude:: ../../examples/cars.py

See :ref:`examples` for further uses.

Contents
--------

.. toctree::
    :maxdepth: 1

    install
    elicitation
    priors
    samplers
    examples


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
