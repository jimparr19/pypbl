============
Installation
============

Package managers
----------------

The recommended way to install the stable version of pypbl is using pip

.. code-block:: bash

    pip install pypbl


From source
-----------

If installing from source, it is recommended to use `Poetry <https://python-poetry.org/>`_ to manage dependencies and virtual environments


.. code-block:: bash

    poetry install


To run tests

.. code-block:: bash

    poetry run pytest --cov=src --cov-branch --cov-fail-under=90 tests/