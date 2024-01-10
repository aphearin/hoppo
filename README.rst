hoppo
============

This is a temporary repository used to debug DiffstarPop kernels.

The goal of this repo is to get the unit testing suite to pass.
Currently there is only a single unit test, test_mcpop_evaluates. 
Once it passes, this code will be deleted ported into other repos and deleted.

Installation
------------
To install hoppo into your environment from the source code::

    $ cd /path/to/root/hoppo
    $ pip install .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/hoppo
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

