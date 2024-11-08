# Tensor Networks

A library for tensor network manipulation.

## Installation
I suggest you create a python virtual environment. Then within that environment you can install an editable installation with
```
python -m pip install -e .
```
a regular install would remove the `-e` term.

## Running

Most of the functionality is available in the unit tests. You should be able to execute 
```
python -m unittest tests/main_test.py -v
```
and see all tests pass.

You can also create some scaling plots for the inner command via
```
python examples/inner_product_scaling.py
```

## Development 

Prior to creating a pull request to merge with the default branch, you need to make sure that 
```
make ci
```
doesnt return any problems.

### Linting

To run just listing one can do 
```
make lint
```
this requires pylint, flake8 and flake8-pyproject (note this is different from pyproject-flake8) to be installed for python 3.12.

## POTENTIAL ISSUES:
0. currently edges are not actually used to determine contraction path, just index names. This causes issues when contracting when attaching tensors of edges are the same
1. benchmarking done but dimension scaling is off (possibly because path is not optimal). Could hardcode optimal TT path, but it should be ok for lower dimensions (lower number of cores) May be a problem if QTT considered in the future

## Author
Copyright 2024, Alex Gorodetsky, goroda@umich.edu
