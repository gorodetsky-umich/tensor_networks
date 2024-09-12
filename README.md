# Python Tensor Solvers

A library for python solvers

## Pytest
  - TODO

## To build docs

   - =sphinx-build -M html source build= in doc directory
   - =make html= in docs/ directory

## Style

   - =pylint -v -j 4 pytens examples=
   - =ruff check= can see the files with =ruff check --show-files=
   - =mypy --strict= (configs are read from pyproject.toml
   - =flake8 algs.py=
   - use emacs whitespace mode to see extra whitespace

## Other stuff
Using -e flag tells pip install to read package in an editable mode, which means you don't need to reinstall the package after making your changes. They get detected automatically. 

## Coding TODO
1. Regression tests for the solver
2. Cleanup of the solver problems

## Immediate


## Next
3. ALS
4. contract two nodes along edge / reshape a node
5. orthogonalize (general)
6. round/truncate (general)


## POTENTIAL ISSUES:
-1. currently edges are not actually used to determine contraction path, just index names. This causes issues when contracting when attaching tensors of edges are the same
0. benchmarking done but dimension scaling is off (possibly because path is not optimal). Could hardcode optimal TT path, but it should be ok for lower dimensions (lower number of cores) May be a problem if QTT considered in the future


## To test 
<!-- python -m unittest tests/test_something.py -->

## To profile 
python -m cProfile -o output.prof transport_solver.py input_files/hohlraum_2d.yaml
pip install snakeviz
snakeviz output.prof
