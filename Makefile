init:
	python -m pip install -e .

check:
	ruff check 

format:
	ruff format -v

lint:
	python -m pylint pytens

type-check:
	python -m mypy 

test:
	python -m unittest tests/main_test.py -v

.PHONY: init format lint test type-check
