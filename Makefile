.PHONY: format lint test

test:
	poetry run pytest

format: black ruff-fix

lint: black-check ruff-check mypy

ruff-fix:
	poetry run ruff --fix .

black:
	poetry run black .

black-check:
	poetry run black --check .

ruff-check:
	poetry run ruff check .

mypy:
	poetry run mypy . --config-file ./pyproject.toml