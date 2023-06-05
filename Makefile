black:
	@black .

isort:
	@isort .

autoflake:
	@autoflake --remove-unused-variables --remove-all-unused-imports -r -i .

format: black isort autoflake

lint-black:
	@black --check .

lint-isort:
	@isort --check .

lint-flake8:
	@pflake8 .

lint: lint-black lint-isort lint-flake8

pre-commit:
	@pre-commit run --all-files
