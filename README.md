# Neural Network Framework

This project is a framework for building neural networks. It is built on top of the [NumPy](https://numpy.org/) library and is designed to be easy to use and extend. The framework is designed to be used for educational purposes and is not optimized for performance.

## Features

- [x] Fully connected layers
- [ ] Convolutional layers
- [x] Stochastic Gradient Descent
- [x] Momentum
- [x] Multiple activation functions
- [x] Multiple loss functions
- [x] Dropout
- [ ] Batch normalization

## Installation and Setup

1. Install poetry by following the instructions on the [poetry website](https://python-poetry.org/docs/#installation)

2. After installation, navigate to the root directory of this project in your terminal and run the following command to install the project's dependencies:

```bash
poetry install
```

## MNIST data download

The MNIST data is not included in the repository. Please download the following file, unzip it and place its content in the data directory:
[MNIST data](https://drive.google.com/file/d/1n-oOA7XDukD1rOSAlN_2nS2m5oSFeJNe/view?usp=sharing).

## Interacting with Poetry

### Poetry Shell

You can use the poetry shell command to activate a virtual environment for the project. This allows you to run python commands and use project-specific dependencies without affecting your global python environment. To activate the poetry shell, run the following command in the root directory of the project:

```bash
poetry shell
```

### Poetry Add

You can use the poetry add command to add new dependencies to the project. For example, to add the package requests you can run:

```bash
poetry add requests # adds the code to pyproject.toml and installs it in venv
poetry update # updates the lock file
```

Note: also push the updated pyproject.toml and poetry.lock files to the repository.

### Export Requirements

The dependencies for this project are defined in the pyproject.toml file. If you need to export the dependencies to a requirements.txt file, you can run the following command in the root directory of the project:

```bash
poetry export --without-hashes --format=requirements.txt
```

## Pre-commit Hooks

This project also utilizes pre-commit for code formatting and linting. To install the pre-commit hooks, run the following command in the root directory of the project after activating the poetry shell:

```bash
pre-commit install
```

This will install the pre-commit hooks defined in the .pre-commit-config.yaml file. These hooks will automatically format and lint your code when committing changes.

## Testing

This project uses pytest for testing. To run the tests, run the following command in the root directory of the project after activating the poetry shell:

```bash
pytest
```

## Examples

To use examples, copy them to the root directory of the project and run them with the following command:

```bash
cp examples/example.py .
python example.py
```

It could be necessary in VSCode to set the PYTHONPATH environment variable to the root directory of the project.
