# Contributing to KV Cache Prefetch Experiments

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the KV Cache Prefetch Experiments repository.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies: `pip install -r requirements.txt`
4. Create a new branch for your changes

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing
- Test your changes thoroughly
- Run the main orchestrator to ensure experiments work
- Verify that metrics are collected correctly

### Adding New Features

#### New KV Cache Strategies
1. Create a new module following the pattern of existing implementations
2. Implement the required interface methods
3. Add configuration options to `experiment_config.yaml`
4. Update the main orchestrator to support your new strategy
5. Add documentation and examples

#### New Metrics
1. Add metric collection to the appropriate modules
2. Update the metrics aggregation in `main.py`
3. Include the new metrics in the summary table
4. Update documentation

### Configuration Changes
- Update `experiment_config.yaml` for new experiment parameters
- Ensure backward compatibility when possible
- Document new configuration options

## Submitting Changes

1. Ensure all tests pass
2. Update documentation as needed
3. Commit your changes with descriptive messages
4. Push to your fork
5. Submit a pull request

## Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Include test results if applicable
- Ensure the code follows project conventions

## Questions?

Feel free to open an issue for questions or discussions about the project.
