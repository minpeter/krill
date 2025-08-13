# Project: Krill

## Overview
Krill is a Python CLI tool designed for pre-training Large Language Models (LLMs). It leverages `click` for its command-line interface, `Pydantic` for robust configuration validation, and `PyTorch`, `Hugging Face Transformers`, and `Accelerate` for efficient LLM-related tasks.

## Key Features
- **LLM Pre-training**: Facilitates the pre-training of large language models.
- **CLI Interface**: User-friendly command-line interface built with `click`.
- **Configurable**: Utilizes YAML files for flexible configuration, with schema validation provided by `Pydantic`.
- **Deep Learning Stack**: Built on `PyTorch`, integrated with `Hugging Face Transformers` for model architectures and `Accelerate` for distributed training.

## Installation

### For Training (with CUDA support)
```bash
uv pip install 'krill[cuda]@git+https://github.com/minpeter/krill.git' --torch-backend=cu128
```

### For Preprocessing (CPU only)
```bash
uv pip install 'krill@git+https://github.im/minpeter/krill.git' --torch-backend=cpu
```

## Usage
The CLI commands can be invoked using either `krill` or `kr`.

### Example Commands (Refer to `src/krill/main.py` for full command list)
- `krill train <config_file.yaml>`: Initiates the training process using the specified configuration.
- `krill preprocess <config_file.yaml>`: Runs data preprocessing steps.
- `krill inspect-dataset <config_file.yaml>`: Inspects the dataset based on the configuration.

## Project Structure
- **`pyproject.toml`**: Defines project metadata, dependencies, and CLI entry points.
- **`src/krill/main.py`**: Contains the main CLI application logic and command dispatch using `click`.
- **`src/krill/utils/config.py`**: Defines the `Pydantic` schema for validating YAML configuration files.
- **`examples/`**: Contains example YAML configuration files for various tasks (e.g., `llama-pico-1k.yaml`, `qwen-micro-1k.yaml`).

## Configuration
Configuration for Krill is managed through YAML files. The schema for these configurations is defined using `Pydantic` in `src/krill/utils/config.py`. This ensures that all configurations are valid and adhere to the expected structure.

## Development Notes
- **Coding Style**: Adhere to PEP 8 guidelines. Use clear, descriptive variable and function names. Prioritize readability and maintainability.
- **Component Naming**: Follow consistent naming conventions for modules, classes, functions, and variables. For instance, modules should be lowercase with underscores, classes should be CapWords, and functions/variables should be lowercase with underscores.
- **Structured Functional Classification**: Code is organized into logical modules and sub-packages based on functionality (e.g., `utils`, `optimizer`). This promotes modularity and ease of navigation.
- **`uv` Usage**: `uv` is used for dependency management and installation due to its speed and efficiency. It ensures consistent and reproducible environments across different development setups.
- **README Updates**: After developing new features, always review `README.md` to determine if updates are necessary to reflect the changes and ensure documentation remains current.
- **Dependencies**: Managed via `pyproject.toml` and `uv.lock`.
- **Testing**: Refer to `test_memory.py`, `test_memory_efficient.py`, `test_preprocessing.py`, and `test_simple.py` for existing test patterns.
- **Contribution**: See `CONTRIBUTING.md` for guidelines.
