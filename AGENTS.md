# Project: Krill

## Overview
Krill is a Python CLI tool designed for pre-training Large Language Models (LLMs). It leverages `click` for its command-line interface, `Pydantic` for robust configuration validation, and `PyTorch`, `Hugging Face Transformers`, and `Accelerate` for efficient LLM-related tasks. It emphasizes simplicity, flexibility, and integration with Hugging Face tools.

**Core Technologies:**
*   **Language:** Python (>= 3.13)
*   **Core Libraries:** PyTorch, Hugging Face Transformers, Accelerate, Datasets, TRL, Flash Attention
*   **Configuration:** YAML files validated with Pydantic
*   **CLI:** Built with Click
*   **Package Manager:** `uv` (implied by `pyproject.toml` and lock file)

## Key Features
- **LLM Pre-training**: Facilitates the pre-training of large language models, handling preprocessing, tokenizer training, model training, and inference
- **CLI Interface**: User-friendly command-line interface built with `click`. The CLI commands can be invoked using either `krill` or `kr`
- **Configurable**: Utilizes YAML files for flexible configuration, with schema validation provided by `Pydantic`
- **Deep Learning Stack**: Built on `PyTorch`, integrated with `Hugging Face Transformers` for model architectures, `Accelerate` for distributed training, `Datasets` for data loading, and `TRL` for potential RLHF tasks
- **Performance Optimizations**: Includes conditional logic to enable Flash Attention and other performance optimizations (like Liger Kernel) if available
- **Custom Optimizers**: Supports custom optimizers like Muon via `src/krill/utils/optimizer/index.py`. Different implementations (Moonlight, KellerJordan, pytorch_optimizer) can be selected
- **Data Preprocessing**: Involves loading datasets, cleaning text, filtering by quality/length, deduplication, tokenization, and packing into fixed-length sequences. Utilizes multiprocessing for efficiency

## Installation

### For Training (with CUDA support)
```bash
uv pip install 'krill[cuda]@git+https://github.com/minpeter/krill.git' --torch-backend=cu128
```

### For Preprocessing (CPU only)
```bash
uv pip install 'krill@git+https://github.com/minpeter/krill.git' --torch-backend=cpu
```

## Usage (CLI Commands)

*   **Train a Model:**
    ```bash
    krill train path/to/config.yaml [--num_processes 2 ...]
    # Extra arguments after config.yaml are passed to 'accelerate launch'
    ```
*   **Preprocess Dataset:**
    ```bash
    krill preprocess path/to/config.yaml
    ```
*   **Inspect Preprocessed Dataset:**
    ```bash
    krill inspect-dataset path/to/config.yaml
    ```
*   **Train a Tokenizer:**
    ```bash
    krill train-tokenizer path/to/config.yaml
    ```
*   **Run Interactive Inference:**
    ```bash
    # From a model on Hugging Face Hub
    krill inference model_id_on_hub

    # From a local model directory
    krill inference /path/to/local/model

    # From a config file (uses hub_model_id from config)
    krill inference path/to/config.yaml
    ```
*   **Evaluate a Model (Not Implemented):**
    ```bash
    krill evaluate model_id_or_config
    ```

## Project Structure
*   **`pyproject.toml`**: Defines project metadata, dependencies, and CLI entry points
*   **`src/krill/main.py`**: Contains the main CLI application logic and command dispatch using `click`. Entry point for the CLI application (`krill` or `kr`)
*   **`src/krill/train.py`**: Core logic for model training using Hugging Face `Trainer`
*   **`src/krill/preprocess.py`**: Logic for dataset preprocessing (tokenization, packing)
*   **`src/krill/inference.py`**: Interactive text generation
*   **`src/krill/train_tokenizer.py`**: Logic for training/fine-tuning tokenizers
*   **`src/krill/inspect_dataset.py`**: Utilities for inspecting processed datasets
*   **`src/krill/evaluate.py`**: (Placeholder) Logic for model evaluation
*   **`src/krill/utils/`**: Helper modules
    *   `config.py`: Pydantic models for validating YAML configuration
    *   `dataset_utils.py`: Helper functions for loading and inspecting datasets, including text cleaning and deduplication
    *   `optimizer/`: Contains different implementations of the Muon optimizer
        *   `index.py`: Main entry point to get the optimizer, selecting the implementation
        *   `moonlight_muon.py`: Implementation from MoonlightAI
    *   `inspect_model.py`: Utilities for inspecting model internals during inference (experimental)
    *   `memory_monitor.py`: Utility for monitoring memory usage during preprocessing
*   **`examples/`**: Sample YAML configuration files
*   **`artifacts/`**: Default directory for output (preprocessed data, trained models)

## Configuration
Configuration for Krill is managed through YAML files. The schema for these configurations is defined using `Pydantic` in `src/krill/utils/config.py`. This ensures that all configurations are valid and adhere to the expected structure. Almost all operations are driven by a YAML configuration file (e.g., `examples/tiny-ko-small-250802.yaml`). This file specifies datasets, tokenizer details, preprocessing parameters, and training hyperparameters.

## Development Notes
- **Coding Style**: Adhere to PEP 8 guidelines. Use clear, descriptive variable and function names. Prioritize readability and maintainability
- **Component Naming**: Follow consistent naming conventions for modules, classes, functions, and variables. For instance, modules should be lowercase with underscores, classes should be CapWords, and functions/variables should be lowercase with underscores
- **Structured Functional Classification**: Code is organized into logical modules and sub-packages based on functionality (e.g., `utils`, `optimizer`). This promotes modularity and ease of navigation
- **`uv` Usage**: `uv` is used for dependency management and installation due to its speed and efficiency. It ensures consistent and reproducible environments across different development setups
- **README Updates**: After developing new features, always review `README.md` to determine if updates are necessary to reflect the changes and ensure documentation remains current
- **GitHub CLI (`gh`)**: If available in your environment, the `gh` CLI tool can be used to streamline common GitHub workflows, such as creating pull requests, managing issues, and interacting with repositories directly from the command line
- **Dependencies**: Managed via `pyproject.toml` and `uv.lock`
- **Testing**: Refer to `test_memory.py`, `test_memory_efficient.py`, `test_preprocessing.py`, and `test_simple.py` for existing test patterns
- **Contribution**: See `CONTRIBUTING.md` for guidelines