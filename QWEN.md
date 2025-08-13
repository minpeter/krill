# Qwen Code Context for `krill`

This document provides context for Qwen Code to effectively assist with the `krill` project.

## Project Overview

**Name:** krill
**Description:** 🦐 A minimal pretraining trainer for LLMs — from scratch.
**Purpose:** A simple, modular framework for training Large Language Models (LLMs) from the ground up. It handles preprocessing, tokenizer training, model training, and inference. It emphasizes simplicity, flexibility, and integration with Hugging Face tools.

**Core Technologies:**
*   **Language:** Python (>= 3.13)
*   **Core Libraries:** PyTorch, Hugging Face Transformers, Accelerate, Datasets, TRL, Flash Attention.
*   **Configuration:** YAML files validated with Pydantic.
*   **CLI:** Built with Click.
*   **Package Manager:** `uv` (implied by `pyproject.toml` and lock file).

## Project Structure

*   **`src/krill/`**: Main source code.
    *   **`main.py`**: Entry point for the CLI application (`krill` or `kr`).
    *   **`train.py`**: Core logic for model training using Hugging Face `Trainer`.
    *   **`preprocess.py`**: Logic for dataset preprocessing (tokenization, packing).
    *   **`inference.py`**: Interactive text generation.
    *   **`train_tokenizer.py`**: Logic for training/fine-tuning tokenizers.
    *   **`inspect_dataset.py`**: Utilities for inspecting processed datasets.
    *   **`evaluate.py`**: (Placeholder) Logic for model evaluation.
    *   **`utils/`**: Helper modules.
        *   `config.py`: Pydantic models for validating YAML configuration.
        *   `dataset_utils.py`: Helper functions for loading and inspecting datasets, including text cleaning and deduplication.
        *   `optimizer/`: Contains different implementations of the Muon optimizer.
            *   `index.py`: Main entry point to get the optimizer, selecting the implementation.
            *   `moonlight_muon.py`: Implementation from MoonlightAI.
        *   `inspect_model.py`: Utilities for inspecting model internals during inference (experimental).
        *   `memory_monitor.py`: Utility for monitoring memory usage during preprocessing.
*   **`examples/`**: Sample YAML configuration files.
*   **`artifacts/`**: Default directory for output (preprocessed data, trained models).
*   **`pyproject.toml`**: Project metadata, dependencies, and build system configuration.
*   **`uv.lock`**: Dependency lock file for `uv`.
*   **`README.md`**: Main documentation and quickstart guide.

## Key Concepts

1.  **Configuration Driven:** Almost all operations are driven by a YAML configuration file (e.g., `examples/tiny-ko-small-250802.yaml`). This file specifies datasets, tokenizer details, preprocessing parameters, and training hyperparameters. It is validated using Pydantic models defined in `src/krill/utils/config.py`.
2.  **Modular CLI:** The `krill` (or `kr`) command-line tool offers subcommands for different stages: `train`, `preprocess`, `inspect-dataset`, `train-tokenizer`, `inference`.
3.  **Hugging Face Integration:** Heavily relies on Hugging Face libraries (Transformers, Datasets, Accelerate, TRL) for core functionalities.
4.  **Accelerate for Training:** Model training (`krill train`) uses `accelerate launch` under the hood, allowing distributed training configurations to be passed via extra CLI arguments.
5.  **Flash Attention & Optimizations:** Includes conditional logic to enable Flash Attention and other performance optimizations (like Liger Kernel) if available.
6.  **Custom Optimizers:** Supports custom optimizers like Muon via `src/krill/utils/optimizer/index.py`. Different implementations (Moonlight, KellerJordan, pytorch_optimizer) can be selected.
7.  **Data Preprocessing:** Involves loading datasets, cleaning text, filtering by quality/length, deduplication, tokenization, and packing into fixed-length sequences. Utilizes multiprocessing for efficiency.

## Building, Running, and Development

### Installation

*   **Full Training Setup (with CUDA):**
    ```bash
    uv pip install 'krill[cuda]@git+https://github.com/minpeter/krill.git' --torch-backend=cu128
    ```
*   **Preprocessing Setup (CPU):**
    ```bash
    uv pip install 'krill@git+https://github.com/minpeter/krill.git' --torch-backend=cpu
    ```

### Usage (CLI Commands)

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

### Development

*   **Language:** Python 3.13+
*   **Dependency Management:** Uses `uv` with `pyproject.toml`.
*   **Entry Point:** The CLI is defined in `src/krill/main.py`.
*   **Main Training Logic:** Located in `src/krill/train.py`.
*   **Configuration Schema:** Defined by Pydantic models in `src/krill/utils/config.py`.
*   **Data Preprocessing:** Logic in `src/krill/preprocess.py` and utilities in `src/krill/utils/dataset_utils.py`.
*   **Optimizers:** Custom optimizer logic resides in `src/krill/utils/optimizer/`.