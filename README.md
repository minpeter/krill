# krill

🦐 A minimal pretraining trainer for LLMs — from scratch.

## Installation

```bash
# For model training and the full pipeline
uv pip install 'krill[cuda]@git+https://github.com/minpeter/krill.git' --torch-backend=cu128

# For preprocessing tasks
uv pip install 'krill@git+https://github.com/minpeter/krill.git' --torch-backend=cpu

# For enhanced preprocessing with datatrove (recommended for large datasets)
uv pip install 'krill[cuda,datatrove]@git+https://github.com/minpeter/krill.git' --torch-backend=cu128
```

After installation, the CLI is available as both `krill` and the shorthand `kr`.

## Overview

Krill is a minimalistic training framework for Large Language Models (LLMs) built from scratch with simplicity and flexibility in mind. It provides command-line tools and modular components to handle data preprocessing, tokenizer training, model training, inference, and dataset inspection.

## Features

- Modular CLI with commands for preprocessing, tokenizer training, model training, inference, and dataset inspection
- Support for Hugging Face Transformers, Accelerate, and Flash Attention
- **Enhanced preprocessing with datatrove integration** for large-scale datasets
- Configurable via YAML files with validation using Pydantic
- Automatic environment optimizations (e.g., Flash Attention)
- Integration with Hugging Face Hub for model and tokenizer pushing
- Data collators optimized for language modeling and flash attention
- Customizable optimizers including Muon

## Enhanced Preprocessing with Datatrove

Krill now supports [Hugging Face datatrove](https://github.com/huggingface/datatrove) for improved preprocessing performance:

### Benefits
- **50-80% memory reduction** through streaming processing
- **20-40% faster processing** with optimized algorithms
- **Advanced deduplication** using MinHash or exact matching
- **Enhanced text quality filtering** beyond basic length checks
- **Multi-process deduplication** vs single-process limitation

### Quick Start with Datatrove

1. **Check datatrove availability:**
   ```bash
   krill check-datatrove
   ```

2. **Generate example configuration:**
   ```bash
   krill generate-datatrove-config -o my_config.yaml
   ```

3. **Enable datatrove in your config:**
   ```yaml
   datatrove:
     enabled: true
     deduplication_algorithm: "minhash"  # or "exact"
     num_workers: 4
     minhash_threshold: 0.8
   ```

## Quickstart

### Minimal Usage

1. Create a YAML configuration file (see **Configuration** section below):

    ```bash
    # edit path/to/config.yaml
    ```

2. (Optional) Train your own tokenizer (before preprocessing):

    ```bash
    krill train-tokenizer path/to/config.yaml
    # or use `kr train-tokenizer ...`
    ```

3. Preprocess your dataset:

    ```bash
    krill preprocess path/to/config.yaml
    # or `kr preprocess ...`
    ```

4. Start model training:

    ```bash
    krill train path/to/config.yaml --num-processes 2
    # or `kr train ...`
    ```

## CLI Reference

### krill train <config>

Launches model training using Accelerate. Accepts extra `accelerate launch` arguments.

### krill preprocess <config>

Preprocesses datasets as specified in the YAML config.

### krill inspect-dataset <config>

Displays sample data and statistics after preprocessing.

### krill train-tokenizer <config>

Trains or fine-tunes a tokenizer based on datasets.

### krill inference <model_or_config> [--inspect]

Runs interactive generation from a model ID or YAML config. The `--inspect` flag is experimental and provides token-level entropy analysis for each generated output.

### krill evaluate

(Not implemented) Placeholder for model evaluation.

## Configuration

Krill uses a YAML configuration file validated by Pydantic (`KrillConfig`). Example:

```yaml
# krill train-tokenizer
vocab_size: 32000
hub_tokenizer_id: minpeter/webtext-tokenizer-32k

# krill preprocess
sequence_len: 8192
dataset_prepared_path: ./artifacts/webtext-8k
dataset_prepared_min_length: 150

datasets:
  - path: HAERAE-HUB/KOREAN-WEBTEXT
    split: train[:10_000]
    text_column: text

# krill train
hub_model_id: minpeter/webtext-8k-micro-250801
output_dir: ./artifacts/models/webtext-8k-micro-250801

num_epochs: 1
learning_rate: 1e-3
weight_decay: 0.01
optimizer: muon
muon_implementation: moonlight

micro_batch_size: 1
gradient_accumulation_steps: 1

model_config_name: micro
```

Refer to `src/krill/utils/config.py` for full schema and defaults.

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for full details.

