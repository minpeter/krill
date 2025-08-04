import click
import subprocess
import sys
import os
from krill import HAS_FLASH_ATTENTION, get_statistics
from krill.utils import patch_optimized_env


@click.group()
def cli():
    """A CLI tool for krill, inspired by axolotl and unsloth."""

    patch_optimized_env()
    print(get_statistics())


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("config", type=click.Path(exists=True))
@click.pass_context
def train(ctx, config: str):
    """
    Train a model using accelerate.

    Any extra arguments (e.g., --num_processes=2) are passed to 'accelerate launch'.
    """
    # Restrict training to CUDA-enabled GPUs

    if not HAS_FLASH_ATTENTION:
        print(
            "\033[1;41;97mWarning: For CUDA-enabled environments,\033[0m please run: pip install 'krill[cuda] @ git+https://github.com/minpeter/krill.git'", file=sys.stderr)

    print("Preparing to launch training with accelerate...")
    # Main command: accelerate launch
    base_cmd = ["accelerate", "launch"]

    # Additional arguments for accelerate launch (e.g., --num_processes 2)
    # These are stored in ctx.args.
    accelerate_args = ctx.args

    # Script module to run and its arguments
    script_cmd = ["-m", "krill.train", config]

    # Assemble the final command
    final_cmd = base_cmd + accelerate_args + script_cmd

    print(f"Executing command: {' '.join(final_cmd)}")

    try:
        # Use subprocess to run accelerate launch
        subprocess.run(final_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def preprocess(config: str):
    """Preprocess a dataset."""
    from krill.utils.config import load_config
    from krill.preprocess import do_preprocess
    cfg = load_config(config)
    do_preprocess(cfg)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def inspect_dataset(config: str):
    """Peek into the dataset after preprocessing."""
    from krill.utils.config import load_config
    from krill.inspect_dataset import do_inspect_dataset
    cfg = load_config(config)
    do_inspect_dataset(cfg)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def train_tokenizer(config: str):
    """Train a tokenizer."""
    from krill.utils.config import load_config
    from krill.train_tokenizer import do_train_tokenizer
    cfg = load_config(config)
    do_train_tokenizer(cfg)


@cli.command()
def echo():
    """Echo the command line arguments. (For testing purposes.)"""
    print("🦐 Krill: EEeeeecccchooo!")


@cli.command()
@click.argument("model", type=str)
@click.option("--inspect", is_flag=True, default=False, help="Enable inspect mode (experimental)")
def inference(model: str, inspect: bool):
    """Run interactive inference on a text generation model or a YAML config file."""

    model_id = model
    # If a YAML config is passed, load hub_model_id
    if os.path.isfile(model) and model.lower().endswith((".yaml", ".yml")):
        print(f"⚓️ Loading config file: {model}...")
        try:
            import yaml
            with open(model, "r") as f:
                cfg = yaml.safe_load(f)
            model_id = cfg.get("hub_model_id")
            if not model_id:
                raise KeyError("hub_model_id not found in config")
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"⚓️ Using model from config: {model_id}...")

    # Delegate to core inference logic
    from krill.inference import do_inference
    do_inference(model_id, inspect)


@cli.command()
def evaluate():
    """Evaluate a model."""

    print("IMPLEMENT ME: Evaluate command is not yet implemented.")


@cli.command()
@click.option("--output", "-o", default="krill_config_datatrove.yaml", help="Output configuration file path")
def generate_datatrove_config(output: str):
    """Generate an example configuration file with datatrove settings."""
    import yaml
    from krill.utils.datatrove_utils import create_datatrove_example_config, is_datatrove_available
    
    if not is_datatrove_available():
        print("⚠️  Datatrove is not installed. Install with: pip install 'krill[datatrove]'")
        print("Generating config anyway for reference...")
    
    # Create a complete example configuration
    example_config = {
        "sequence_len": 2048,
        "vocab_size": 32000,
        "hub_tokenizer_id": "huggingface/tokenizers-base",
        "dataset_prepared_path": "./prepared_data",
        "dataset_prepared_min_length": 150,
        "datasets": [
            {
                "path": "your/dataset/path",
                "split": "train",
                "text_column": "text"
            }
        ],
        "hub_model_id": "your-model-id",
        "output_dir": "./output",
        "num_epochs": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "optimizer": "muon",
        "muon_implementation": "moonlight",
        "model_config_name": "small",
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1
    }
    
    # Add datatrove configuration
    example_config.update(create_datatrove_example_config())
    
    # Write to file
    with open(output, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Example configuration with datatrove settings written to: {output}")
    print("\nTo use datatrove preprocessing:")
    print("1. Install datatrove: pip install 'krill[datatrove]'")
    print("2. Edit the configuration file to match your datasets")
    print("3. Set datatrove.enabled: true")
    print("4. Run: krill preprocess your_config.yaml")


@cli.command()
def check_datatrove():
    """Check datatrove installation and availability."""
    from krill.utils.datatrove_utils import is_datatrove_available
    
    if is_datatrove_available():
        print("✅ Datatrove is available and ready to use!")
        print("You can enable it by setting 'datatrove.enabled: true' in your config.")
        
        # Show version if possible
        try:
            import datatrove
            print(f"Datatrove version: {datatrove.__version__}")
        except (ImportError, AttributeError):
            pass
    else:
        print("❌ Datatrove is not available.")
        print("Install with: pip install 'krill[datatrove]' or pip install datatrove>=0.2.0")
        print("\nDatatrove provides:")
        print("- 50-80% memory reduction through streaming")
        print("- 20-40% faster processing")
        print("- Advanced deduplication algorithms")
        print("- Enhanced text quality filtering")


if __name__ == "__main__":
    cli()
