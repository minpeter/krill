import click
import subprocess
import sys
import os


@click.group()
def cli():
    """A CLI tool for krill, inspired by axolotl and unsloth."""

    pass


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("config", type=click.Path(exists=True))
@click.pass_context
def train(ctx, config: str):
    """
    Train a model using accelerate.

    Any extra arguments (e.g., --num_processes=2) are passed to 'accelerate launch'.
    """
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


def main():
    cli()
