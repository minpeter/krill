import click
import subprocess
import sys

from transformers import __version__ as transformers_version
from triton import __version__ as triton_version

import torch

from krill import DEVICE_TYPE, DEVICE_COUNT, SUPPORTS_BFLOAT16, HAS_FLASH_ATTENTION, PLATFORM_SYSTEM


__version__ = "2025.8.2"


@click.group()
def cli():
    """A CLI tool for krill, inspired by axolotl and unsloth."""

    if DEVICE_TYPE == "cuda":
        gpu_stats = torch.cuda.get_device_properties(0)
        gpu_version = torch.version.cuda
        gpu_stats_snippet = f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {gpu_version}."
    else:
        raise ValueError(f"🦐 Krill: Unsupported device type: {DEVICE_TYPE}")

    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    statistics = \
        f"  /¯¯¯¯{chr(92)}   🦐 Krill {__version__}: A minimal pretraining trainer for LLMs — from scratch.\n"\
        f" ( #|{chr(92)}_ü|  {gpu_stats.name}. \033[1;43;30mNum GPUs = {DEVICE_COUNT}.\033[0m Max memory: {max_memory} GB. Platform: {PLATFORM_SYSTEM}.\n"\
        f" ( #{chr(92)}  ƒƒ  Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"\
        f"  {chr(92)} #{chr(92)}     Transformers: {transformers_version}. Bfloat16 = {SUPPORTS_BFLOAT16}. FA2 = {HAS_FLASH_ATTENTION}\n"\
        f'  /|||{chr(92)}    Source code: https://github.com/minpeter/krill\n'

    print(statistics)


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

    from krill.preprocess import do_preprocess
    do_preprocess(config)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def inspect_dataset(config: str):
    """Peek into the dataset after preprocessing."""

    from krill.inspect_dataset import do_inspect_dataset
    do_inspect_dataset(config)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
def train_tokenizer(config: str):
    """Train a tokenizer."""
    from krill.train_tokenizer import do_train_tokenizer
    do_train_tokenizer(config)


@cli.command()
def echo():
    """Echo the command line arguments. (For testing purposes.)"""
    print("🦐 Krill: EEeeeecccchooo!")


@cli.command()
@click.argument("model", type=str)
def inference(model: str):
    """Run interactive inference on a text generation model."""
    from krill.inference import do_inference
    do_inference(model)


@cli.command()
def evaluate():
    """Evaluate a model."""

    print("IMPLEMENT ME: Evaluate command is not yet implemented.")


@cli.command()
def purge():
    """Purge the cache and temporary files."""

    print("IMPLEMENT ME: Purge command is not yet implemented.")


def main():
    cli()
