import click
import subprocess
import sys

@click.group()
def cli():
    """A CLI tool for krill, inspired by axolotl."""
    print("🌊 Welcome to Krill CLI! 🌊")

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

    from .preprocess import do_preprocess
    do_preprocess(config)

def main():
    cli()