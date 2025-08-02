
from datasets import load_from_disk
from transformers import AutoTokenizer

from krill.utils.config import load_config
from krill.utils.inspect_dataset import inspect_pretrain_dataset


def do_inspect_dataset(config_path: str):
    """Preprocesses the data based on the YAML config file."""
    # Load config centrally
    config = load_config(config_path)
    print(
        f"🦐 Krill: Starting to inspect packed dataset for {config.dataset_prepared_path}...")

    # Load the preprocessed dataset
    dataset = load_from_disk(config.dataset_prepared_path)
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)

    print(f"🦐 Krill: Loaded dataset with {len(dataset)} examples.")

    inspect_pretrain_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        show_example_rows_limit=5
    )
