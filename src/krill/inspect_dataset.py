
from datasets import load_from_disk
from transformers import AutoTokenizer

from krill.utils.config import KrillConfig
from krill.utils.dataset_utils import inspect_pretrain_dataset


def do_inspect_dataset(config: KrillConfig):
    """Inspect a packed dataset based on the loaded Config object."""
    print(
        f"🦐 Krill: Starting to inspect packed dataset for {config.preprocess.prepared_path}...")

    # Load the preprocessed dataset
    dataset = load_from_disk(config.preprocess.prepared_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.hub_id)

    print(f"🦐 Krill: Loaded dataset with {len(dataset)} examples.")

    inspect_pretrain_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        show_example_rows_limit=5
    )
