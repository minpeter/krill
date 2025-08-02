import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import pack_dataset

from krill.config import load_config, DatasetConfig
from krill.libs.inspect_dataset import inspect_pretrain_dataset


def load_raw_datasets(dataset_config: DatasetConfig):
    print("Loading raw datasets...")
    for ds in dataset_config:
        print(
            f"Loading dataset {ds.path} columns={ds.text_column} split={ds.split}...")
        dataset = load_dataset(ds.path, split=ds.split)

        # Rename the text column to "text" for consistency
        if ds.text_column != "text":
            dataset = dataset.rename_column(ds.text_column, "text")

        # Drop all columns except "text"
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col != "text"])

        yield dataset


def do_preprocess(config_path: str):
    """Preprocesses the data based on the YAML config file."""
    print(f"🦐 Krill: Starting preprocessing with config: {config_path}")
    # Load config centrally
    config = load_config(config_path)

    # Prepare output directory
    os.makedirs(config.dataset_prepared_path, exist_ok=True)

    # Load and prepare raw datasets
    from krill.utils.dataset_utils import load_and_prepare_raw_datasets
    raw_dataset = load_and_prepare_raw_datasets(config.datasets)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"], padding=False, truncation=False)

        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
                if "token_type_ids" in tokenized_inputs:
                    tokenized_inputs["token_type_ids"][i].append(0)

        return tokenized_inputs

    num_proc = max(1, os.cpu_count() - 8)
    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_proc} processes for mapping.")

    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing"
    )

    # Filter by min_length
    lengths = tokenized["input_ids"].map(len) if hasattr(
        tokenized["input_ids"], 'map') else [len(x) for x in tokenized["input_ids"]]
    selected = [i for i, l in enumerate(
        lengths) if l >= config.dataset_prepared_min_length]
    # Compute token drop statistics from filtering
    total_tokens_before_filter = sum(lengths)
    total_tokens_after_filter = sum(lengths[i] for i in selected)
    filter_dropped_tokens = total_tokens_before_filter - total_tokens_after_filter
    tokenized = tokenized.select(selected)

    # Pack sequences
    print(f"Packing into sequences of length {config.sequence_len}...", end="")
    packed = pack_dataset(tokenized, seq_length=config.sequence_len, strategy="wrapped",
                          map_kwargs={"batch_size": len(tokenized)})
    # Drop incomplete last chunk and record dropped tokens
    last_dropped_chunk_length = 0
    if len(packed) > 0:
        last_len = len(packed[-1]["input_ids"])
        if last_len < config.sequence_len:
            last_dropped_chunk_length = last_len
            packed = packed.select(list(range(len(packed) - 1)))

    # Save
    packed.save_to_disk(config.dataset_prepared_path)

    inspect_pretrain_dataset(
        dataset=packed,
        tokenizer=tokenizer,
        show_example_rows_limit=1
    )

    print(
        f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from final incomplete chunk (length {last_dropped_chunk_length} < sequence_len={config.sequence_len})"
    )

    print(f"\nOriginal dataset rows: {len(raw_dataset)}")
    print(f"Packed dataset rows: {len(packed)}")

    if len(packed) > 0:
        # Check for any samples that do not match the expected context length
        wrong_indices = [
            i for i, sample in enumerate(packed)
            if len(sample["input_ids"]) != config.sequence_len
        ]
        if wrong_indices:
            print(
                f"\033[1;41;97mWarning: Found {len(wrong_indices)} samples "
                f"with incorrect length (expected {config.sequence_len}). "
                f"Indices: {wrong_indices}\033[0m"
            )
        else:
            print(
                "\033[1;32mAll packed samples have the correct context length.\033[0m")

    print(
        f"🦐 Krill: Finished. Packed data saved to {config.dataset_prepared_path}")

    print("""
To inspect the packed dataset, you can use the `peekdata` command:
\033[1;34m krill inspect-dataset {path}\033[0m
Or to train a model with this dataset, use:
\033[1;34m krill train {path}\033[0m
""".format(path=config_path))


def main():
    parser = argparse.ArgumentParser(description="Krill Preprocessing Script")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()
    do_preprocess(args.config)


if __name__ == "__main__":
    main()
