import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import pack_dataset

from krill.config import load_config, DatasetConfig


def load_raw_datasets(dataset_config: DatasetConfig):
    print("Loading raw datasets...")
    for ds in dataset_config:
        print(f"Loading dataset {ds.path} columns={ds.text_column} split={ds.split}...")
        dataset = load_dataset(ds.path, split=ds.split)

        # Rename the text column to "text" for consistency
        if ds.text_column != "text":
            dataset = dataset.rename_column(ds.text_column, "text")
        
        # Drop all columns except "text"
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

        yield dataset
    

def do_preprocess(config_path: str):
    """Preprocesses the data based on the YAML config file."""
    print(f"🦐 Krill: Starting preprocessing with config: {config_path}")
    # Load config centrally
    config = load_config(config_path)
    # Extract settings from Pydantic model
    context_length = config.sequence_len
    save_path = config.dataset_prepared_path
    ds_cfg = config.datasets[0]
    dataset_id = ds_cfg.path
    split = ds_cfg.split
    text_col = ds_cfg.text_column
    
    # Prepare output directory
    os.makedirs(save_path, exist_ok=True)

    # Load raw dataset(s)
    raw_ds_list = list(load_raw_datasets(config.datasets))
    if not raw_ds_list:
        raise ValueError("No datasets found to preprocess. Check your config file.")
    # Combine datasets if multiple
    if len(raw_ds_list) > 1:
        from datasets import concatenate_datasets
        raw_dataset = concatenate_datasets(raw_ds_list)
    else:
        raw_dataset = raw_ds_list[0]

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
    lengths = tokenized["input_ids"].map(len) if hasattr(tokenized["input_ids"], 'map') else [len(x) for x in tokenized["input_ids"]]
    selected = [i for i, l in enumerate(lengths) if l >= config.dataset_prepared_min_length]
    tokenized = tokenized.select(selected)

    # Pack sequences
    print(f"Packing into sequences of length {context_length}...")
    packed = pack_dataset(tokenized, seq_length=context_length, strategy="wrapped",
                          map_kwargs={"batch_size": len(tokenized)})
    # Drop incomplete last chunk
    if len(packed) > 0 and len(packed[-1]["input_ids"]) < context_length:
        packed = packed.select(list(range(len(packed) - 1)))
    
    # Save
    packed.save_to_disk(save_path)
    print(f"🦐 Krill: Finished. Packed data saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Krill Preprocessing Script")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()
    do_preprocess(args.config)

if __name__ == "__main__":
    main()