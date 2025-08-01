import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import pack_dataset
from .config import load_config

def do_preprocess(config_path: str):
    """Preprocesses the data based on the YAML config file."""
    print(f"🚀 [Preprocess] Starting preprocessing with config: {config_path}")
    # Load config centrally
    config = load_config(config_path)
    # Extract settings from Pydantic model
    context_length = config.sequence_len
    tokenizer_id = config.hub_tokenizer_id
    save_path = config.dataset_prepared_path
    min_length = config.dataset_prepared_min_length
    ds_cfg = config.datasets[0]
    dataset_id = ds_cfg.path
    split = ds_cfg.split
    text_col = ds_cfg.text_column
    # Prepare output directory
    os.makedirs(save_path, exist_ok=True)
    # Load raw dataset
    print(f"Loading dataset {dataset_id} split={split}...")
    raw_datasets = load_dataset(dataset_id, split=split)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # Tokenize
    def tokenize_fn(examples):
        tokens = tokenizer(examples[text_col], padding=False, truncation=False)
        if tokenizer.eos_token_id is not None:
            for i in range(len(tokens["input_ids"])):
                tokens["input_ids"][i].append(tokenizer.eos_token_id)
                tokens["attention_mask"][i].append(1)
                if "token_type_ids" in tokens:
                    tokens["token_type_ids"][i].append(0)
        return tokens
    num_proc = max(1, os.cpu_count() - 1)
    print(f"Tokenizing with {num_proc} processes...")
    tokenized = raw_datasets.map(tokenize_fn, batched=True, num_proc=num_proc,
                                 remove_columns=raw_datasets.column_names)
    # Filter by min_length
    lengths = tokenized["input_ids"].map(len) if hasattr(tokenized["input_ids"], 'map') else [len(x) for x in tokenized["input_ids"]]
    selected = [i for i, l in enumerate(lengths) if l >= min_length]
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
    print(f"✅ [Preprocess] Finished. Packed data saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Krill Preprocessing Script")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()
    do_preprocess(args.config)

if __name__ == "__main__":
    main()