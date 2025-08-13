"""
Utility functions for dataset loading.
"""

import os
import re
from typing import Any
from datasets import Dataset, DatasetDict
from datasets import load_dataset, concatenate_datasets


def clean_text(text):
    """
    Clean and normalize text data.
    """
    if not isinstance(text, str):
        return ""

    # Remove whitespace from both ends
    text = text.strip()

    # Ensure UTF-8 validity by removing invalid byte sequences
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        print(
            f"Warning: Error during UTF-8 re-encoding/decoding: {e}. Original text: {text[:50]}...")
        text = ""

    # Remove surrogate pairs that can cause encoding issues
    text = re.sub(r'[\uD800-\uDFFF]', '', text)

    return text


# Global set for deduplication
seen_texts = set()


def is_high_quality_and_unique(example):
    """
    Perform quality filtering (length) and deduplication simultaneously.
    """
    text = example['text']

    # Reject if text length is less than 100 characters
    if len(text) < 100:
        return False

    # Reject if text already appeared (deduplication)
    if text in seen_texts:
        return False

    # Add to seen_texts and mark as passed
    seen_texts.add(text)
    return True


def load_and_prepare_raw_datasets(dataset_configs):
    """
    Load raw datasets, rename text column to 'text', drop other columns, concatenate,
    and apply text cleaning, quality filtering, and deduplication.
    Each config should have attributes 'path', 'split', and 'text_column'.
    """
    # Load and concatenate raw datasets
    raw_datasets = []
    for ds_cfg in dataset_configs:
        print(
            f"Loading dataset {ds_cfg.path} columns={getattr(ds_cfg, 'text_column', 'text')} split={ds_cfg.split}...")
        ds = load_dataset(ds_cfg.path, split=ds_cfg.split)
        if getattr(ds_cfg, 'text_column', 'text') != 'text':
            ds = ds.rename_column(ds_cfg.text_column, 'text')
        # Drop all columns except 'text'
        ds = ds.remove_columns(
            [col for col in ds.column_names if col != 'text'])
        raw_datasets.append(ds)

    if not raw_datasets:
        raise ValueError("No datasets to load.")

    if len(raw_datasets) > 1:
        combined_dataset = concatenate_datasets(raw_datasets)
    else:
        combined_dataset = raw_datasets[0]

    print(
        f"Combined dataset total rows: {combined_dataset.num_rows / 1_000_000:.2f}M")

    # Text cleaning and normalization
    num_processors = max(1, os.cpu_count() - 8)
    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_processors} processes for mapping.")

    print("Starting text cleaning and normalization... (.map)")
    cleaned_dataset = combined_dataset.map(
        lambda example: {'text': clean_text(example['text'])},
        num_proc=num_processors,
    )

    print(
        f"Cleaned dataset total rows: {cleaned_dataset.num_rows / 1_000_000:.2f}M")

    # Quality filtering and deduplication
    global seen_texts
    seen_texts = set()

    print("Starting quality and duplicate filtering... (.filter)")
    final_dataset = cleaned_dataset.filter(
        is_high_quality_and_unique,
        # The 'seen_texts' set is a global variable so it may conflict during multiprocessing (num_proc > 1).
        num_proc=1
        # Other deduplication methods may be needed for large-scale data processing.
    )

    print(
        f"Final dataset total rows: {final_dataset.num_rows / 1_000_000:.4f}M")

    return final_dataset


def inspect_pretrain_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer: Any,
    show_example_rows_limit: int = 3
):
    for i in range(min(show_example_rows_limit, len(dataset))):
        sample = dataset[i]

        colored_items = []
        # Display tokens, merging runs of undecodable (replacement char) tokens
        items = sample["input_ids"]
        idx = 0
        # \w already matches Unicode word characters (letters, digits, underscores)
        normal_pattern = re.compile(r"\w", flags=re.UNICODE)
        while idx < len(items):
            token_id = items[idx]
            token_str = tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False)
            # special tokens defined in tokenizer
            if token_id in tokenizer.all_special_ids:
                # yellow background, black text for special tokens
                colored_items.append(
                    f'\033[1;43;30m{token_str}\033[0m({token_id})')
                idx += 1
                continue
            # detect mergeable runs (non-word chars, excluding special tokens)
            if token_id not in tokenizer.all_special_ids and not normal_pattern.match(token_str):
                # gather run of mergeable tokens (skip special tokens)
                start = idx
                while idx < len(items):
                    next_id = items[idx]
                    next_str = tokenizer.decode(
                        [next_id], clean_up_tokenization_spaces=False)
                    if next_id not in tokenizer.all_special_ids and not normal_pattern.match(next_str):
                        idx += 1
                        continue
                    break
                run_ids = items[start:idx]
                run_str = tokenizer.decode(
                    run_ids, clean_up_tokenization_spaces=False)
                ids_str = ",".join(str(x) for x in run_ids)
                # magenta background for special runs
                colored_items.append(f'\033[45;97m{run_str}\033[0m({ids_str})')
                continue
            # normal token
            colored_items.append(f'\033[44;97m{token_str}\033[0m({token_id})')
            idx += 1
        print(
            f"\n\033[1;43;30m[PACKED SAMPLE {i}]\033[0m {' '.join(colored_items)}")
