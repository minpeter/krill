"""
Utility functions for dataset loading.
"""

import os
import re
from datasets import load_dataset, concatenate_datasets


def clean_text(text):
    """
    Clean and normalize text data.
    """
    if not isinstance(text, str):
        return ""  # Return empty string if not a string or handle error

    # Remove whitespace from both ends of the string
    text = text.strip()

    # 0-1. Force UTF-8 validity check and remove invalid characters (added part)
    # This process removes invalid UTF-8 byte sequences.
    # In other words, it removes characters that cannot be re-encoded to UTF-8 within Python strings.
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        # This exception should not occur theoretically, but we log it just in case.
        print(
            f"Warning: Error during UTF-8 re-encoding/decoding: {e}. Original text: {text[:50]}...")
        text = ""  # Clear the text if an error occurs

    # 1-6. fix UnicodeEncodeError: 'utf-8' codec can't encode character '\udd2b' in position 2095: surrogates not allowed
    text = re.sub(r'[\uD800-\uDFFF]', '', text)

    return text


# Global set for deduplication
seen_texts = set()


def is_high_quality_and_unique(example):
    """
    Function that performs quality filtering (length) and deduplication simultaneously
    """
    text = example['text']

    # 2-1. Length filtering: reject if text length is less than 100 characters
    if len(text) < 100:
        return False

    # 2-2. Duplicate filtering: reject if text already appeared
    if text in seen_texts:
        return False

    # If it passes all filters, add to seen_texts and mark as passed
    seen_texts.add(text)
    return True


def load_and_prepare_raw_datasets(dataset_configs):
    """
    Load raw datasets, rename text column to 'text', drop other columns, concatenate,
    and apply text cleaning, quality filtering, and deduplication.
    Each config should have attributes 'path', 'split', and 'text_column'.
    """
    # 1. Load and concatenate raw datasets
    raw_datasets = []
    for ds_cfg in dataset_configs:
        print(
            f"1. Loading dataset {ds_cfg.path} columns={getattr(ds_cfg, 'text_column', 'text')} split={ds_cfg.split}...")
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

    # 2. Text cleaning and normalization
    num_processors = max(1, os.cpu_count() - 8)
    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_processors} processes for mapping.")

    print("\n2. Starting text cleaning and normalization... (.map)")
    cleaned_dataset = combined_dataset.map(
        lambda example: {'text': clean_text(example['text'])},
        num_proc=num_processors,
    )

    print(
        f"Cleaned dataset total rows: {cleaned_dataset.num_rows / 1_000_000:.2f}M")

    # 3. Quality filtering and deduplication
    # Initialize global seen_texts set
    global seen_texts
    seen_texts = set()

    print("\n3. Starting quality and duplicate filtering... (.filter)")
    final_dataset = cleaned_dataset.filter(
        is_high_quality_and_unique,
        # The 'seen_texts' set is a global variable so it may conflict during multiprocessing (num_proc > 1).
        num_proc=1
        # Other deduplication methods may be needed for large-scale data processing.
    )

    print(
        f"Final dataset total rows: {final_dataset.num_rows / 1_000_000:.4f}M")

    return final_dataset
