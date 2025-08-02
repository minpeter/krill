"""
Utility functions for dataset loading.
"""

from datasets import load_dataset, concatenate_datasets


def load_and_prepare_raw_datasets(dataset_configs):
    """
    Load raw datasets, rename text column to 'text', drop other columns, and concatenate.
    Each config should have attributes 'path', 'split', and 'text_column'.
    """
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
        return concatenate_datasets(raw_datasets)
    return raw_datasets[0]
