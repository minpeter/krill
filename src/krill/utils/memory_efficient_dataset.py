"""
Memory-efficient dataset processing utilities.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Set, Optional
from datasets import Dataset, IterableDataset, load_dataset, concatenate_datasets


class FileBasedDeduplicator:
    """File-based deduplication to avoid keeping all hashes in memory."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="krill_dedup_"))
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: Set[str] = set()
        self._load_existing_hashes()
    
    def _get_hash_file(self, hash_value: str) -> Path:
        """Get the file path for a hash value."""
        prefix = hash_value[:2]
        hash_dir = self.cache_dir / prefix
        hash_dir.mkdir(exist_ok=True)
        return hash_dir / hash_value
    
    def _load_existing_hashes(self):
        """Load existing hashes from files into memory for faster lookup."""
        if not self.cache_dir.exists():
            return
        
        for prefix_dir in self.cache_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for hash_file in prefix_dir.iterdir():
                    if hash_file.is_file():
                        self.seen_hashes.add(hash_file.name)
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True
        
        # Check if hash file exists
        hash_file = self._get_hash_file(text_hash)
        if hash_file.exists():
            self.seen_hashes.add(text_hash)
            return True
        
        # Not a duplicate, create the hash file
        hash_file.touch()
        self.seen_hashes.add(text_hash)
        return False
    
    def cleanup(self):
        """Clean up the cache directory."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


def clean_text_memory_efficient(text):
    """Memory-efficient text cleaning function."""
    import re
    
    if not isinstance(text, str):
        return ""
    
    # Remove whitespace from both ends
    text = text.strip()
    
    # UTF-8 validity check
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        print(f"Warning: Error during UTF-8 re-encoding/decoding: {e}")
        text = ""
    
    # Fix surrogate characters
    text = re.sub(r'[\uD800-\uDFFF]', '', text)
    
    return text


def is_high_quality_and_unique_memory_efficient(example, deduplicator: FileBasedDeduplicator):
    """Memory-efficient quality filtering and deduplication."""
    text = example['text']
    
    # Length filtering
    if len(text) < 100:
        return False
    
    # Duplicate filtering using file-based deduplicator
    if deduplicator.is_duplicate(text):
        return False
    
    return True


def load_dataset_chunked(dataset_config, chunk_size: int = 10000):
    """Load dataset in chunks to reduce memory usage."""
    print(f"Loading dataset {dataset_config.path} in chunks of {chunk_size}...")
    
    # Load the full dataset first to get total size
    full_dataset = load_dataset(dataset_config.path, split=dataset_config.split)
    
    # Rename text column if needed
    if getattr(dataset_config, 'text_column', 'text') != 'text':
        full_dataset = full_dataset.rename_column(dataset_config.text_column, 'text')
    
    # Keep only text column
    full_dataset = full_dataset.remove_columns(
        [col for col in full_dataset.column_names if col != 'text']
    )
    
    total_size = len(full_dataset)
    print(f"Dataset {dataset_config.path} has {total_size:,} samples")
    
    # Yield chunks
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        chunk = full_dataset.select(range(i, end_idx))
        yield chunk, i, end_idx, total_size


def process_datasets_memory_efficient(dataset_configs, chunk_size: int = 10000, 
                                   cache_dir: Optional[str] = None):
    """Process datasets in a memory-efficient way."""
    deduplicator = FileBasedDeduplicator(cache_dir)
    processed_chunks = []
    
    try:
        total_processed = 0
        total_filtered = 0
        
        for ds_cfg in dataset_configs:
            print(f"\nProcessing dataset: {ds_cfg.path}")
            
            for chunk, start_idx, end_idx, total_size in load_dataset_chunked(ds_cfg, chunk_size):
                print(f"Processing chunk {start_idx:,}-{end_idx:,}/{total_size:,}")
                
                # Clean text
                num_processors = max(1, os.cpu_count() - 8)
                cleaned_chunk = chunk.map(
                    lambda example: {'text': clean_text_memory_efficient(example['text'])},
                    num_proc=num_processors,
                    desc="Cleaning text"
                )
                
                # Filter for quality and duplicates
                filtered_chunk = cleaned_chunk.filter(
                    lambda example: is_high_quality_and_unique_memory_efficient(example, deduplicator),
                    num_proc=1,  # File-based deduplication requires single process
                    desc="Quality filtering and deduplication"
                )
                
                chunk_processed = len(cleaned_chunk)
                chunk_filtered = len(filtered_chunk)
                total_processed += chunk_processed
                total_filtered += chunk_filtered
                
                print(f"Chunk: {chunk_processed:,} -> {chunk_filtered:,} samples "
                      f"({chunk_filtered/chunk_processed*100:.1f}% kept)")
                
                if len(filtered_chunk) > 0:
                    processed_chunks.append(filtered_chunk)
        
        print(f"\nTotal: {total_processed:,} -> {total_filtered:,} samples "
              f"({total_filtered/total_processed*100:.1f}% kept)")
        
        # Concatenate all processed chunks
        if processed_chunks:
            final_dataset = concatenate_datasets(processed_chunks)
            print(f"Final concatenated dataset: {len(final_dataset):,} samples")
            return final_dataset
        else:
            print("No samples passed filtering")
            return Dataset.from_dict({"text": []})
    
    finally:
        # Clean up deduplication cache if using temporary directory
        if cache_dir is None:
            deduplicator.cleanup()


def tokenize_in_chunks(dataset, tokenizer, chunk_size: int = 1000):
    """Tokenize dataset in chunks to reduce memory usage."""
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"],
                                   padding=False,
                                   truncation=False)
        
        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
                if "token_type_ids" in tokenized_inputs:
                    tokenized_inputs["token_type_ids"][i].append(0)
        
        return tokenized_inputs
    
    total_size = len(dataset)
    tokenized_chunks = []
    
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        chunk = dataset.select(range(i, end_idx))
        
        print(f"Tokenizing chunk {i:,}-{end_idx:,}/{total_size:,}")
        
        num_proc = max(1, os.cpu_count() - 8)
        tokenized_chunk = chunk.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=chunk.column_names,
            desc=f"Tokenizing chunk {i//chunk_size + 1}"
        )
        
        tokenized_chunks.append(tokenized_chunk)
    
    # Concatenate tokenized chunks
    if tokenized_chunks:
        return concatenate_datasets(tokenized_chunks)
    else:
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})


def filter_by_length_memory_efficient(dataset, min_length: int, chunk_size: int = 10000):
    """Filter dataset by length in a memory-efficient way."""
    total_size = len(dataset)
    filtered_chunks = []
    total_tokens_before = 0
    total_tokens_after = 0
    
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        chunk = dataset.select(range(i, end_idx))
        
        print(f"Filtering chunk {i:,}-{end_idx:,}/{total_size:,} by length")
        
        # Calculate lengths for this chunk
        lengths = [len(sample["input_ids"]) for sample in chunk]
        chunk_tokens_before = sum(lengths)
        
        # Filter by length
        selected_indices = [idx for idx, length in enumerate(lengths) if length >= min_length]
        filtered_chunk = chunk.select(selected_indices)
        
        chunk_tokens_after = sum(len(sample["input_ids"]) for sample in filtered_chunk)
        
        total_tokens_before += chunk_tokens_before
        total_tokens_after += chunk_tokens_after
        
        print(f"Chunk: {len(chunk):,} -> {len(filtered_chunk):,} samples, "
              f"{chunk_tokens_before:,} -> {chunk_tokens_after:,} tokens")
        
        if len(filtered_chunk) > 0:
            filtered_chunks.append(filtered_chunk)
    
    filter_dropped_tokens = total_tokens_before - total_tokens_after
    print(f"Total filtered: {filter_dropped_tokens:,} tokens dropped")
    
    if filtered_chunks:
        final_dataset = concatenate_datasets(filtered_chunks)
        return final_dataset, filter_dropped_tokens
    else:
        return Dataset.from_dict({"input_ids": [], "attention_mask": []}), filter_dropped_tokens