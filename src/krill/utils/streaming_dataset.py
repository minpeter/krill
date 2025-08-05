"""
Truly streaming memory-efficient dataset processing.
This implementation never loads the entire dataset into memory.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Set, Optional, Iterator, Dict, Any
from datasets import Dataset, load_dataset, IterableDataset
from transformers import PreTrainedTokenizer
from trl import pack_dataset


class StreamingFileBasedDeduplicator:
    """File-based deduplication for streaming processing."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="krill_streaming_dedup_"))
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Use smaller in-memory cache for streaming
        self.seen_hashes: Set[str] = set()
        self.cache_size_limit = 10000  # Keep only recent hashes in memory
        self._load_existing_hashes()
    
    def _get_hash_file(self, hash_value: str) -> Path:
        """Get the file path for a hash value."""
        prefix = hash_value[:2]
        hash_dir = self.cache_dir / prefix
        hash_dir.mkdir(exist_ok=True)
        return hash_dir / hash_value
    
    def _load_existing_hashes(self):
        """Load a sample of existing hashes into memory for faster lookup."""
        if not self.cache_dir.exists():
            return
        
        # Load only a sample to keep memory usage low
        loaded = 0
        for prefix_dir in self.cache_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for hash_file in prefix_dir.iterdir():
                    if hash_file.is_file() and loaded < self.cache_size_limit:
                        self.seen_hashes.add(hash_file.name)
                        loaded += 1
                    if loaded >= self.cache_size_limit:
                        break
            if loaded >= self.cache_size_limit:
                break
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Check in-memory cache first
        if text_hash in self.seen_hashes:
            return True
        
        # Check if hash file exists
        hash_file = self._get_hash_file(text_hash)
        if hash_file.exists():
            # Add to in-memory cache if there's space
            if len(self.seen_hashes) < self.cache_size_limit:
                self.seen_hashes.add(text_hash)
            return True
        
        # Not a duplicate, create the hash file
        hash_file.touch()
        # Add to in-memory cache if there's space
        if len(self.seen_hashes) < self.cache_size_limit:
            self.seen_hashes.add(text_hash)
        return False
    
    def cleanup(self):
        """Clean up the cache directory."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


def clean_text_streaming(text: str) -> str:
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


def is_high_quality_streaming(text: str, min_length: int = 100) -> bool:
    """Basic quality filtering for streaming."""
    return len(text) >= min_length


def stream_dataset_batches(dataset_config, batch_size: int = 1000):
    """Stream dataset in small batches without loading everything."""
    print(f"Streaming dataset {dataset_config.path} in batches of {batch_size}...")
    
    # Parse the split to handle slicing
    split_str = dataset_config.split
    base_split = 'train'
    start_idx = 0
    max_samples = None
    
    # Handle splits like "train[:500]" or "train[100:500]"
    if '[' in split_str and ']' in split_str:
        base_split = split_str.split('[')[0]
        slice_part = split_str.split('[')[1].split(']')[0]
        
        if ':' in slice_part:
            parts = slice_part.split(':')
            if parts[0]:  # start index specified
                start_idx = int(parts[0])
            if parts[1]:  # end index specified
                max_samples = int(parts[1]) - start_idx
            else:
                max_samples = None  # Open-ended slice
        else:
            # Single index (treat as [:index])
            max_samples = int(slice_part)
    
    # Use streaming mode to avoid loading full dataset
    if hasattr(dataset_config, 'name') and dataset_config.name:
        dataset = load_dataset(
            dataset_config.path, 
            dataset_config.name,
            split=base_split,
            streaming=True  # This is the key - streaming mode!
        )
    else:
        dataset = load_dataset(
            dataset_config.path, 
            split=base_split,
            streaming=True  # This is the key - streaming mode!
        )
    
    # Rename text column if needed
    if getattr(dataset_config, 'text_column', 'text') != 'text':
        dataset = dataset.rename_column(dataset_config.text_column, 'text')
    
    # Keep only text column
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    
    # Skip samples if start_idx > 0
    if start_idx > 0:
        print(f"Skipping first {start_idx} samples...")
        dataset = dataset.skip(start_idx)
    
    # Take only the required number of samples if specified
    if max_samples is not None:
        print(f"Taking {max_samples} samples from stream...")
        dataset = dataset.take(max_samples)
    
    # Collect samples in batches
    batch = []
    batch_num = 0
    total_processed = 0
    
    for sample in dataset:
        batch.append(sample)
        total_processed += 1
        
        if len(batch) >= batch_size:
            yield batch, batch_num, total_processed
            batch = []
            batch_num += 1
        
        # Stop if we've reached the max_samples limit
        if max_samples is not None and total_processed >= max_samples:
            break
    
    # Yield final batch if it has samples
    if batch:
        yield batch, batch_num, total_processed


def process_batch_streaming(batch: list, deduplicator: StreamingFileBasedDeduplicator, 
                          min_length: int = 100) -> list:
    """Process a batch of samples with cleaning, quality filtering, and deduplication."""
    processed_batch = []
    
    for sample in batch:
        # Clean text
        cleaned_text = clean_text_streaming(sample['text'])
        
        # Quality filtering
        if not is_high_quality_streaming(cleaned_text, min_length):
            continue
        
        # Deduplication
        if deduplicator.is_duplicate(cleaned_text):
            continue
        
        processed_batch.append({'text': cleaned_text})
    
    return processed_batch


def tokenize_batch_streaming(batch: list, tokenizer: PreTrainedTokenizer) -> list:
    """Tokenize a batch of samples."""
    if not batch:
        return []
    
    texts = [sample['text'] for sample in batch]
    
    tokenized_inputs = tokenizer(
        texts,
        padding=False,
        truncation=False
    )
    
    # Add EOS tokens
    if tokenizer.eos_token_id is not None:
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"][i].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"][i].append(0)
    
    # Convert to list of samples
    tokenized_batch = []
    for i in range(len(tokenized_inputs["input_ids"])):
        sample = {
            "input_ids": tokenized_inputs["input_ids"][i],
            "attention_mask": tokenized_inputs["attention_mask"][i]
        }
        if "token_type_ids" in tokenized_inputs:
            sample["token_type_ids"] = tokenized_inputs["token_type_ids"][i]
        tokenized_batch.append(sample)
    
    return tokenized_batch


def filter_batch_by_length_streaming(batch: list, min_length: int) -> tuple[list, int]:
    """Filter batch by minimum token length."""
    filtered_batch = []
    dropped_tokens = 0
    
    for sample in batch:
        length = len(sample["input_ids"])
        if length >= min_length:
            filtered_batch.append(sample)
        else:
            dropped_tokens += length
    
    return filtered_batch, dropped_tokens


def pack_batch_streaming(batch: list, sequence_len: int) -> tuple[list, int]:
    """Pack a batch of tokenized samples."""
    if not batch:
        return [], 0
    
    # Convert to Dataset for TRL packing
    batch_dataset = Dataset.from_list(batch)
    
    # Pack with small batch size to minimize memory
    packed = pack_dataset(
        batch_dataset,
        seq_length=sequence_len,
        strategy="wrapped",
        map_kwargs={"batch_size": min(len(batch), 100)}
    )
    
    # Convert back to list and filter incomplete samples
    packed_samples = []
    dropped_tokens = 0
    
    for sample in packed:
        if len(sample["input_ids"]) == sequence_len:
            packed_samples.append(sample)
        else:
            dropped_tokens += len(sample["input_ids"])
    
    return packed_samples, dropped_tokens


def save_batch_to_disk(batch: list, output_dir: str, batch_num: int, shard_size: str = "50MB"):
    """Save a batch of packed samples to disk."""
    if not batch:
        return
    
    # Create batch dataset
    batch_dataset = Dataset.from_list(batch)
    
    # Create shard directory
    shard_dir = os.path.join(output_dir, f"shard_{batch_num:06d}")
    os.makedirs(shard_dir, exist_ok=True)
    
    # Save with controlled shard size
    batch_dataset.save_to_disk(shard_dir, max_shard_size=shard_size)


def combine_saved_shards(output_dir: str, final_output_dir: str, max_shard_size: str = "200MB"):
    """Combine all saved shard datasets into final dataset."""
    import shutil
    from datasets import concatenate_datasets, Dataset
    
    print("Combining saved shards into final dataset...")
    
    # Find all shard directories
    shard_dirs = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.startswith("shard_"):
                shard_dirs.append(item_path)
    
    if not shard_dirs:
        print("No shards found to combine")
        # Return empty dataset
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    
    shard_dirs.sort()
    print(f"Found {len(shard_dirs)} shards to combine")
    
    # Load and combine shards in smaller groups to control memory
    datasets_to_combine = []
    group_size = 3  # Combine at most 3 shards at a time to reduce memory usage
    
    for i in range(0, len(shard_dirs), group_size):
        group = shard_dirs[i:i + group_size]
        group_datasets = []
        
        for shard_dir in group:
            print(f"Loading shard: {os.path.basename(shard_dir)}")
            try:
                shard_dataset = Dataset.load_from_disk(shard_dir)
                if len(shard_dataset) > 0:  # Only add non-empty datasets
                    group_datasets.append(shard_dataset)
            except Exception as e:
                print(f"Warning: Failed to load shard {shard_dir}: {e}")
        
        if group_datasets:
            if len(group_datasets) == 1:
                combined_group = group_datasets[0]
            else:
                combined_group = concatenate_datasets(group_datasets)
            
            if len(combined_group) > 0:  # Only add non-empty datasets
                datasets_to_combine.append(combined_group)
        
        # Clean up loaded datasets to free memory
        del group_datasets
    
    # Final combination
    if not datasets_to_combine:
        print("No non-empty shards found")
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    elif len(datasets_to_combine) == 1:
        final_dataset = datasets_to_combine[0]
    else:
        print("Performing final combination...")
        final_dataset = concatenate_datasets(datasets_to_combine)
    
    # Save final dataset
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving final dataset to {final_output_dir}")
    final_dataset.save_to_disk(final_output_dir, max_shard_size=max_shard_size)
    
    # Clean up temporary shard directories
    print("Cleaning up temporary shard directories...")
    for shard_dir in shard_dirs:
        try:
            shutil.rmtree(shard_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up {shard_dir}: {e}")
    
    return final_dataset


def process_datasets_streaming(dataset_configs, batch_size: int = 1000, 
                             cache_dir: Optional[str] = None,
                             tokenizer: PreTrainedTokenizer = None,
                             min_length: int = 100,
                             sequence_len: int = 2048,
                             output_dir: str = None,
                             shard_size: str = "50MB") -> tuple[Dataset, dict]:
    """Process datasets using true streaming approach without loading everything into memory."""
    
    deduplicator = StreamingFileBasedDeduplicator(cache_dir)
    temp_output_dir = output_dir + "_temp_shards"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    stats = {
        'total_samples_processed': 0,
        'total_samples_after_quality_filter': 0,
        'total_samples_after_dedup': 0,
        'total_samples_after_length_filter': 0,
        'total_packed_samples': 0,
        'total_tokens_dropped_quality': 0,
        'total_tokens_dropped_length': 0,
        'total_tokens_dropped_packing': 0
    }
    
    saved_batch_count = 0
    
    try:
        for ds_cfg in dataset_configs:
            print(f"\nStreaming dataset: {ds_cfg.path}")
            
            for batch, batch_num, total_so_far in stream_dataset_batches(ds_cfg, batch_size):
                print(f"Processing batch {batch_num}, total samples seen: {total_so_far}")
                stats['total_samples_processed'] += len(batch)
                
                # Process batch (clean, quality filter, dedupe)
                processed_batch = process_batch_streaming(batch, deduplicator, min_length)
                stats['total_samples_after_quality_filter'] += len(processed_batch)
                stats['total_samples_after_dedup'] += len(processed_batch)
                
                if not processed_batch:
                    continue
                
                # Tokenize batch
                tokenized_batch = tokenize_batch_streaming(processed_batch, tokenizer)
                
                # Filter by length
                filtered_batch, dropped_tokens = filter_batch_by_length_streaming(
                    tokenized_batch, min_length
                )
                stats['total_samples_after_length_filter'] += len(filtered_batch)
                stats['total_tokens_dropped_length'] += dropped_tokens
                
                if not filtered_batch:
                    continue
                
                # Pack batch
                packed_batch, dropped_packing_tokens = pack_batch_streaming(
                    filtered_batch, sequence_len
                )
                stats['total_packed_samples'] += len(packed_batch)
                stats['total_tokens_dropped_packing'] += dropped_packing_tokens
                
                if packed_batch:
                    # Save batch immediately to disk
                    save_batch_to_disk(packed_batch, temp_output_dir, saved_batch_count, shard_size)
                    saved_batch_count += 1
                
                # Report progress
                if batch_num % 10 == 0:
                    print(f"  Batch {batch_num}: {len(batch)} -> {len(processed_batch)} -> "
                          f"{len(filtered_batch)} -> {len(packed_batch)} samples")
        
        # Combine all saved shards into final dataset
        final_dataset = combine_saved_shards(temp_output_dir, output_dir, shard_size)
        
        print(f"\nStreaming processing complete:")
        print(f"  Total samples processed: {stats['total_samples_processed']:,}")
        print(f"  After quality filtering: {stats['total_samples_after_quality_filter']:,}")
        print(f"  After deduplication: {stats['total_samples_after_dedup']:,}")
        print(f"  After length filtering: {stats['total_samples_after_length_filter']:,}")
        print(f"  Final packed samples: {stats['total_packed_samples']:,}")
        print(f"  Tokens dropped (length): {stats['total_tokens_dropped_length']:,}")
        print(f"  Tokens dropped (packing): {stats['total_tokens_dropped_packing']:,}")
        
        return final_dataset, stats
    
    finally:
        # Clean up deduplication cache if using temporary directory
        if cache_dir is None:
            deduplicator.cleanup()
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)


def process_datasets_streaming_optimized(dataset_configs, batch_size: int = 1000, 
                                        cache_dir: Optional[str] = None,
                                        tokenizer: PreTrainedTokenizer = None,
                                        min_length: int = 100,
                                        sequence_len: int = 2048,
                                        output_dir: str = None,
                                        shard_size: str = "50MB") -> tuple[Dataset, dict]:
    """
    Truly memory-efficient streaming approach that never accumulates all data in memory.
    This approach directly builds the final dataset structure without intermediate collections.
    """
    
    deduplicator = StreamingFileBasedDeduplicator(cache_dir)
    
    stats = {
        'total_samples_processed': 0,
        'total_samples_after_quality_filter': 0,
        'total_samples_after_dedup': 0,
        'total_samples_after_length_filter': 0,
        'total_packed_samples': 0,
        'total_tokens_dropped_quality': 0,
        'total_tokens_dropped_length': 0,
        'total_tokens_dropped_packing': 0
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a temporary directory for incremental dataset building
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="krill_opt_")
    chunk_datasets = []
    chunk_counter = 0
    
    # Memory-efficient chunk size (process in smaller chunks)
    chunk_size = 500  # Keep only 500 samples in memory at a time (reduced from 1000)
    current_chunk = []
    
    try:
        for ds_cfg in dataset_configs:
            print(f"\nStreaming dataset: {ds_cfg.path}")
            
            for batch, batch_num, total_so_far in stream_dataset_batches(ds_cfg, batch_size):
                print(f"Processing batch {batch_num}, total samples seen: {total_so_far}")
                stats['total_samples_processed'] += len(batch)
                
                # Process batch (clean, quality filter, dedupe)
                processed_batch = process_batch_streaming(batch, deduplicator, min_length)
                stats['total_samples_after_quality_filter'] += len(processed_batch)
                stats['total_samples_after_dedup'] += len(processed_batch)
                
                if not processed_batch:
                    continue
                
                # Tokenize batch
                tokenized_batch = tokenize_batch_streaming(processed_batch, tokenizer)
                
                # Filter by length
                filtered_batch, dropped_tokens = filter_batch_by_length_streaming(
                    tokenized_batch, min_length
                )
                stats['total_samples_after_length_filter'] += len(filtered_batch)
                stats['total_tokens_dropped_length'] += dropped_tokens
                
                if not filtered_batch:
                    continue
                
                # Pack batch
                packed_batch, dropped_packing_tokens = pack_batch_streaming(
                    filtered_batch, sequence_len
                )
                stats['total_packed_samples'] += len(packed_batch)
                stats['total_tokens_dropped_packing'] += dropped_packing_tokens
                
                if packed_batch:
                    # Add to current chunk
                    current_chunk.extend(packed_batch)
                    
                    # If chunk is full, save it and clear memory
                    if len(current_chunk) >= chunk_size:
                        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_counter:05d}")
                        chunk_dataset = Dataset.from_list(current_chunk)
                        chunk_dataset.save_to_disk(chunk_path)
                        chunk_datasets.append(chunk_path)
                        
                        print(f"  Saved chunk {chunk_counter} with {len(current_chunk)} samples")
                        current_chunk.clear()  # Clear memory immediately
                        chunk_counter += 1
                
                # Report progress
                if batch_num % 10 == 0:
                    print(f"  Batch {batch_num}: {len(batch)} -> {len(processed_batch)} -> "
                          f"{len(filtered_batch)} -> {len(packed_batch)} samples")
        
        # Save any remaining samples in the current chunk
        if current_chunk:
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_counter:05d}")
            chunk_dataset = Dataset.from_list(current_chunk)
            chunk_dataset.save_to_disk(chunk_path)
            chunk_datasets.append(chunk_path)
            print(f"  Saved final chunk {chunk_counter} with {len(current_chunk)} samples")
            current_chunk.clear()
        
        print(f"\nCombining {len(chunk_datasets)} chunks efficiently...")
        
        # Combine chunks efficiently in small groups to minimize memory usage
        if len(chunk_datasets) == 0:
            final_dataset = Dataset.from_dict({"input_ids": [], "attention_mask": []})
        elif len(chunk_datasets) == 1:
            # Just move the single chunk
            import shutil
            final_dataset = Dataset.load_from_disk(chunk_datasets[0])
            final_dataset.save_to_disk(output_dir, max_shard_size=shard_size)
        else:
            # Combine in groups of 2 to minimize memory usage (reduced from 3)
            combined_chunks = []
            group_size = 2
            
            for i in range(0, len(chunk_datasets), group_size):
                group = chunk_datasets[i:i + group_size]
                group_datasets = [Dataset.load_from_disk(path) for path in group]
                
                if len(group_datasets) == 1:
                    combined_group = group_datasets[0]
                else:
                    from datasets import concatenate_datasets
                    combined_group = concatenate_datasets(group_datasets)
                
                # Save combined group
                group_path = os.path.join(temp_dir, f"group_{i // group_size}")
                combined_group.save_to_disk(group_path, max_shard_size=shard_size)
                combined_chunks.append(group_path)
                
                # Clear memory immediately
                del group_datasets, combined_group
                print(f"  Combined group {i // group_size} with {len(group)} chunks")
            
            # Final combination
            if len(combined_chunks) == 1:
                import shutil
                final_dataset = Dataset.load_from_disk(combined_chunks[0])
                final_dataset.save_to_disk(output_dir, max_shard_size=shard_size)
            else:
                final_groups = []
                for path in combined_chunks:
                    final_groups.append(Dataset.load_from_disk(path))
                
                from datasets import concatenate_datasets
                final_dataset = concatenate_datasets(final_groups)
                final_dataset.save_to_disk(output_dir, max_shard_size=shard_size)
                
                # Clear memory
                del final_groups
        
        # Load the final dataset (metadata only)
        final_dataset = Dataset.load_from_disk(output_dir)
        
        print(f"\nOptimized processing complete:")
        print(f"  Total samples processed: {stats['total_samples_processed']:,}")
        print(f"  After quality filtering: {stats['total_samples_after_quality_filter']:,}")
        print(f"  After deduplication: {stats['total_samples_after_dedup']:,}")
        print(f"  After length filtering: {stats['total_samples_after_length_filter']:,}")
        print(f"  Final packed samples: {len(final_dataset):,}")
        print(f"  Tokens dropped (length): {stats['total_tokens_dropped_length']:,}")
        print(f"  Tokens dropped (packing): {stats['total_tokens_dropped_packing']:,}")
        
        return final_dataset, stats
    
    finally:
        # Clean up deduplication cache if using temporary directory
        if cache_dir is None:
            deduplicator.cleanup()
        
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)