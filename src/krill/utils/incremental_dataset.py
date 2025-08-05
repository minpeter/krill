"""
Truly memory-efficient dataset processing that never loads the entire dataset into memory.
This implementation builds the final dataset incrementally without intermediate concatenations.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import Dataset
import tempfile


class IncrementalDatasetBuilder:
    """
    Builds a HuggingFace dataset incrementally without loading all data into memory.
    Uses small temporary datasets that are combined efficiently.
    """
    
    def __init__(self, output_dir: str, max_shard_size: str = "200MB"):
        self.output_dir = Path(output_dir)
        self.max_shard_size = max_shard_size
        self.temp_dir = Path(tempfile.mkdtemp(prefix="krill_incremental_"))
        self.shard_datasets = []
        self.current_shard_data = []
        self.total_samples = 0
        
        # Convert shard size to number of samples (rough estimate)
        # Assume ~2KB per sample (2048 tokens * 4 bytes / token / 4 for compression)
        size_bytes = self._parse_size(max_shard_size)
        self.max_shard_samples = max(100, size_bytes // 2048)  # At least 100 samples per shard
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '200MB' to bytes."""
        size_str = size_str.upper().strip()
        if size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            return int(size_str)  # Assume bytes
    
    def add_batch(self, batch: List[Dict[str, Any]]):
        """Add a batch of samples to the dataset."""
        if not batch:
            return
        
        # Add samples to current shard
        self.current_shard_data.extend(batch)
        self.total_samples += len(batch)
        
        # Check if we need to flush current shard
        if len(self.current_shard_data) >= self.max_shard_samples:
            self._flush_current_shard()
    
    def _flush_current_shard(self):
        """Save current shard data as a small dataset and clear memory."""
        if not self.current_shard_data:
            return
        
        shard_idx = len(self.shard_datasets)
        shard_path = self.temp_dir / f"shard_{shard_idx:05d}"
        
        # Create small dataset and save it
        shard_dataset = Dataset.from_list(self.current_shard_data)
        shard_dataset.save_to_disk(str(shard_path))
        
        self.shard_datasets.append(shard_path)
        print(f"  Saved shard {shard_idx} with {len(self.current_shard_data)} samples")
        
        # Clear current shard data from memory
        self.current_shard_data.clear()
    
    def finalize(self) -> Dataset:
        """Finalize the dataset by efficiently combining shards."""
        # Flush any remaining data
        self._flush_current_shard()
        
        if not self.shard_datasets:
            print("No data to finalize")
            return Dataset.from_dict({"input_ids": [], "attention_mask": []})
        
        print(f"Combining {len(self.shard_datasets)} shards efficiently...")
        
        # If only one shard, just move it
        if len(self.shard_datasets) == 1:
            shutil.move(str(self.shard_datasets[0]), str(self.output_dir))
            final_dataset = Dataset.load_from_disk(str(self.output_dir))
        else:
            # Combine shards in groups to avoid loading everything at once
            combined_datasets = []
            group_size = 4  # Combine at most 4 shards at a time
            
            for i in range(0, len(self.shard_datasets), group_size):
                group = self.shard_datasets[i:i + group_size]
                group_datasets = [Dataset.load_from_disk(str(path)) for path in group]
                
                if len(group_datasets) == 1:
                    combined_group = group_datasets[0]
                else:
                    from datasets import concatenate_datasets
                    combined_group = concatenate_datasets(group_datasets)
                
                # Save the combined group
                group_path = self.temp_dir / f"combined_group_{i // group_size}"
                combined_group.save_to_disk(str(group_path))
                combined_datasets.append(group_path)
                
                # Clear memory
                del group_datasets, combined_group
            
            # Final combination
            if len(combined_datasets) == 1:
                shutil.move(str(combined_datasets[0]), str(self.output_dir))
            else:
                final_groups = [Dataset.load_from_disk(str(path)) for path in combined_datasets]
                from datasets import concatenate_datasets
                final_dataset = concatenate_datasets(final_groups)
                final_dataset.save_to_disk(str(self.output_dir), max_shard_size=self.max_shard_size)
        
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Load the final dataset
        final_dataset = Dataset.load_from_disk(str(self.output_dir))
        print(f"Incremental dataset finalized with {len(final_dataset)} samples")
        
        return final_dataset


def process_datasets_incremental(dataset_configs, batch_size: int = 1000, 
                                cache_dir: Optional[str] = None,
                                tokenizer = None,
                                min_length: int = 100,
                                sequence_len: int = 2048,
                                output_dir: str = None,
                                shard_size: str = "200MB") -> tuple[Dataset, dict]:
    """
    Process datasets using truly incremental approach that minimizes memory usage.
    """
    from .streaming_dataset import (
        StreamingFileBasedDeduplicator, stream_dataset_batches, 
        process_batch_streaming, tokenize_batch_streaming, 
        filter_batch_by_length_streaming, pack_batch_streaming
    )
    
    deduplicator = StreamingFileBasedDeduplicator(cache_dir)
    builder = IncrementalDatasetBuilder(output_dir, shard_size)
    
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
                    # Add directly to incremental builder
                    builder.add_batch(packed_batch)
                
                # Report progress
                if batch_num % 10 == 0:
                    print(f"  Batch {batch_num}: {len(batch)} -> {len(processed_batch)} -> "
                          f"{len(filtered_batch)} -> {len(packed_batch)} samples")
        
        # Finalize the dataset
        final_dataset = builder.finalize()
        
        print(f"\nIncremental processing complete:")
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