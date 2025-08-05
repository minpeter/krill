#!/usr/bin/env python3
"""
Simple memory analysis script to understand the memory usage differences
between standard and memory-efficient modes.
"""

import os
import sys
import psutil
import tempfile
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import pack_dataset

def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class MemoryTracker:
    def __init__(self, name):
        self.name = name
        self.start_memory = get_memory_mb()
        self.peak_memory = self.start_memory
        print(f"🧠 {name}: Starting memory: {self.start_memory:.1f} MB")
    
    def report(self, stage):
        current = get_memory_mb()
        if current > self.peak_memory:
            self.peak_memory = current
        increase = current - self.start_memory
        print(f"🧠 {self.name} - {stage}: {current:.1f} MB (+{increase:.1f} MB, peak: {self.peak_memory:.1f} MB)")
        return current
    
    def final_report(self):
        current = get_memory_mb()
        total_increase = self.peak_memory - self.start_memory
        print(f"🧠 {self.name} FINAL: Peak increase: {total_increase:.1f} MB")
        return total_increase

def test_standard_preprocessing():
    """Test standard preprocessing approach."""
    print("\n" + "="*50)
    print("TESTING STANDARD PREPROCESSING")
    print("="*50)
    
    tracker = MemoryTracker("STANDARD")
    
    # Load dataset
    tracker.report("before dataset load")
    dataset = load_dataset("minpeter/fineweb-2-edu-korean", split="train[:1000]")
    tracker.report("after dataset load")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("minpeter/fw2-edu-kr-tokenizer-48k")
    tracker.report("after tokenizer load")
    
    # Tokenize function
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"], padding=False, truncation=False)
        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
        return tokenized_inputs
    
    # Tokenize
    tracker.report("before tokenization")
    tokenized = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=dataset.column_names)
    tracker.report("after tokenization")
    
    # Filter by length
    min_length = 100
    lengths = [len(x) for x in tokenized["input_ids"]]
    selected = [i for i, l in enumerate(lengths) if l >= min_length]
    tokenized = tokenized.select(selected)
    tracker.report("after filtering")
    
    # Pack
    tracker.report("before packing")
    packed = pack_dataset(tokenized, seq_length=2048, strategy="wrapped")
    tracker.report("after packing")
    
    # Save to temporary location
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker.report("before saving")
        packed.save_to_disk(temp_dir)
        tracker.report("after saving")
    
    return tracker.final_report()

def test_memory_efficient_preprocessing():
    """Test current memory-efficient preprocessing approach."""
    print("\n" + "="*50)
    print("TESTING MEMORY-EFFICIENT PREPROCESSING")
    print("="*50)
    
    tracker = MemoryTracker("MEMORY_EFFICIENT")
    
    # Load dataset
    tracker.report("before dataset load")
    dataset = load_dataset("minpeter/fineweb-2-edu-korean", split="train[:1000]")
    tracker.report("after dataset load")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("minpeter/fw2-edu-kr-tokenizer-48k")
    tracker.report("after tokenizer load")
    
    # Tokenize function
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"], padding=False, truncation=False)
        if tokenizer.eos_token_id is not None:
            for i in range(len(tokenized_inputs["input_ids"])):
                tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                tokenized_inputs["attention_mask"][i].append(1)
        return tokenized_inputs
    
    # Tokenize with "memory-efficient" settings
    tracker.report("before tokenization")
    tokenized = dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=1000,  # smaller batch size
        num_proc=1,  # fewer processes
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # disable caching
        keep_in_memory=False  # don't keep in memory
    )
    tracker.report("after tokenization")
    
    # Filter by length
    min_length = 100
    lengths = [len(x) for x in tokenized["input_ids"]]
    selected = [i for i, l in enumerate(lengths) if l >= min_length]
    tokenized = tokenized.select(selected)
    tracker.report("after filtering")
    
    # Pack with "memory-efficient" settings
    tracker.report("before packing")
    packed = pack_dataset(
        tokenized, 
        seq_length=2048, 
        strategy="wrapped",
        map_kwargs={
            "batch_size": 500,  # smaller batch size
            "load_from_cache_file": False,
            "keep_in_memory": False
        }
    )
    tracker.report("after packing")
    
    # Save with smaller shard size
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker.report("before saving")
        packed.save_to_disk(temp_dir, max_shard_size="50MB", num_proc=1)
        tracker.report("after saving")
    
    return tracker.final_report()

def test_truly_memory_efficient_preprocessing():
    """Test a truly memory-efficient approach using chunked processing."""
    print("\n" + "="*50)
    print("TESTING TRULY MEMORY-EFFICIENT PREPROCESSING")
    print("="*50)
    
    tracker = MemoryTracker("TRULY_EFFICIENT")
    
    # Instead of loading entire dataset, we'll process in chunks
    print("🧠 Using chunked processing approach...")
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("minpeter/fw2-edu-kr-tokenizer-48k")
    tracker.report("after tokenizer load")
    
    # Process dataset in chunks using streaming
    chunk_size = 200  # Process only 200 samples at a time
    total_samples = 1000
    
    processed_chunks = []
    
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_split = f"train[{start_idx}:{end_idx}]"
        
        print(f"🧠 Processing chunk {start_idx}-{end_idx}...")
        
        # Load small chunk
        tracker.report(f"before chunk {start_idx} load")
        chunk_dataset = load_dataset("minpeter/fineweb-2-edu-korean", split=chunk_split)
        tracker.report(f"after chunk {start_idx} load")
        
        # Tokenize chunk
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples["text"], padding=False, truncation=False)
            if tokenizer.eos_token_id is not None:
                for i in range(len(tokenized_inputs["input_ids"])):
                    tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                    tokenized_inputs["attention_mask"][i].append(1)
            return tokenized_inputs
        
        tokenized_chunk = chunk_dataset.map(
            tokenize_function, 
            batched=True, 
            batch_size=50,  # very small batch
            num_proc=1,
            remove_columns=chunk_dataset.column_names,
            load_from_cache_file=False,
            keep_in_memory=False
        )
        tracker.report(f"after chunk {start_idx} tokenization")
        
        # Filter chunk
        min_length = 100
        lengths = [len(x) for x in tokenized_chunk["input_ids"]]
        selected = [i for i, l in enumerate(lengths) if l >= min_length]
        if selected:
            tokenized_chunk = tokenized_chunk.select(selected)
            processed_chunks.append(tokenized_chunk)
        
        tracker.report(f"after chunk {start_idx} processing")
        
        # Clear chunk from memory
        del chunk_dataset, tokenized_chunk
        
    # Combine chunks incrementally
    tracker.report("before combining chunks")
    if processed_chunks:
        combined = processed_chunks[0]
        for i, chunk in enumerate(processed_chunks[1:], 1):
            # Use concatenate_datasets for small chunks only
            from datasets import concatenate_datasets
            combined = concatenate_datasets([combined, chunk])
            tracker.report(f"after combining chunk {i}")
            del chunk
    else:
        combined = processed_chunks[0] if processed_chunks else None
    
    if combined:
        # Pack the combined dataset
        tracker.report("before packing")
        packed = pack_dataset(
            combined, 
            seq_length=2048, 
            strategy="wrapped",
            map_kwargs={
                "batch_size": 100,
                "load_from_cache_file": False,
                "keep_in_memory": False
            }
        )
        tracker.report("after packing")
        
        # Save with smaller shard size
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker.report("before saving")
            packed.save_to_disk(temp_dir, max_shard_size="10MB", num_proc=1)
            tracker.report("after saving")
    
    return tracker.final_report()

if __name__ == "__main__":
    print("🧠 Memory Analysis for Krill Preprocessing")
    print("Testing different approaches with small dataset...")
    
    try:
        standard_increase = test_standard_preprocessing()
        efficient_increase = test_memory_efficient_preprocessing()
        truly_efficient_increase = test_truly_memory_efficient_preprocessing()
        
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        print(f"Standard mode peak increase:        {standard_increase:.1f} MB")
        print(f"Memory-efficient mode peak increase: {efficient_increase:.1f} MB")
        print(f"Truly efficient mode peak increase:  {truly_efficient_increase:.1f} MB")
        print("="*60)
        
        if efficient_increase > standard_increase:
            print("❌ PROBLEM: Memory-efficient mode uses MORE memory than standard!")
            print(f"   Difference: +{efficient_increase - standard_increase:.1f} MB")
        else:
            print("✅ Memory-efficient mode uses less memory than standard")
            
        if truly_efficient_increase < min(standard_increase, efficient_increase):
            print("✅ Truly efficient mode uses the least memory")
        else:
            print("❌ Truly efficient mode doesn't provide the best memory usage")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()