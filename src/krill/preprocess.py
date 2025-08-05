import os
import tempfile

from transformers import AutoTokenizer
from trl import pack_dataset

from krill.utils.config import KrillConfig
from krill.utils.inspect_dataset import inspect_pretrain_dataset
from krill.utils.memory_monitor import MemoryMonitor


def do_preprocess(config: KrillConfig):
    """Preprocesses the data based on the loaded Config object."""
    print("🦐 Krill: Starting preprocessing...")
    
    # Initialize memory monitoring
    monitor = MemoryMonitor()
    monitor.start_monitoring()

    # Prepare output directory
    os.makedirs(config.dataset_prepared_path, exist_ok=True)

    if config.preprocess_memory_efficient:
        print("🧠 Using memory-efficient preprocessing mode")
        _do_preprocess_memory_efficient(config, monitor)
    else:
        print("📚 Using standard preprocessing mode")
        _do_preprocess_standard(config, monitor)
    
    monitor.report_final()


def _do_preprocess_standard(config: KrillConfig, monitor: MemoryMonitor):
    """Standard preprocessing (original implementation)."""
    # Load and prepare raw datasets
    from krill.utils.dataset_utils import load_and_prepare_raw_datasets
    monitor.report_current("before loading datasets")
    raw_dataset = load_and_prepare_raw_datasets(config.datasets)
    monitor.report_current("after loading datasets")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)

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

    num_proc = max(1, os.cpu_count() - 8)
    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_proc} processes for mapping."
    )

    monitor.report_current("before tokenization")
    tokenized = raw_dataset.map(tokenize_function,
                                batched=True,
                                num_proc=num_proc,
                                remove_columns=raw_dataset.column_names,
                                desc="Tokenizing")
    monitor.report_current("after tokenization")

    # Filter by min_length
    lengths = tokenized["input_ids"].map(len) if hasattr(
        tokenized["input_ids"],
        'map') else [len(x) for x in tokenized["input_ids"]]
    selected = [
        i for i, l in enumerate(lengths)
        if l >= config.dataset_prepared_min_length
    ]
    # Compute token drop statistics from filtering
    total_tokens_before_filter = sum(lengths)
    total_tokens_after_filter = sum(lengths[i] for i in selected)
    filter_dropped_tokens = total_tokens_before_filter - total_tokens_after_filter
    tokenized = tokenized.select(selected)
    monitor.report_current("after length filtering")

    _pack_and_save_dataset(config, tokenized, filter_dropped_tokens, monitor)


def _do_preprocess_memory_efficient(config: KrillConfig, monitor: MemoryMonitor):
    """Truly memory-efficient preprocessing using chunked processing."""
    print(f"🧠 Memory-efficient mode: Using chunked processing for bounded memory usage")
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    monitor.report_current("after tokenizer load")

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

    # Process each dataset configuration in chunks
    total_filter_dropped_tokens = 0
    processed_chunks = []
    temp_chunk_dir = os.path.join(config.dataset_prepared_path, "temp_chunks")
    os.makedirs(temp_chunk_dir, exist_ok=True)
    
    try:
        for ds_idx, ds_cfg in enumerate(config.datasets):
            print(f"🧠 Processing dataset {ds_idx + 1}/{len(config.datasets)}: {ds_cfg.path}")
            
            # Parse the split to get total size and chunk it
            split_info = ds_cfg.split
            if split_info.startswith("train[") and ":" in split_info:
                # Extract range like train[0:50000] or train[:1000]
                range_part = split_info[6:-1]  # Remove "train[" and "]"
                if range_part.startswith(":"):
                    start_idx = 0
                    end_idx = int(range_part[1:].replace("_", ""))
                elif range_part.endswith(":"):
                    start_idx = int(range_part[:-1].replace("_", ""))
                    end_idx = None  # Will need to determine actual size
                else:
                    start_str, end_str = range_part.split(":")
                    start_idx = int(start_str.replace("_", "")) if start_str else 0
                    end_idx = int(end_str.replace("_", "")) if end_str else None
                    
                if end_idx is None:
                    print(f"⚠️  Cannot determine dataset size for split '{split_info}', using standard mode for this dataset")
                    from krill.utils.dataset_utils import load_dataset_single
                    raw_dataset = load_dataset_single(ds_cfg)
                    monitor.report_current(f"after loading dataset {ds_idx}")
                    
                    # Process as single chunk (fallback to standard mode for this dataset)
                    tokenized = raw_dataset.map(
                        tokenize_function,
                        batched=True,
                        batch_size=config.preprocess_chunk_size,
                        num_proc=1,
                        remove_columns=raw_dataset.column_names,
                        desc=f"Tokenizing dataset {ds_idx} (fallback)",
                        load_from_cache_file=False,
                        keep_in_memory=False
                    )
                    
                    # Filter and add to processed chunks
                    lengths = [len(x) for x in tokenized["input_ids"]]
                    selected = [i for i, l in enumerate(lengths) if l >= config.dataset_prepared_min_length]
                    total_filter_dropped_tokens += sum(lengths) - sum(lengths[i] for i in selected)
                    
                    if selected:
                        tokenized = tokenized.select(selected)
                        processed_chunks.append(tokenized)
                    
                    monitor.report_current(f"after processing dataset {ds_idx}")
                    continue
                
                # Process in chunks
                chunk_size = min(config.preprocess_chunk_size, 500)  # Use smaller chunks for memory efficiency
                print(f"🧠 Processing {end_idx - start_idx} samples in chunks of {chunk_size}")
                
                for chunk_start in range(start_idx, end_idx, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, end_idx)
                    chunk_split = f"train[{chunk_start}:{chunk_end}]"
                    
                    print(f"🧠 Processing chunk {chunk_start}-{chunk_end}...")
                    monitor.report_current(f"before chunk {chunk_start}")
                    
                    # Load small chunk
                    from krill.utils.dataset_utils import load_dataset_single
                    from krill.utils.config import DatasetConfig
                    chunk_config = DatasetConfig(
                        path=ds_cfg.path,
                        split=chunk_split,
                        text_column=ds_cfg.text_column,
                        name=ds_cfg.name
                    )
                    chunk_dataset = load_dataset_single(chunk_config)
                    monitor.report_current(f"after loading chunk {chunk_start}")
                    
                    # Tokenize chunk
                    tokenized_chunk = chunk_dataset.map(
                        tokenize_function,
                        batched=True,
                        batch_size=50,  # Very small batch for memory efficiency
                        num_proc=1,
                        remove_columns=chunk_dataset.column_names,
                        desc=f"Tokenizing chunk {chunk_start}-{chunk_end}",
                        load_from_cache_file=False,
                        keep_in_memory=False
                    )
                    monitor.report_current(f"after tokenizing chunk {chunk_start}")
                    
                    # Filter chunk by length
                    lengths = [len(x) for x in tokenized_chunk["input_ids"]]
                    selected = [i for i, l in enumerate(lengths) if l >= config.dataset_prepared_min_length]
                    total_filter_dropped_tokens += sum(lengths) - sum(lengths[i] for i in selected)
                    
                    if selected:
                        filtered_chunk = tokenized_chunk.select(selected)
                        processed_chunks.append(filtered_chunk)
                    
                    monitor.report_current(f"after processing chunk {chunk_start}")
                    
                    # Clear chunk from memory immediately
                    del chunk_dataset, tokenized_chunk
                    if 'filtered_chunk' in locals():
                        del filtered_chunk
            else:
                # Handle other split formats (fallback to standard loading)
                print(f"⚠️  Split format '{split_info}' not supported for chunked processing, using standard mode")
                from krill.utils.dataset_utils import load_dataset_single
                raw_dataset = load_dataset_single(ds_cfg)
                monitor.report_current(f"after loading dataset {ds_idx}")
                
                tokenized = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=config.preprocess_chunk_size,
                    num_proc=1,
                    remove_columns=raw_dataset.column_names,
                    desc=f"Tokenizing dataset {ds_idx}",
                    load_from_cache_file=False,
                    keep_in_memory=False
                )
                
                lengths = [len(x) for x in tokenized["input_ids"]]
                selected = [i for i, l in enumerate(lengths) if l >= config.dataset_prepared_min_length]
                total_filter_dropped_tokens += sum(lengths) - sum(lengths[i] for i in selected)
                
                if selected:
                    tokenized = tokenized.select(selected)
                    processed_chunks.append(tokenized)
                
                monitor.report_current(f"after processing dataset {ds_idx}")

        # Combine processed chunks using memory-efficient approach
        print(f"🧠 Combining {len(processed_chunks)} processed chunks...")
        monitor.report_current("before combining chunks")
        
        if not processed_chunks:
            print("❌ No data remaining after filtering")
            return
            
        combined_dataset = _combine_chunks_memory_efficient(processed_chunks, monitor)
        monitor.report_current("after combining chunks")
        
        # Pack and save the combined dataset
        _pack_and_save_dataset_memory_efficient(config, combined_dataset, total_filter_dropped_tokens, monitor)
        
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)


def _combine_chunks_memory_efficient(chunks, monitor):
    """Combine dataset chunks using memory-efficient approach."""
    if len(chunks) == 1:
        return chunks[0]
    
    print(f"🧠 Combining {len(chunks)} chunks using incremental approach...")
    
    # Combine chunks incrementally to avoid memory spikes
    combined = chunks[0]
    
    for i, chunk in enumerate(chunks[1:], 1):
        print(f"🧠 Combining chunk {i}/{len(chunks)-1}...")
        
        # Use concatenate_datasets for small chunks
        from datasets import concatenate_datasets
        combined = concatenate_datasets([combined, chunk])
        monitor.report_current(f"after combining chunk {i}")
        
        # Clear the chunk reference
        del chunk
        
        # Force garbage collection if many chunks
        if i % 5 == 0:
            import gc
            gc.collect()
    
    return combined


def _pack_and_save_dataset_memory_efficient(config: KrillConfig, tokenized, filter_dropped_tokens, monitor: MemoryMonitor):
    """Memory-efficient pack and save with truly bounded memory usage."""
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    
    print(f"🧠 Packing {len(tokenized)} samples into sequences of length {config.sequence_len} (memory-efficient mode)...")
    
    # Use very small batch size for memory efficiency
    batch_size = min(100, len(tokenized) // 10 + 1)  # Adaptive batch size, but very small
    print(f"🧠 Using adaptive batch_size={batch_size} for packing")
    
    monitor.report_current("before packing")
    
    # Pack sequences with memory-optimized parameters
    packed = pack_dataset(tokenized,
                          seq_length=config.sequence_len,
                          strategy="wrapped",
                          map_kwargs={
                              "batch_size": batch_size,
                              "load_from_cache_file": False,  # Disable caching
                              "keep_in_memory": False,  # Don't keep in memory
                              "num_proc": 1  # Single process to minimize memory usage
                          })
    monitor.report_current("after packing")
    
    # Filter out incomplete samples efficiently
    last_dropped_chunk_length = 0
    if len(packed) > 0:
        print(f"🧠 Filtering packed samples for correct length...")
        # Check samples in small batches to avoid loading everything into memory
        correct_indices = []
        incorrect_tokens = 0
        
        # Process in small batches to maintain memory efficiency
        batch_size_filter = 100
        for i in range(0, len(packed), batch_size_filter):
            end_idx = min(i + batch_size_filter, len(packed))
            batch = packed.select(range(i, end_idx))
            
            for j, sample in enumerate(batch):
                global_idx = i + j
                if len(sample["input_ids"]) == config.sequence_len:
                    correct_indices.append(global_idx)
                else:
                    incorrect_tokens += len(sample["input_ids"])
            
            # Clear batch from memory
            del batch
        
        if len(correct_indices) < len(packed):
            dropped_samples = len(packed) - len(correct_indices)
            last_dropped_chunk_length = incorrect_tokens
            packed = packed.select(correct_indices)
            print(f"🧠 Dropped {dropped_samples} incomplete samples ({incorrect_tokens} tokens)")

    # Save with very small shard size to minimize memory spikes
    monitor.report_current("before saving")
    
    print(f"🧠 Saving dataset with minimal memory settings...")
    # Calculate adaptive shard size based on dataset size
    if len(packed) > 0:
        estimated_size_mb = len(packed) * config.sequence_len * 4 / (1024 * 1024)  # Rough estimate
        adaptive_shard_size = min(config.preprocess_save_shard_size, f"{max(10, estimated_size_mb // 10):.0f}MB")
        print(f"🧠 Using adaptive shard size: {adaptive_shard_size}")
        
        packed.save_to_disk(
            config.dataset_prepared_path, 
            max_shard_size=adaptive_shard_size,
            num_proc=1  # Single process to minimize memory usage
        )
    else:
        print("❌ No data to save after filtering")
        return
    
    monitor.report_current("after saving")

    inspect_pretrain_dataset(dataset=packed,
                             tokenizer=tokenizer,
                             show_example_rows_limit=1)

    print(
        f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from incomplete sequences"
    )

    print(f"\nOriginal dataset rows: {len(tokenized)}")
    print(f"Packed dataset rows: {len(packed)}")
    
    # Calculate total tokens in small batches to avoid memory spike
    total_tokens = 0
    batch_size_count = 1000
    for i in range(0, len(packed), batch_size_count):
        end_idx = min(i + batch_size_count, len(packed))
        batch = packed.select(range(i, end_idx))
        total_tokens += sum(len(sample["input_ids"]) for sample in batch)
        del batch
    
    print(
        f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m")

    # Verify sample lengths in small batches
    wrong_indices = []
    for i in range(0, min(len(packed), 1000), 100):  # Check only first 1000 samples in batches
        end_idx = min(i + 100, len(packed))
        batch = packed.select(range(i, end_idx))
        for j, sample in enumerate(batch):
            if len(sample["input_ids"]) != config.sequence_len:
                wrong_indices.append(i + j)
        del batch
        
    if wrong_indices:
        print(f"\033[1;41;97mWarning: Found {len(wrong_indices)} samples "
              f"with incorrect length (expected {config.sequence_len}). "
              f"First few indices: {wrong_indices[:10]}\033[0m")
    else:
        print(
            "\033[1;32mAll checked samples have the correct context length.\033[0m"
        )

    print(
        f"🦐 Krill: Finished memory-efficient preprocessing. Data saved to {config.dataset_prepared_path}"
    )


def _pack_and_save_dataset(config: KrillConfig, tokenized, filter_dropped_tokens, monitor: MemoryMonitor):
    """Pack and save the processed dataset."""
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    
    # Pack sequences
    print(f"Packing into sequences of length {config.sequence_len}...", end="")
    
    # Use smaller batch size for memory efficiency when in memory-efficient mode
    batch_size = len(tokenized) if not config.preprocess_memory_efficient else min(config.preprocess_chunk_size, len(tokenized))
    
    monitor.report_current("before packing")
    packed = pack_dataset(tokenized,
                          seq_length=config.sequence_len,
                          strategy="wrapped",
                          map_kwargs={"batch_size": batch_size})
    monitor.report_current("after packing")
    
    # Drop incomplete samples and record dropped tokens
    last_dropped_chunk_length = 0
    if len(packed) > 0:
        # In memory-efficient mode, smaller batch sizes can cause more incomplete samples
        # So we need to filter all samples with incorrect lengths, not just the last one
        if config.preprocess_memory_efficient:
            correct_indices = [
                i for i, sample in enumerate(packed)
                if len(sample["input_ids"]) == config.sequence_len
            ]
            if len(correct_indices) < len(packed):
                dropped_samples = len(packed) - len(correct_indices)
                # Calculate total dropped tokens from incomplete samples
                incorrect_indices = [
                    i for i in range(len(packed)) 
                    if i not in correct_indices
                ]
                last_dropped_chunk_length = sum(
                    len(packed[i]["input_ids"]) for i in incorrect_indices
                )
                packed = packed.select(correct_indices)
                print(f"Dropped {dropped_samples} incomplete samples due to memory-efficient batch processing")
        else:
            # Standard mode: only drop the last sample if incomplete
            last_len = len(packed[-1]["input_ids"])
            if last_len < config.sequence_len:
                last_dropped_chunk_length = last_len
                packed = packed.select(list(range(len(packed) - 1)))

    # Save with controlled shard size to reduce memory usage
    monitor.report_current("before saving")
    
    # Use smaller shard size and multiprocessing in memory-efficient mode to reduce memory spike during save
    if config.preprocess_memory_efficient:
        print(f"Saving dataset with shard size: {config.preprocess_save_shard_size} (memory-efficient mode)")
        # Use multiprocessing to distribute save workload
        num_proc = min(4, max(1, os.cpu_count() // 2))
        packed.save_to_disk(
            config.dataset_prepared_path, 
            max_shard_size=config.preprocess_save_shard_size,
            num_proc=num_proc
        )
    else:
        packed.save_to_disk(config.dataset_prepared_path)
    
    monitor.report_current("after saving")

    inspect_pretrain_dataset(dataset=packed,
                             tokenizer=tokenizer,
                             show_example_rows_limit=1)

    print(
        f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from final incomplete chunk (length {last_dropped_chunk_length} < sequence_len={config.sequence_len})"
    )

    print(f"\nOriginal dataset rows: {len(tokenized)}")  # Note: this is after initial processing
    print(f"Packed dataset rows: {len(packed)}")
    total_tokens = sum(len(sample["input_ids"]) for sample in packed)
    print(
        f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m")

    if len(packed) > 0:
        # Check for any samples that do not match the expected context length
        wrong_indices = [
            i for i, sample in enumerate(packed)
            if len(sample["input_ids"]) != config.sequence_len
        ]
        if wrong_indices:
            print(f"\033[1;41;97mWarning: Found {len(wrong_indices)} samples "
                  f"with incorrect length (expected {config.sequence_len}). "
                  f"Indices: {wrong_indices}\033[0m")
        else:
            print(
                "\033[1;32mAll packed samples have the correct context length.\033[0m"
            )

    print(
        f"🦐 Krill: Finished. Packed data saved to {config.dataset_prepared_path}"
    )
