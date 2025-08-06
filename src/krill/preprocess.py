import os

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

    if getattr(config, 'preprocess_memory_efficient', False):
        print("🧠 Using memory-efficient preprocessing mode")
        _do_preprocess_memory_efficient(config, monitor)
    else:
        print("📚 Using standard preprocessing mode")
        _do_preprocess_standard(config, monitor)
    
    monitor.report_final()


def _do_preprocess_standard(config: KrillConfig, monitor: MemoryMonitor):
    """Standard preprocessing (exact original implementation)."""
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
    monitor.report_current("after filtering")

    # Pack sequences
    print(f"Packing into sequences of length {config.sequence_len}...", end="")
    monitor.report_current("before packing")
    packed = pack_dataset(tokenized,
                          seq_length=config.sequence_len,
                          strategy="wrapped",
                          map_kwargs={"batch_size": len(tokenized)})
    monitor.report_current("after packing")
    # Drop incomplete last chunk and record dropped tokens
    last_dropped_chunk_length = 0
    if len(packed) > 0:
        last_len = len(packed[-1]["input_ids"])
        if last_len < config.sequence_len:
            last_dropped_chunk_length = last_len
            packed = packed.select(list(range(len(packed) - 1)))

    # Save
    monitor.report_current("before saving")
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

    print(f"\nOriginal dataset rows: {len(raw_dataset)}")
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


def _do_preprocess_memory_efficient(config: KrillConfig, monitor: MemoryMonitor):
    """Truly memory-efficient streaming preprocessing with bounded memory usage."""
    print("🧠 Memory-efficient streaming mode")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    
    # Initialize for streaming processing
    import tempfile
    import os
    from datasets import Dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    chunk_size = getattr(config, 'preprocess_chunk_size', 500)
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    total_filter_dropped_tokens = 0
    total_original_rows = 0
    carryover_tokens = []
    
    monitor.report_current("before starting streaming processing")
    
    try:
        # Process each dataset in true streaming mode
        for ds_cfg in config.datasets:
            print(f"🧠 Streaming dataset {ds_cfg.path}:{ds_cfg.split} with chunk size {chunk_size}")
            
            # Parse split notation for slice bounds
            base_split = ds_cfg.split
            start_idx, end_idx = 0, None
            
            if '[' in ds_cfg.split and ds_cfg.split.endswith(']'):
                base_split, slice_part = ds_cfg.split.split('[', 1)
                slice_part = slice_part[:-1]  # Remove ']'
                if ':' in slice_part:
                    start_str, end_str = slice_part.split(':', 1)
                    start_idx = int(start_str.replace('_', '')) if start_str else 0
                    if end_str:
                        end_idx = int(end_str.replace('_', ''))
            
            # Stream process dataset without loading full dataset into memory
            from datasets import load_dataset
            dataset_streaming = load_dataset(
                ds_cfg.path, 
                split=base_split,
                name=getattr(ds_cfg, 'name', None),
                streaming=True  # Key: Use streaming mode!
            )
            
            # Apply slice bounds to streaming dataset
            if start_idx > 0 or end_idx is not None:
                dataset_streaming = dataset_streaming.skip(start_idx)
                if end_idx is not None:
                    dataset_streaming = dataset_streaming.take(end_idx - start_idx)
            
            # Process streaming dataset in chunks
            current_chunk = []
            chunk_idx = 0
            
            for sample in dataset_streaming:
                current_chunk.append(sample[ds_cfg.text_column])
                
                # Process chunk when it's full
                if len(current_chunk) >= chunk_size:
                    sequences_created, chunk_dropped = _process_chunk_streaming(
                        current_chunk, tokenizer, config, carryover_tokens,
                        chunk_idx, temp_dir, monitor
                    )
                    total_filter_dropped_tokens += chunk_dropped
                    if sequences_created > 0:
                        chunk_files.append(f"{temp_dir}/chunk_{chunk_idx}.parquet")
                    
                    total_original_rows += len(current_chunk)
                    current_chunk = []
                    chunk_idx += 1
                    
                    # Report memory after each chunk
                    monitor.report_current(f"after processing chunk {chunk_idx}")
            
            # Process final chunk if not empty
            if current_chunk:
                sequences_created, chunk_dropped = _process_chunk_streaming(
                    current_chunk, tokenizer, config, carryover_tokens,
                    chunk_idx, temp_dir, monitor
                )
                total_filter_dropped_tokens += chunk_dropped
                if sequences_created > 0:
                    chunk_files.append(f"{temp_dir}/chunk_{chunk_idx}.parquet")
                total_original_rows += len(current_chunk)
        
        # Calculate final carryover
        final_carryover_tokens = len(carryover_tokens)
        
        # Combine chunks using streaming approach (no full dataset in memory)
        monitor.report_current("before combining chunks")
        if chunk_files:
            _combine_chunks_streaming(chunk_files, config.dataset_prepared_path, monitor)
            
            # Load just for inspection and validation (this is the only time we load the result)
            packed = Dataset.load_from_disk(config.dataset_prepared_path)
        else:
            # Create empty dataset
            packed = Dataset.from_list([])
            packed.save_to_disk(config.dataset_prepared_path)
        
        monitor.report_current("after combining chunks")
        
        # Display results
        inspect_pretrain_dataset(dataset=packed,
                                 tokenizer=tokenizer,
                                 show_example_rows_limit=1)

        print(
            f"\n\033[1;41;97mDropped {total_filter_dropped_tokens} tokens\033[0m during filtering (min_length={config.dataset_prepared_min_length})"
        )
        print(
            f"\n\033[1;41;97mDropped {final_carryover_tokens} carryover tokens\033[0m from final incomplete sequence"
        )

        print(f"\nOriginal dataset rows: {total_original_rows}")
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
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_chunk_streaming(texts, tokenizer, config, carryover_tokens, chunk_idx, temp_dir, monitor):
    """Process a single chunk in streaming mode and save to disk immediately."""
    from datasets import Dataset
    
    # Create chunk dataset
    temp_chunk_dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize chunk
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
    
    tokenized_chunk = temp_chunk_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        num_proc=1,
        remove_columns=temp_chunk_dataset.column_names,
        desc=f"Tokenizing chunk {chunk_idx}",
        load_from_cache_file=False,
        keep_in_memory=False
    )
    
    # Collect all tokens from this chunk (including carryover from previous chunk)
    all_tokens = list(carryover_tokens)
    chunk_dropped_tokens = 0
    
    # Filter and collect tokens from current chunk
    for sample in tokenized_chunk:
        tokens = sample["input_ids"]
        if len(tokens) >= config.dataset_prepared_min_length:
            all_tokens.extend(tokens)
        else:
            chunk_dropped_tokens += len(tokens)
    
    # Create complete sequences from all available tokens
    chunk_sequences = []
    while len(all_tokens) >= config.sequence_len:
        sequence = all_tokens[:config.sequence_len]
        all_tokens = all_tokens[config.sequence_len:]
        chunk_sequences.append({
            "input_ids": sequence,
            "attention_mask": [1] * config.sequence_len
        })
    
    # Update carryover tokens for next chunk
    carryover_tokens.clear()
    carryover_tokens.extend(all_tokens)
    
    # Save chunk to disk immediately if we have sequences
    sequences_count = 0
    if chunk_sequences:
        result_dataset = Dataset.from_list(chunk_sequences)
        chunk_path = f"{temp_dir}/chunk_{chunk_idx}.parquet"
        result_dataset.to_parquet(chunk_path)
        sequences_count = len(chunk_sequences)
        
        # Clear memory immediately
        del chunk_sequences, result_dataset
    
    # Clear tokenized data
    del temp_chunk_dataset, tokenized_chunk
    
    return sequences_count, chunk_dropped_tokens


def _combine_chunks_streaming(chunk_files, output_path, monitor):
    """Combine chunk files using streaming approach to minimize memory usage."""
    import os
    from datasets import Dataset, concatenate_datasets
    
    os.makedirs(output_path, exist_ok=True)
    
    # Combine chunks in small groups to control memory usage
    all_data = []
    batch_size = 2  # Combine only 2 chunks at a time to minimize memory
    
    for i in range(0, len(chunk_files), batch_size):
        batch_files = chunk_files[i:i+batch_size]
        
        # Load batch of chunks
        batch_datasets = []
        for chunk_file in batch_files:
            chunk_data = Dataset.from_parquet(chunk_file)
            batch_datasets.append(chunk_data)
        
        # Combine batch
        if len(batch_datasets) == 1:
            combined_batch = batch_datasets[0]
        else:
            combined_batch = concatenate_datasets(batch_datasets)
        
        all_data.append(combined_batch)
        
        # Clear batch from memory
        del batch_datasets, combined_batch
        monitor.report_current(f"combined chunks {i} to {i+len(batch_files)-1}")
    
    # Final combination
    if len(all_data) == 1:
        final_dataset = all_data[0]
    else:
        final_dataset = concatenate_datasets(all_data)
    
    # Save final dataset
    final_dataset.save_to_disk(output_path)
    
    # Clean up
    del all_data, final_dataset


def validate_preprocessed(config: KrillConfig):
    """Validate that all packed sequences match the expected sequence length."""
    import sys
    from datasets import load_from_disk
    
    # Both standard and memory-efficient modes now use the same format
    ds = load_from_disk(config.dataset_prepared_path)
    seq_len = config.sequence_len
    
    wrong_indices = [i for i, sample in enumerate(ds)
                     if len(sample.get("input_ids", [])) != seq_len]
    if wrong_indices:
        print(f"\033[1;41;97mValidation failed: Found {len(wrong_indices)} samples "
              f"with incorrect length (expected {seq_len}). Indices: {wrong_indices}\033[0m")
        sys.exit(1)
    else:
        print(
            f"\033[1;32mAll {len(ds)} samples have correct length {seq_len}.\033[0m")