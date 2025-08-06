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
    
    chunk_size = getattr(config, 'preprocess_chunk_size', 500)
    
    # Use incremental saving to avoid accumulating all sequences in memory
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    
    total_filter_dropped_tokens = 0
    total_original_rows = 0
    carryover_tokens = []  # Store incomplete tokens between chunks
    
    monitor.report_current("before starting streaming processing")
    
    try:
        # Process each dataset in streaming mode
        for ds_cfg in config.datasets:
            print(f"🧠 Streaming dataset {ds_cfg.path} with chunk size {chunk_size}")
            
            # Load dataset with streaming=True to avoid loading entire dataset into memory
            from datasets import load_dataset
            
            # Parse the split to handle slicing like train[:1000]
            base_split = ds_cfg.split
            slice_start, slice_end = None, None
            
            if '[' in ds_cfg.split and ds_cfg.split.endswith(']'):
                base_split, slice_part = ds_cfg.split.split('[', 1)
                slice_part = slice_part[:-1]  # Remove ']'
                if ':' in slice_part:
                    start_str, end_str = slice_part.split(':', 1)
                    slice_start = int(start_str.replace('_', '')) if start_str else 0
                    if end_str:
                        slice_end = int(end_str.replace('_', ''))
            
            # Load dataset in streaming mode
            dataset_stream = load_dataset(
                ds_cfg.path,
                name=getattr(ds_cfg, 'name', None),
                split=base_split,
                streaming=True
            )
            
            # Apply slicing if needed
            if slice_start is not None:
                dataset_stream = dataset_stream.skip(slice_start)
            if slice_end is not None:
                slice_length = slice_end - (slice_start or 0) 
                dataset_stream = dataset_stream.take(slice_length)
            
            # Process dataset in chunks
            current_chunk = []
            chunk_idx = 0
            
            for example in dataset_stream:
                # Clean and filter text (same as standard mode)
                text = example[ds_cfg.text_column]
                
                # Apply text cleaning (same as standard mode)
                text = _clean_text(text)
                
                # Apply basic quality filtering (same as standard mode) 
                if len(text) >= 100:  # Same min_length as used in standard mode filtering
                    current_chunk.append(text)
                
                # Process chunk when it reaches chunk_size
                if len(current_chunk) >= chunk_size:
                    chunk_file = _process_and_save_chunk(
                        current_chunk, tokenizer, config, carryover_tokens, 
                        chunk_idx, temp_dir
                    )
                    if chunk_file:
                        chunk_files.append(chunk_file)
                    
                    total_filter_dropped_tokens += _count_dropped_tokens(current_chunk, tokenizer, config)
                    total_original_rows += len(current_chunk)
                    
                    current_chunk = []
                    chunk_idx += 1
                    
                    # Report memory after each chunk - should stay bounded
                    monitor.report_current(f"after processing chunk {chunk_idx}")
            
            # Process final chunk if any
            if current_chunk:
                chunk_file = _process_and_save_chunk(
                    current_chunk, tokenizer, config, carryover_tokens,
                    chunk_idx, temp_dir
                )
                if chunk_file:
                    chunk_files.append(chunk_file)
                
                total_filter_dropped_tokens += _count_dropped_tokens(current_chunk, tokenizer, config)
                total_original_rows += len(current_chunk)
        
        # Calculate final carryover tokens that will be dropped
        final_carryover_tokens = len(carryover_tokens)
        
        # Combine all chunk files into final dataset
        monitor.report_current("before combining chunk files")
        packed = _combine_chunk_files(chunk_files, config.dataset_prepared_path)
        monitor.report_current("after combining chunk files")
        
        # Display results (exactly same as standard mode)
        inspect_pretrain_dataset(dataset=packed,
                                 tokenizer=tokenizer,
                                 show_example_rows_limit=1)

        print(
            f"\n\033[1;41;97mDropped {total_filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
        )
        print(
            f"\n\033[1;41;97mDropped {final_carryover_tokens} tokens\033[0m from final incomplete chunk (length {final_carryover_tokens} < sequence_len={config.sequence_len})"
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


def _clean_text(text):
    """Clean text using same logic as standard mode."""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        text = ""
    
    # Fix surrogates  
    import re
    text = re.sub(r'[\uD800-\uDFFF]', '', text)
    
    return text


def _count_dropped_tokens(texts, tokenizer, config):
    """Count tokens that would be dropped during filtering."""
    dropped = 0
    for text in texts:
        tokenized = tokenizer(text, padding=False, truncation=False)
        tokens = tokenized["input_ids"]
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
        if len(tokens) < config.dataset_prepared_min_length:
            dropped += len(tokens)
    return dropped


def _process_and_save_chunk(texts, tokenizer, config, carryover_tokens, chunk_idx, temp_dir):
    """Process a chunk of texts and save immediately to disk."""
    import os
    
    # Start with carryover tokens from previous chunk
    all_token_ids = list(carryover_tokens)
    
    for text in texts:
        # Tokenize individual text  
        tokenized = tokenizer(text, padding=False, truncation=False)
        tokens = tokenized["input_ids"]
        
        # Add EOS token if configured
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
        
        # Apply length filtering (same as standard mode)
        if len(tokens) >= config.dataset_prepared_min_length:
            all_token_ids.extend(tokens)
    
    # Pack tokens into complete sequences
    sequences = []
    while len(all_token_ids) >= config.sequence_len:
        sequence_tokens = all_token_ids[:config.sequence_len]
        all_token_ids = all_token_ids[config.sequence_len:]
        
        sequences.append({
            "input_ids": sequence_tokens,
            "attention_mask": [1] * config.sequence_len
        })
    
    # Update carryover tokens for next chunk
    carryover_tokens.clear()
    carryover_tokens.extend(all_token_ids)
    
    # Save sequences to file if any were created
    if sequences:
        from datasets import Dataset
        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.parquet")
        chunk_dataset = Dataset.from_list(sequences)
        chunk_dataset.to_parquet(chunk_path)
        
        # Clear memory immediately
        del sequences, chunk_dataset
        return chunk_path
    
    return None


def _combine_chunk_files(chunk_files, output_path):
    """Combine chunk files efficiently and save to final location."""
    from datasets import Dataset, concatenate_datasets
    import os
    
    if not chunk_files:
        # Create empty dataset
        empty_dataset = Dataset.from_list([])
        empty_dataset.save_to_disk(output_path)
        return empty_dataset
    
    # Load and combine chunk files in small batches to control memory
    datasets = []
    for chunk_file in chunk_files:
        chunk_data = Dataset.from_parquet(chunk_file)
        datasets.append(chunk_data)
    
    # Combine all datasets
    if len(datasets) == 1:
        final_dataset = datasets[0]
    else:
        final_dataset = concatenate_datasets(datasets)
    
    # Save to final location
    final_dataset.save_to_disk(output_path)
    
    return final_dataset


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