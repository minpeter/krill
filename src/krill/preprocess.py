import os

from transformers import AutoTokenizer
from trl import pack_dataset

from krill.utils.config import KrillConfig
from krill.utils.dataset_utils import inspect_pretrain_dataset
from krill.utils.memory_monitor import MemoryMonitor


def do_preprocess(config: KrillConfig):
    """Preprocesses the data based on the loaded Config object."""
    print("🦐 Krill: Starting preprocessing...")

    monitor = MemoryMonitor()
    monitor.start_monitoring()

    if config.preprocess_memory_efficient:
        _do_preprocess_memory_efficient(config, monitor)
    else:
        _do_preprocess_standard(config, monitor)

    monitor.report_final()


def _do_preprocess_memory_efficient(config: KrillConfig, monitor: MemoryMonitor):
    """
    Preprocesses the data in a truly memory-efficient way.
    
    This processes each chunk through the complete pipeline independently (tokenize -> filter -> pack -> save)
    and uses incremental Apache Arrow storage to avoid any memory accumulation between chunks.
    """
    print(f"🔄 Memory-efficient streaming mode: {config.preprocess_chunk_size=}")

    # Prepare output directory
    os.makedirs(config.dataset_prepared_path, exist_ok=True)

    # Load tokenizer (this is small and doesn't change memory usage significantly)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)

    # Load and prepare datasets in streaming mode
    monitor.report_current("before loading datasets")
    
    # Use streaming to avoid loading all raw data at once
    from krill.utils.dataset_utils import _load_raw_datasets_streaming
    dataset_iterator = _load_raw_datasets_streaming(config.datasets)
    
    monitor.report_current("after loading datasets")

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

    monitor.report_current("before chunk processing")
    
    # Process chunks completely independently with incremental storage
    from datasets import Dataset, concatenate_datasets
    from trl import pack_dataset
    import tempfile
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Statistics tracking
    total_tokens_before_filter = 0
    total_tokens_after_filter = 0
    total_original_rows = 0
    
    # Use Apache Arrow for incremental storage
    parquet_files = []
    
    # Process in chunks
    chunk_size = config.preprocess_chunk_size
    current_chunk = []
    chunk_count = 0
    
    # Create output directory for incremental files
    incremental_dir = os.path.join(config.dataset_prepared_path + "_incremental")
    os.makedirs(incremental_dir, exist_ok=True)
    
    try:
        for example in dataset_iterator:
            current_chunk.append(example)
            total_original_rows += 1
            
            if len(current_chunk) >= chunk_size:
                # Process this chunk completely and save immediately
                chunk_stats = _process_chunk_completely_independent(
                    current_chunk, chunk_count, tokenize_function, config, 
                    incremental_dir, monitor
                )
                
                # Update statistics
                total_tokens_before_filter += chunk_stats["tokens_before_filter"]
                total_tokens_after_filter += chunk_stats["tokens_after_filter"]
                
                # Track parquet file if chunk produced data
                if chunk_stats["parquet_file"]:
                    parquet_files.append(chunk_stats["parquet_file"])
                
                # Clear chunk data immediately - this is key for memory efficiency
                current_chunk = []
                chunk_count += 1
                
                # Report memory usage occasionally
                if chunk_count % 10 == 0:
                    monitor.report_current(f"after processing chunk {chunk_count}")
        
        # Process final chunk if any
        if current_chunk:
            chunk_stats = _process_chunk_completely_independent(
                current_chunk, chunk_count, tokenize_function, config,
                incremental_dir, monitor, is_final=True
            )
            
            # Update statistics
            total_tokens_before_filter += chunk_stats["tokens_before_filter"]
            total_tokens_after_filter += chunk_stats["tokens_after_filter"]
            
            # Track parquet file if chunk produced data
            if chunk_stats["parquet_file"]:
                parquet_files.append(chunk_stats["parquet_file"])

        monitor.report_current("after all chunk processing")

        # Combine parquet files into final dataset using memory-efficient concatenation
        monitor.report_current("before combining parquet files")
        final_dataset = _combine_parquet_files_efficiently(parquet_files, monitor)
        monitor.report_current("after combining parquet files")

        # Final cleanup and save
        monitor.report_current("before final save")
        
        # Handle incomplete final chunk
        last_dropped_chunk_length = 0
        if len(final_dataset) > 0:
            last_len = len(final_dataset[-1]["input_ids"])
            if last_len < config.sequence_len:
                last_dropped_chunk_length = last_len
                final_dataset = final_dataset.select(list(range(len(final_dataset) - 1)))

        final_dataset.save_to_disk(config.dataset_prepared_path)
        monitor.report_current("after final save")

        # Compute filter dropped tokens
        filter_dropped_tokens = total_tokens_before_filter - total_tokens_after_filter

        inspect_pretrain_dataset(dataset=final_dataset,
                                 tokenizer=tokenizer,
                                 show_example_rows_limit=1)

        print(
            f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
        )
        print(
            f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from final incomplete chunk (length {last_dropped_chunk_length} < sequence_len={config.sequence_len})"
        )

        print(f"\nOriginal dataset rows: {total_original_rows}")
        print(f"Packed dataset rows: {len(final_dataset)}")
        total_tokens = sum(len(sample["input_ids"]) for sample in final_dataset)
        print(
            f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m")

        if len(final_dataset) > 0:
            # Check for any samples that do not match the expected context length
            wrong_indices = [
                i for i, sample in enumerate(final_dataset)
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
            f"🦐 Krill: Finished memory-efficient processing. Data saved to {config.dataset_prepared_path}"
        )
        
    finally:
        # Cleanup incremental directory
        if os.path.exists(incremental_dir):
            shutil.rmtree(incremental_dir)


def _process_chunk_completely_independent(chunk_data, chunk_idx, tokenize_function, config, output_dir, monitor, is_final=False):
    """
    Process a single chunk through the complete pipeline independently and save as Parquet.
    This ensures no memory accumulation between chunks.
    """
    from datasets import Dataset
    from trl import pack_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    chunk_name = f"final_chunk" if is_final else f"chunk_{chunk_idx:04d}"
    
    # Create dataset from chunk
    chunk_dataset = Dataset.from_list(chunk_data)
    
    # Tokenize chunk
    tokenized_chunk = chunk_dataset.map(tokenize_function,
                                      batched=True,
                                      remove_columns=chunk_dataset.column_names,
                                      desc=f"Tokenizing {chunk_name}")
    
    # Filter by min_length
    lengths = [len(x) for x in tokenized_chunk["input_ids"]]
    chunk_tokens_before_filter = sum(lengths)
    
    # Select valid samples 
    valid_indices = [i for i, length in enumerate(lengths) if length >= config.dataset_prepared_min_length]
    
    parquet_file = None
    chunk_tokens_after_filter = 0
    
    if valid_indices:
        filtered_chunk = tokenized_chunk.select(valid_indices)
        chunk_tokens_after_filter = sum(len(filtered_chunk[i]["input_ids"]) for i in range(len(filtered_chunk)))
        
        # Pack the filtered chunk
        packed_chunk = pack_dataset(filtered_chunk,
                                  seq_length=config.sequence_len,
                                  strategy="wrapped",
                                  map_kwargs={"batch_size": len(filtered_chunk)})
        
        # Save packed chunk as Parquet (memory-efficient format)
        if len(packed_chunk) > 0:
            parquet_file = os.path.join(output_dir, f"{chunk_name}.parquet")
            
            # Convert to Arrow table and save as Parquet
            arrow_table = packed_chunk.data.table
            pq.write_table(arrow_table, parquet_file, compression='snappy')
        
        # Clean up intermediate data immediately
        del chunk_dataset
        del tokenized_chunk
        del filtered_chunk
        del packed_chunk
        
    else:
        # No valid samples in this chunk
        # Clean up intermediate data
        del chunk_dataset
        del tokenized_chunk
    
    return {
        "tokens_before_filter": chunk_tokens_before_filter,
        "tokens_after_filter": chunk_tokens_after_filter,
        "parquet_file": parquet_file
    }


def _combine_parquet_files_efficiently(parquet_files, monitor):
    """
    Combine Parquet files into a single dataset using memory-efficient Arrow operations.
    """
    from datasets import Dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    if not parquet_files:
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    
    # Read all parquet files and concatenate using Arrow (memory efficient)
    arrow_tables = []
    for pq_file in parquet_files:
        if os.path.exists(pq_file):
            table = pq.read_table(pq_file)
            arrow_tables.append(table)
    
    if not arrow_tables:
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    elif len(arrow_tables) == 1:
        combined_table = arrow_tables[0]
    else:
        # Concatenate Arrow tables (very memory efficient)
        combined_table = pa.concat_tables(arrow_tables)
    
    # Convert back to Dataset
    return Dataset(combined_table)


def _process_chunk_for_tokens(chunk_data, chunk_idx, tokenize_function, config, monitor, is_final=False):
    """
    Process a single chunk for tokenization and filtering only.
    Returns filtered tokens without packing.
    """
    from datasets import Dataset
    
    chunk_name = f"final_chunk" if is_final else f"chunk_{chunk_idx:04d}"
    
    # Create dataset from chunk
    chunk_dataset = Dataset.from_list(chunk_data)
    
    # Tokenize chunk
    tokenized_chunk = chunk_dataset.map(tokenize_function,
                                      batched=True,
                                      remove_columns=chunk_dataset.column_names,
                                      desc=f"Tokenizing {chunk_name}")
    
    # Filter by min_length and extract tokens
    lengths = [len(x) for x in tokenized_chunk["input_ids"]]
    chunk_tokens_before_filter = sum(lengths)
    
    # Extract valid samples
    filtered_input_ids = []
    filtered_attention_masks = []
    chunk_tokens_after_filter = 0
    
    for i, length in enumerate(lengths):
        if length >= config.dataset_prepared_min_length:
            filtered_input_ids.append(tokenized_chunk[i]["input_ids"])
            filtered_attention_masks.append(tokenized_chunk[i]["attention_mask"])
            chunk_tokens_after_filter += length
    
    # Clean up intermediate data immediately
    del chunk_dataset
    del tokenized_chunk
    
    return {
        "tokens_before_filter": chunk_tokens_before_filter,
        "tokens_after_filter": chunk_tokens_after_filter,
        "filtered_input_ids": filtered_input_ids,
        "filtered_attention_masks": filtered_attention_masks
    }


def _pack_and_save_buffer(rolling_buffer, config, temp_dir, packed_chunk_idx, monitor):
    """
    Pack the current buffer and save it to disk.
    """
    from datasets import Dataset
    from trl import pack_dataset
    
    if not rolling_buffer["input_ids"]:
        return {"packed_samples": 0, "packed_tokens": 0}
    
    # Create dataset from buffer
    buffer_dataset = Dataset.from_dict(rolling_buffer)
    
    # Pack the buffer
    packed_chunk = pack_dataset(buffer_dataset,
                              seq_length=config.sequence_len,
                              strategy="wrapped",
                              map_kwargs={"batch_size": len(buffer_dataset)})
    
    # Save packed chunk to temporary location
    chunk_save_path = os.path.join(temp_dir, f"packed_{packed_chunk_idx:04d}")
    packed_chunk.save_to_disk(chunk_save_path)
    
    # Calculate statistics
    packed_samples = len(packed_chunk)
    packed_tokens = sum(len(sample["input_ids"]) for sample in packed_chunk)
    
    # Clean up
    del buffer_dataset
    del packed_chunk
    
    return {
        "packed_samples": packed_samples,
        "packed_tokens": packed_tokens
    }


def _reset_buffer_with_remainder(rolling_buffer, sequence_len):
    """
    Reset buffer but keep some tokens to maintain continuity.
    Keep the last few sequences worth of tokens.
    """
    total_tokens = sum(len(ids) for ids in rolling_buffer["input_ids"])
    keep_tokens = sequence_len * 2  # Keep 2 sequences worth of tokens
    
    if total_tokens <= keep_tokens:
        # Keep everything if buffer is small
        return
    
    # Find the point to cut at
    accumulated = 0
    cut_point = len(rolling_buffer["input_ids"])
    
    for i in range(len(rolling_buffer["input_ids"]) - 1, -1, -1):
        accumulated += len(rolling_buffer["input_ids"][i])
        if accumulated >= keep_tokens:
            cut_point = i
            break
    
    # Keep only the last portion
    rolling_buffer["input_ids"] = rolling_buffer["input_ids"][cut_point:]
    rolling_buffer["attention_mask"] = rolling_buffer["attention_mask"][cut_point:]


def _process_chunk_independently(chunk_data, chunk_idx, tokenize_function, config, temp_dir, monitor, is_final=False):
    """
    Process a single chunk through the complete pipeline: tokenize -> filter -> pack -> save.
    Returns statistics about the processed chunk.
    """
    from datasets import Dataset
    from trl import pack_dataset
    
    chunk_name = f"final_chunk" if is_final else f"chunk_{chunk_idx:04d}"
    
    # Create dataset from chunk
    chunk_dataset = Dataset.from_list(chunk_data)
    
    # Tokenize chunk
    tokenized_chunk = chunk_dataset.map(tokenize_function,
                                      batched=True,
                                      remove_columns=chunk_dataset.column_names,
                                      desc=f"Tokenizing {chunk_name}")
    
    # Filter by min_length
    lengths = [len(x) for x in tokenized_chunk["input_ids"]]
    chunk_tokens_before_filter = sum(lengths)
    
    # Select valid samples 
    valid_indices = [i for i, length in enumerate(lengths) if length >= config.dataset_prepared_min_length]
    
    if valid_indices:
        filtered_chunk = tokenized_chunk.select(valid_indices)
        chunk_tokens_after_filter = sum(len(filtered_chunk[i]["input_ids"]) for i in range(len(filtered_chunk)))
        
        # Pack the filtered chunk
        packed_chunk = pack_dataset(filtered_chunk,
                                  seq_length=config.sequence_len,
                                  strategy="wrapped",
                                  map_kwargs={"batch_size": len(filtered_chunk)})
        
        # Save packed chunk to temporary location
        chunk_save_path = os.path.join(temp_dir, chunk_name)
        packed_chunk.save_to_disk(chunk_save_path)
        
        # Calculate statistics
        packed_samples = len(packed_chunk)
        packed_tokens = sum(len(sample["input_ids"]) for sample in packed_chunk)
        
        # Clean up intermediate data immediately
        del chunk_dataset
        del tokenized_chunk
        del filtered_chunk
        del packed_chunk
        
    else:
        # No valid samples in this chunk
        chunk_tokens_after_filter = 0
        packed_samples = 0
        packed_tokens = 0
        
        # Clean up intermediate data
        del chunk_dataset
        del tokenized_chunk
    
    return {
        "tokens_before_filter": chunk_tokens_before_filter,
        "tokens_after_filter": chunk_tokens_after_filter,
        "packed_samples": packed_samples,
        "packed_tokens": packed_tokens
    }


def _combine_processed_chunks(temp_chunks_dir, config, monitor):
    """
    Combine all processed chunks into a single dataset using concatenation.
    """
    from datasets import Dataset, concatenate_datasets
    import os
    
    # Find all chunk directories
    chunk_dirs = [d for d in os.listdir(temp_chunks_dir) 
                  if os.path.isdir(os.path.join(temp_chunks_dir, d))]
    chunk_dirs.sort()  # Ensure consistent ordering
    
    if not chunk_dirs:
        # Return empty dataset if no chunks
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    
    # Load and concatenate all chunks
    datasets_to_combine = []
    for chunk_dir in chunk_dirs:
        chunk_path = os.path.join(temp_chunks_dir, chunk_dir)
        try:
            chunk_dataset = Dataset.load_from_disk(chunk_path)
            if len(chunk_dataset) > 0:  # Only add non-empty chunks
                datasets_to_combine.append(chunk_dataset)
        except Exception as e:
            print(f"Warning: Failed to load chunk {chunk_dir}: {e}")
    
    if not datasets_to_combine:
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    elif len(datasets_to_combine) == 1:
        return datasets_to_combine[0]
    else:
        # Concatenate all datasets
        combined = concatenate_datasets(datasets_to_combine)
        
        # Clean up loaded datasets from memory
        for ds in datasets_to_combine:
            del ds
        
        return combined


def _do_preprocess_standard(config: KrillConfig, monitor: MemoryMonitor):
    # Prepare output directory
    os.makedirs(config.dataset_prepared_path, exist_ok=True)

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
    monitor.report_current("before filtering")
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
    monitor.report_current("before packing")
    print(f"Packing into sequences of length {config.sequence_len}...", end="")
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
