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
    """Memory-efficient streaming preprocessing with bounded memory usage."""
    print("🧠 Memory-efficient streaming mode")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    
    # Initialize collections for final dataset
    from datasets import Dataset
    final_sequences = []
    total_filter_dropped_tokens = 0
    carryover_tokens = []
    chunk_size = getattr(config, 'preprocess_chunk_size', 500)
    
    monitor.report_current("before starting chunked processing")
    
    # Process each dataset in chunks
    for ds_cfg in config.datasets:
        print(f"🧠 Processing dataset {ds_cfg.path}:{ds_cfg.split} with chunk size {chunk_size}")
        
        # Determine total size and process in chunks
        from krill.utils.dataset_utils import load_dataset_single
        
        # Handle slice notation in split
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
        
        # Load dataset to get total size for chunking
        from krill.utils.config import DatasetConfig
        temp_config = DatasetConfig(
            path=ds_cfg.path,
            split=base_split,
            text_column=ds_cfg.text_column,
            name=getattr(ds_cfg, 'name', None)
        )
        temp_dataset = load_dataset_single(temp_config)
        total_size = len(temp_dataset)
        
        # Calculate actual range
        actual_start = start_idx
        actual_end = min(end_idx, total_size) if end_idx is not None else total_size
        
        # Process dataset in chunks
        current_idx = actual_start
        while current_idx < actual_end:
            chunk_end = min(current_idx + chunk_size, actual_end)
            print(f"🧠 Processing chunk [{current_idx}:{chunk_end}]")
            
            # Load only this chunk
            chunk_split = f"{base_split}[{current_idx}:{chunk_end}]"
            chunk_config = DatasetConfig(
                path=ds_cfg.path,
                split=chunk_split,
                text_column=ds_cfg.text_column,
                name=getattr(ds_cfg, 'name', None)
            )
            chunk_dataset = load_dataset_single(chunk_config)
            
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
            
            tokenized_chunk = chunk_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=100,
                num_proc=1,
                remove_columns=chunk_dataset.column_names,
                desc=f"Tokenizing chunk [{current_idx}:{chunk_end}]",
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
            
            total_filter_dropped_tokens += chunk_dropped_tokens
            
            # Create complete sequences from all available tokens
            while len(all_tokens) >= config.sequence_len:
                sequence = all_tokens[:config.sequence_len]
                all_tokens = all_tokens[config.sequence_len:]
                final_sequences.append({
                    "input_ids": sequence,
                    "attention_mask": [1] * config.sequence_len
                })
            
            # Save remaining tokens for next chunk
            carryover_tokens = all_tokens
            
            # Clear chunk data to free memory
            del chunk_dataset, tokenized_chunk
            monitor.report_current(f"after processing chunk [{current_idx}:{chunk_end}]")
            
            current_idx = chunk_end
    
    # Calculate final statistics
    final_carryover_tokens = len(carryover_tokens)
    
    # Create final dataset
    monitor.report_current("before creating final dataset")
    if final_sequences:
        packed = Dataset.from_list(final_sequences)
    else:
        packed = Dataset.from_list([])
    
    # Save to disk using same format as standard mode
    monitor.report_current("before saving")
    packed.save_to_disk(config.dataset_prepared_path)
    monitor.report_current("after saving")

    # Display results in same format as standard mode
    inspect_pretrain_dataset(dataset=packed,
                             tokenizer=tokenizer,
                             show_example_rows_limit=1)

    print(
        f"\n\033[1;41;97mDropped {total_filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {final_carryover_tokens} tokens\033[0m from final incomplete chunk (length {final_carryover_tokens} < sequence_len={config.sequence_len})"
    )

    # Calculate original dataset rows from all chunks processed
    original_rows = actual_end - actual_start
    
    print(f"\nOriginal dataset rows: {original_rows}")
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