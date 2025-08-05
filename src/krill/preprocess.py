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
    """Truly memory-efficient preprocessing using streaming chunks."""
    print(f"🧠 Memory-efficient mode: Processing {config.preprocess_chunk_size} samples at a time")
    
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

    # Process each dataset in streaming chunks
    total_filter_dropped_tokens = 0
    total_carryover_tokens = 0
    carryover_tokens = []  # Tokens from previous incomplete chunk
    final_dataset = None
    
    for ds_idx, ds_cfg in enumerate(config.datasets):
        print(f"🧠 Processing dataset {ds_idx + 1}/{len(config.datasets)}: {ds_cfg.path}")
        
        # Parse the split to get total size
        split_info = ds_cfg.split
        if split_info.startswith("train[") and ":" in split_info:
            # Extract range like train[0:50000] or train[:1000]
            range_part = split_info[6:-1]  # Remove "train[" and "]"
            if range_part.startswith(":"):
                start_idx = 0
                end_idx = int(range_part[1:].replace("_", ""))
            elif range_part.endswith(":"):
                start_idx = int(range_part[:-1].replace("_", ""))
                # Cannot determine end, fallback to standard mode
                print(f"⚠️  Cannot determine dataset size for split '{split_info}', fallback to standard mode")
                _process_dataset_fallback(config, ds_cfg, tokenizer, tokenize_function, monitor)
                continue
            else:
                start_str, end_str = range_part.split(":")
                start_idx = int(start_str.replace("_", "")) if start_str else 0
                end_idx = int(end_str.replace("_", "")) if end_str else None
                if end_idx is None:
                    print(f"⚠️  Cannot determine dataset size for split '{split_info}', fallback to standard mode")
                    _process_dataset_fallback(config, ds_cfg, tokenizer, tokenize_function, monitor)
                    continue
            
            # Process in true streaming chunks
            chunk_size = config.preprocess_chunk_size
            print(f"🧠 Processing {end_idx - start_idx} samples in streaming chunks of {chunk_size}")
            
            for chunk_start in range(start_idx, end_idx, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end_idx)
                
                print(f"🧠 Processing chunk {chunk_start}-{chunk_end}...")
                monitor.report_current(f"before chunk {chunk_start}")
                
                # Load and process this chunk immediately
                chunk_dataset = _load_chunk_streaming(ds_cfg, chunk_start, chunk_end)
                monitor.report_current(f"after loading chunk {chunk_start}")
                
                # Process chunk through full pipeline with carryover
                processed_chunk, chunk_carryover, chunk_dropped_tokens = _process_chunk_with_carryover(
                    chunk_dataset, carryover_tokens, tokenize_function, config, monitor
                )
                
                total_filter_dropped_tokens += chunk_dropped_tokens
                carryover_tokens = chunk_carryover
                
                # Save chunk immediately if it has data
                if processed_chunk is not None and len(processed_chunk) > 0:
                    if final_dataset is None:
                        final_dataset = processed_chunk
                    else:
                        # Append to existing dataset
                        from datasets import concatenate_datasets
                        final_dataset = concatenate_datasets([final_dataset, processed_chunk])
                    
                    # Clear processed chunk immediately
                    del processed_chunk
                
                # Clear chunk from memory
                del chunk_dataset
                monitor.report_current(f"after processing chunk {chunk_start}")
        else:
            # Handle other split formats (fallback to standard loading)
            print(f"⚠️  Split format '{split_info}' not supported for streaming, fallback to standard mode")
            _process_dataset_fallback(config, ds_cfg, tokenizer, tokenize_function, monitor)

    # Handle final carryover tokens
    if carryover_tokens:
        total_carryover_tokens = len(carryover_tokens)
        print(f"🧠 Final carryover: {total_carryover_tokens} tokens will be dropped")
    
    if final_dataset is None or len(final_dataset) == 0:
        print("❌ No data remaining after processing")
        return
    
    # Save the final dataset
    monitor.report_current("before final save")
    final_dataset.save_to_disk(config.dataset_prepared_path, num_proc=1)
    monitor.report_current("after final save")
    
    # Final reporting
    from krill.utils.inspect_dataset import inspect_pretrain_dataset
    inspect_pretrain_dataset(dataset=final_dataset, tokenizer=tokenizer, show_example_rows_limit=1)
    
    print(f"\n\033[1;41;97mDropped {total_filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})")
    print(f"\n\033[1;41;97mDropped {total_carryover_tokens} tokens\033[0m from final incomplete chunk")
    
    print(f"\nFinal dataset rows: {len(final_dataset)}")
    total_tokens = sum(len(sample["input_ids"]) for sample in final_dataset)
    print(f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m")
    
    print(f"🦐 Krill: Finished memory-efficient preprocessing. Data saved to {config.dataset_prepared_path}")


def _load_chunk_streaming(ds_cfg, start_idx, end_idx):
    """Load a small chunk of data directly without loading the entire dataset."""
    from krill.utils.dataset_utils import load_dataset_single
    from krill.utils.config import DatasetConfig
    
    chunk_split = f"train[{start_idx}:{end_idx}]"
    chunk_config = DatasetConfig(
        path=ds_cfg.path,
        split=chunk_split,
        text_column=ds_cfg.text_column,
        name=ds_cfg.name
    )
    return load_dataset_single(chunk_config)


def _process_chunk_with_carryover(chunk_dataset, carryover_tokens, tokenize_function, config, monitor):
    """Process a chunk with carryover logic for incomplete sequences."""
    from trl import pack_dataset
    
    # Tokenize the chunk
    tokenized_chunk = chunk_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,  # Small batch for memory efficiency
        num_proc=1,
        remove_columns=chunk_dataset.column_names,
        desc="Tokenizing chunk",
        load_from_cache_file=False,
        keep_in_memory=False
    )
    
    # Filter by length and collect all tokens
    all_tokens = list(carryover_tokens)  # Start with carryover from previous chunk
    chunk_dropped_tokens = 0
    
    for sample in tokenized_chunk:
        tokens = sample["input_ids"]
        if len(tokens) >= config.dataset_prepared_min_length:
            all_tokens.extend(tokens)
        else:
            chunk_dropped_tokens += len(tokens)
    
    # If no tokens to process, return empty
    if len(all_tokens) < config.sequence_len:
        return None, all_tokens, chunk_dropped_tokens
    
    # Pack tokens into sequences
    # Calculate how many complete sequences we can make
    num_complete_sequences = len(all_tokens) // config.sequence_len
    if num_complete_sequences == 0:
        return None, all_tokens, chunk_dropped_tokens
    
    # Create complete sequences
    sequences = []
    for i in range(num_complete_sequences):
        start_pos = i * config.sequence_len
        end_pos = start_pos + config.sequence_len
        sequence = all_tokens[start_pos:end_pos]
        sequences.append({"input_ids": sequence, "attention_mask": [1] * len(sequence)})
    
    # Calculate carryover tokens for next chunk
    used_tokens = num_complete_sequences * config.sequence_len
    carryover_tokens = all_tokens[used_tokens:]
    
    # Create dataset from sequences
    if sequences:
        from datasets import Dataset
        processed_chunk = Dataset.from_list(sequences)
        return processed_chunk, carryover_tokens, chunk_dropped_tokens
    else:
        return None, carryover_tokens, chunk_dropped_tokens


def _process_dataset_fallback(config, ds_cfg, tokenizer, tokenize_function, monitor):
    """Fallback to standard processing for unsupported split formats."""
    print(f"⚠️  Using fallback processing for dataset: {ds_cfg.path}")
    from krill.utils.dataset_utils import load_dataset_single
    
    raw_dataset = load_dataset_single(ds_cfg)
    monitor.report_current("after fallback dataset load")
    
    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=config.preprocess_chunk_size,
        num_proc=1,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset (fallback)",
        load_from_cache_file=False,
        keep_in_memory=False
    )
    
    # This is still a fallback that loads everything - 
    # in practice, datasets should use proper split formats for memory efficiency
    monitor.report_current("after fallback processing")
    




def _pack_and_save_dataset(config: KrillConfig, tokenized, filter_dropped_tokens, monitor: MemoryMonitor):
    """Pack and save the processed dataset."""
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    
    # Pack sequences
    print(f"Packing into sequences of length {config.sequence_len}...", end="")
    
    monitor.report_current("before packing")
    packed = pack_dataset(tokenized,
                          seq_length=config.sequence_len,
                          strategy="wrapped")
    monitor.report_current("after packing")
    
    # Drop incomplete samples and record dropped tokens
    last_dropped_chunk_length = 0
    if len(packed) > 0:
        # Only drop the last sample if incomplete
        last_len = len(packed[-1]["input_ids"])
        if last_len < config.sequence_len:
            last_dropped_chunk_length = last_len
            packed = packed.select(list(range(len(packed) - 1)))

    # Save dataset
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
