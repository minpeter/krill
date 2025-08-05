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
    """Truly memory-efficient preprocessing with optimized parameters."""
    # Load and prepare raw datasets (same as standard mode, but with optimizations)
    from krill.utils.dataset_utils import load_and_prepare_raw_datasets
    monitor.report_current("before loading datasets")
    
    print(f"🧠 Memory-efficient mode: Using optimized dataset operations")
    
    # Load datasets with memory optimizations
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

    # Use reduced num_proc for memory efficiency (less parallel memory usage)
    num_proc = max(1, min(4, os.cpu_count() // 4))  # Use 1/4 of available cores instead of most cores
    print(f"Memory-efficient mode: Using {num_proc} processes for mapping (vs {max(1, os.cpu_count() - 8)} in standard mode)")

    monitor.report_current("before tokenization")
    
    # Use smaller batch size and disable caching for memory efficiency
    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=config.preprocess_chunk_size,  # Use configurable chunk size instead of default
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing (memory-efficient)",
        load_from_cache_file=False,  # Disable caching to save memory
        keep_in_memory=False  # Don't keep in memory
    )
    monitor.report_current("after tokenization")

    # Filter by min_length with memory-efficient operations
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
    
    # Use select operation without caching
    tokenized = tokenized.select(selected)
    monitor.report_current("after length filtering")

    _pack_and_save_dataset_memory_efficient(config, tokenized, filter_dropped_tokens, monitor)


def _pack_and_save_dataset_memory_efficient(config: KrillConfig, tokenized, filter_dropped_tokens, monitor: MemoryMonitor):
    """Memory-efficient pack and save with optimized parameters."""
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    
    # Pack sequences with memory-optimized parameters
    print(f"Packing into sequences of length {config.sequence_len} (memory-efficient mode)...", end="")
    
    # Use smaller batch size for memory efficiency
    batch_size = min(config.preprocess_chunk_size // 2, 1000)  # Even smaller batch size
    print(f" using batch_size={batch_size}")
    
    monitor.report_current("before packing")
    packed = pack_dataset(tokenized,
                          seq_length=config.sequence_len,
                          strategy="wrapped",
                          map_kwargs={
                              "batch_size": batch_size,
                              "load_from_cache_file": False,  # Disable caching
                              "keep_in_memory": False  # Don't keep in memory
                          })
    monitor.report_current("after packing")
    
    # Filter out incomplete samples efficiently
    last_dropped_chunk_length = 0
    if len(packed) > 0:
        # Check all samples and filter out incorrect lengths
        correct_indices = []
        incorrect_tokens = 0
        
        for i, sample in enumerate(packed):
            if len(sample["input_ids"]) == config.sequence_len:
                correct_indices.append(i)
            else:
                incorrect_tokens += len(sample["input_ids"])
        
        if len(correct_indices) < len(packed):
            dropped_samples = len(packed) - len(correct_indices)
            last_dropped_chunk_length = incorrect_tokens
            packed = packed.select(correct_indices)
            print(f"Dropped {dropped_samples} incomplete samples ({incorrect_tokens} tokens)")

    # Save with optimized parameters
    monitor.report_current("before saving")
    
    print(f"Saving dataset with optimized memory settings...")
    # Use minimal shard size and single process to reduce memory usage
    packed.save_to_disk(
        config.dataset_prepared_path, 
        max_shard_size=config.preprocess_save_shard_size,
        num_proc=1  # Single process to minimize memory usage
    )
    
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
