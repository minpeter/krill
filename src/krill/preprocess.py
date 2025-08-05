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
    """Memory-efficient preprocessing implementation."""
    from krill.utils.memory_efficient_dataset import (
        process_datasets_memory_efficient,
        tokenize_in_chunks,
        filter_by_length_memory_efficient
    )
    
    # Set up deduplication cache directory
    if config.preprocess_dedup_cache_dir:
        cache_dir = config.preprocess_dedup_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = None  # Will use temporary directory

    print(f"Processing datasets in chunks of {config.preprocess_chunk_size}")
    
    # Load and prepare raw datasets in chunks
    monitor.report_current("before chunked dataset processing")
    raw_dataset = process_datasets_memory_efficient(
        config.datasets, 
        chunk_size=config.preprocess_chunk_size,
        cache_dir=cache_dir
    )
    monitor.report_current("after chunked dataset processing")

    if len(raw_dataset) == 0:
        print("No samples remaining after filtering. Exiting.")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)

    # Tokenize in chunks
    print(f"Tokenizing dataset in chunks...")
    monitor.report_current("before chunked tokenization")
    tokenized = tokenize_in_chunks(
        raw_dataset, 
        tokenizer, 
        chunk_size=config.preprocess_chunk_size
    )
    monitor.report_current("after chunked tokenization")

    # Filter by length in chunks
    print(f"Filtering by minimum length ({config.dataset_prepared_min_length})...")
    monitor.report_current("before chunked length filtering")
    tokenized, filter_dropped_tokens = filter_by_length_memory_efficient(
        tokenized, 
        config.dataset_prepared_min_length,
        chunk_size=config.preprocess_chunk_size
    )
    monitor.report_current("after chunked length filtering")

    if len(tokenized) == 0:
        print("No samples remaining after length filtering. Exiting.")
        return

    _pack_and_save_dataset(config, tokenized, filter_dropped_tokens, monitor)


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
