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
    """Memory-efficient streaming preprocessing using ArrowWriter."""
    print("🧠 Memory-efficient streaming mode")
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    monitor.report_current("after tokenizer load")
    from datasets import load_dataset, load_from_disk
    from datasets.arrow_writer import ArrowWriter

    seq_len = config.sequence_len
    batch_size = config.preprocess_chunk_size
    # Prepare arrow file for streaming output
    writer_file = os.path.join(config.dataset_prepared_path, "streaming.arrow")
    writer = ArrowWriter(path=writer_file,
                         writer_batch_size=batch_size)
    num_written = 0
    carryover = []
    total_dropped = 0

    import itertools
    for ds_cfg in config.datasets:
        print(f"🧠 Loading {ds_cfg.path}:{ds_cfg.split} as streaming dataset")
        # handle optional slice in split like 'train[start:end]' with underscores
        if '[' in ds_cfg.split and ds_cfg.split.endswith(']'):
            base_split, rng = ds_cfg.split.split('[', 1)
            rng = rng[:-1]  # remove trailing ]
            start_str, end_str = rng.split(':')
            start_idx = int(start_str.replace('_', '')) if start_str else 0
            end_idx = int(end_str.replace('_', ''))
            ds_full = load_dataset(ds_cfg.path,
                                   split=base_split,
                                   streaming=True)
            ds_stream = itertools.islice(ds_full, start_idx, end_idx)
        else:
            ds_stream = load_dataset(ds_cfg.path,
                                     split=ds_cfg.split,
                                     streaming=True)
        for example in ds_stream:
            toks = tokenizer(example["text"],
                             padding=False,
                             truncation=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                toks.append(tokenizer.eos_token_id)
            if len(toks) < config.dataset_prepared_min_length:
                total_dropped += len(toks)
                continue
            carryover.extend(toks)
            while len(carryover) >= seq_len:
                seq = carryover[:seq_len]
                carryover = carryover[seq_len:]
                writer.write_batch({"input_ids": seq,
                                    "attention_mask": [1] * seq_len})
                num_written += 1
    # finalize writing and drop leftover carryover tokens
    dropped_carry = len(carryover)
    if dropped_carry:
        print(f"🧠 Dropping final carryover: {dropped_carry} tokens")
    writer.finalize()
    monitor.report_current("after finalize")
    # Summary
    print(
        f"\n\033[1;41;97mDropped {total_dropped} tokens\033[0m during filtering (min_length={config.dataset_prepared_min_length})")
    if dropped_carry:
        print(
            f"\033[1;41;97mDropped {dropped_carry} carryover tokens\033[0m from final incomplete sequence")
    print(f"\nFinal dataset sequences: {num_written}")
    total_tokens = num_written * seq_len
    print(
        f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m")
    print(f"🦐 Krill: Finished. Data saved to {config.dataset_prepared_path}")


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
    # Start with carryover from previous chunk
    all_tokens = list(carryover_tokens)
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
        sequences.append(
            {"input_ids": sequence, "attention_mask": [1] * len(sequence)})

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

    monitor.report_current("before saving")
    packed.save_to_disk(config.dataset_prepared_path)
    monitor.report_current("after saving")

    inspect_pretrain_dataset(dataset=packed,
                             tokenizer=tokenizer,
                             show_example_rows_limit=1)
    # Compute length of final incomplete chunk dropped
    total_tokens_after_filter = sum(
        len(sample["input_ids"]) for sample in tokenized)
    last_dropped_chunk_length = total_tokens_after_filter - \
        len(packed) * config.sequence_len

    print(
        f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from final incomplete chunk (length {last_dropped_chunk_length} < sequence_len={config.sequence_len})"
    )

    print(f"\nOriginal dataset rows: {len(tokenized)}")
    print(f"Packed dataset rows: {len(packed)}")
    total_tokens = sum(len(sample["input_ids"]) for sample in packed)
    print(
        f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m"
    )

    print(
        f"🦐 Krill: Finished. Packed data saved to {config.dataset_prepared_path}"
    )
    inspect_pretrain_dataset(dataset=packed,
                             tokenizer=tokenizer,
                             show_example_rows_limit=1)

    print(
        f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from final incomplete chunk (length {last_dropped_chunk_length} < sequence_len={config.sequence_len})"
    )

    # Note: this is after initial processing
    print(f"\nOriginal dataset rows: {len(tokenized)}")
    print(f"Packed dataset rows: {len(packed)}")
    total_tokens = sum(len(sample["input_ids"]) for sample in packed)
    print(
        f"Total tokens in packed dataset: \033[45;97m{total_tokens / 1_000_000_000:.3f}B\033[0m")

    # Validation of sequence lengths has been moved to the `validate` subcommand

    print(
        f"🦐 Krill: Finished. Packed data saved to {config.dataset_prepared_path}"
    )
    # Validation removed; use separate validate_preprocessed() function.


def validate_preprocessed(config: KrillConfig):
    """Validate that all packed sequences match the expected sequence length."""
    import sys
    import os
    from datasets import load_from_disk, load_dataset
    # Load dataset: from disk for standard mode, from streaming arrow file for memory-efficient mode
    seq_len = config.sequence_len
    if getattr(config, 'preprocess_memory_efficient', False):
        # streaming validation: arrow file contains flat tokens
        arrow_file = os.path.join(
            config.dataset_prepared_path, 'streaming.arrow')
        stream = load_dataset('arrow', data_files=arrow_file,
                              split='train', streaming=True)
        token_count = 0
        for _ in stream:
            token_count += 1
        # check divisibility by sequence length
        if token_count % seq_len != 0:
            print(
                f"\033[1;41;97mValidation failed: Total tokens {token_count} not divisible by sequence length {seq_len}\033[0m")
            sys.exit(1)
        seq_count = token_count // seq_len
        print(
            f"\033[1;32mAll {seq_count} sequences validated with length {seq_len} ({token_count} tokens total).\033[0m")
    else:
        ds = load_from_disk(config.dataset_prepared_path)
        wrong_indices = [i for i, sample in enumerate(ds)
                         if len(sample.get("input_ids", [])) != seq_len]
        if wrong_indices:
            print(f"\033[1;41;97mValidation failed: Found {len(wrong_indices)} samples "
                  f"with incorrect length (expected {seq_len}). Indices: {wrong_indices}\033[0m")
            sys.exit(1)
        else:
            print(
                f"\033[1;32mAll {len(ds)} samples have correct length {seq_len}.\033[0m")
