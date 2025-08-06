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
    """Preprocesses the data in a memory-efficient way."""

    print(f"{config.preprocess_chunk_size=}, {config.preprocess_memory_efficient=}")

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

    monitor.report_current("before tokenization")
    
    # Process all data, but in smaller chunks to keep peak memory lower
    from datasets import Dataset
    from trl import pack_dataset
    
    total_tokens_before_filter = 0
    total_tokens_after_filter = 0
    total_original_rows = 0
    
    # Collect all filtered tokenized data efficiently
    all_input_ids = []
    all_attention_masks = []
    
    # Process in smaller chunks to avoid peak memory spikes
    chunk_size = config.preprocess_chunk_size
    current_chunk = []
    chunk_count = 0
    
    for example in dataset_iterator:
        current_chunk.append(example)
        total_original_rows += 1
        
        if len(current_chunk) >= chunk_size:
            # Process this chunk
            chunk_dataset = Dataset.from_list(current_chunk)
            
            # Tokenize chunk
            tokenized_chunk = chunk_dataset.map(tokenize_function,
                                              batched=True,
                                              remove_columns=chunk_dataset.column_names,
                                              desc=f"Tokenizing chunk {chunk_count}")
            
            # Filter by min_length and collect results immediately
            lengths = [len(x) for x in tokenized_chunk["input_ids"]]
            chunk_tokens_before = sum(lengths)
            total_tokens_before_filter += chunk_tokens_before
            
            for i, length in enumerate(lengths):
                if length >= config.dataset_prepared_min_length:
                    all_input_ids.append(tokenized_chunk[i]["input_ids"])
                    all_attention_masks.append(tokenized_chunk[i]["attention_mask"])
                    total_tokens_after_filter += length
            
            # Clear intermediate data immediately to reduce memory
            del tokenized_chunk
            del chunk_dataset
            current_chunk = []
            chunk_count += 1
            
            # Report memory usage occasionally 
            if chunk_count % 10 == 0:
                monitor.report_current(f"after processing chunk {chunk_count}")
    
    # Process final chunk if any
    if current_chunk:
        chunk_dataset = Dataset.from_list(current_chunk)
        
        tokenized_chunk = chunk_dataset.map(tokenize_function,
                                          batched=True,
                                          remove_columns=chunk_dataset.column_names,
                                          desc=f"Tokenizing final chunk")
        
        lengths = [len(x) for x in tokenized_chunk["input_ids"]]
        chunk_tokens_before = sum(lengths)
        total_tokens_before_filter += chunk_tokens_before
        
        for i, length in enumerate(lengths):
            if length >= config.dataset_prepared_min_length:
                all_input_ids.append(tokenized_chunk[i]["input_ids"])
                all_attention_masks.append(tokenized_chunk[i]["attention_mask"])
                total_tokens_after_filter += length
        
        del tokenized_chunk
        del chunk_dataset

    monitor.report_current("after tokenization")

    # Now create the combined dataset for packing
    monitor.report_current("before packing")
    if all_input_ids:
        combined_tokenized = Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        })
        
        # Pack the combined dataset (this should produce identical results to standard mode)
        print(f"Packing into sequences of length {config.sequence_len}...", end="")
        packed = pack_dataset(combined_tokenized,
                              seq_length=config.sequence_len,
                              strategy="wrapped",
                              map_kwargs={"batch_size": len(combined_tokenized)})
        
        # Clear the intermediate data
        del combined_tokenized
        del all_input_ids
        del all_attention_masks
    else:
        # Create empty dataset with correct schema
        packed = Dataset.from_dict({"input_ids": [], "attention_mask": []})

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

    # Compute filter dropped tokens
    filter_dropped_tokens = total_tokens_before_filter - total_tokens_after_filter

    inspect_pretrain_dataset(dataset=packed,
                             tokenizer=tokenizer,
                             show_example_rows_limit=1)

    print(
        f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering (samples shorter than min_length={config.dataset_prepared_min_length})"
    )
    print(
        f"\n\033[1;41;97mDropped {last_dropped_chunk_length} tokens\033[0m from final incomplete chunk (length {last_dropped_chunk_length} < sequence_len={config.sequence_len})"
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
