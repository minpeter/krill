# uv run 01-preprocess.py --context_length 8192 --tokenizer_id minpeter/tiny-ko-tokenizer-32k-250725 --save_path ./artifacts/prepacked-8k-new

import argparse
import os
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import pack_dataset
import multiprocessing
import resource
import re


parser = argparse.ArgumentParser(description="Preprocess datasets for tiny-ko")
parser.add_argument("--context_length", type=int, default=8192,
                    help="Context length for grouping texts")
parser.add_argument("--tokenizer_id", type=str,
                    default="minpeter/tiny-ko-tokenizer-32k-250725", help="Path to tokenizer")
parser.add_argument("--save_path", type=str,
                    default="./artifacts/prepacked", help="Path to save processed data")
parser.add_argument("--dataset_id", type=str, default="minpeter/tiny-corpus",
                    help="Dataset ID to load from Hugging Face Hub")
parser.add_argument("--min_length", type=int, default=150,
                    help="Minimum length of samples to keep after tokenization")

args = parser.parse_args()


def setup_directories():
    os.makedirs(args.save_path, exist_ok=True)


def load_raw_datasets():
    print("Loading raw datasets...")
    return load_dataset(args.dataset_id, split="train")


def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding=False, truncation=False)

    if tokenizer.eos_token_id is not None:
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"][i].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"][i].append(0)

    return tokenized_inputs


if __name__ == "__main__":
    total_start_time = time.time()
    setup_directories()

    raw_datasets = load_raw_datasets()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    num_processors = max(1, os.cpu_count() - 8)
    # # During working hours, do not occupy all cluster resources.
    # num_processors = 32

    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_processors} processes for mapping.")

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_processors,
        remove_columns=raw_datasets.column_names,
        desc="Tokenizing"
    )

    # 1) Compute and report tokens dropped by filtering out short samples
    # 1) Compute token lengths in parallel
    with multiprocessing.Pool(processes=num_processors) as pool:
        lengths = pool.map(len, tokenized_datasets["input_ids"])
    total_tokens_before_filter = sum(lengths)

    # 2) Select samples meeting min_length
    selected_indices = [idx for idx, length in enumerate(
        lengths) if length >= args.min_length]
    filtered_datasets = tokenized_datasets.select(selected_indices)
    total_tokens_after_filter = sum(lengths[idx] for idx in selected_indices)
    filter_dropped_tokens = total_tokens_before_filter - total_tokens_after_filter
    # PRINT: The filter_dropped_tokens variable is used for statistics output below.
    tokenized_datasets = filtered_datasets

    # 2) Pack into fixed‐length sequences
    packed_dataset = pack_dataset(
        tokenized_datasets,
        seq_length=args.context_length,
        strategy="wrapped",
        # The default batch_size for the "wrapped" strategy (e.g., 1000) creates remainders in each internal batch.
        # We set the batch_size to the total dataset size (len(tokenized_datasets)) so that it processes the data in one go.
        # If this setting causes issues in the future, remove it and modify the code to drop any data that exceeds the context_length after packing.
        map_kwargs={"batch_size": len(tokenized_datasets)}
    )

    # 3) Drop last incomplete chunk and record its dropped tokens
    pack_dropped_tokens = 0
    if len(packed_dataset) > 0:
        last_len = len(packed_dataset[-1]["input_ids"])
        if last_len < args.context_length:
            pack_dropped_tokens = last_len
            # PRINT: dropped pack tokens is moved to the statistics section below
            packed_dataset = packed_dataset.select(
                list(range(len(packed_dataset) - 1)))

    # >>>>>> Preview of the tokenized and packed results >>>>>>
    SHOW_EXAMPLE_ROWS_LIMIT = 3

    for i in range(min(SHOW_EXAMPLE_ROWS_LIMIT, len(packed_dataset))):
        sample = packed_dataset[i]

        colored_items = []
        # Display tokens, merging runs of undecodable (replacement char) tokens
        items = sample["input_ids"]
        idx = 0
        # \w already matches Unicode word characters (letters, digits, underscores)
        normal_pattern = re.compile(r"\w", flags=re.UNICODE)
        while idx < len(items):
            token_id = items[idx]
            token_str = tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False)
            # special tokens defined in tokenizer
            if token_id in tokenizer.all_special_ids:
                # yellow background, black text for special tokens
                colored_items.append(
                    f'\033[1;43;30m{token_str}\033[0m({token_id})')
                idx += 1
                continue
            # detect mergeable runs (non-word chars, excluding special tokens)
            if token_id not in tokenizer.all_special_ids and not normal_pattern.match(token_str):
                # gather run of mergeable tokens (skip special tokens)
                start = idx
                while idx < len(items):
                    next_id = items[idx]
                    next_str = tokenizer.decode(
                        [next_id], clean_up_tokenization_spaces=False)
                    if next_id not in tokenizer.all_special_ids and not normal_pattern.match(next_str):
                        idx += 1
                        continue
                    break
                run_ids = items[start:idx]
                run_str = tokenizer.decode(
                    run_ids, clean_up_tokenization_spaces=False)
                ids_str = ",".join(str(x) for x in run_ids)
                # magenta background for special runs
                colored_items.append(f'\033[45;97m{run_str}\033[0m({ids_str})')
                continue
            # normal token
            colored_items.append(f'\033[44;97m{token_str}\033[0m({token_id})')
            idx += 1
        print(
            f"\n\033[1;43;30m[PACKED SAMPLE {i}]\033[0m {' '.join(colored_items)}")
    # <<<<<< End of preview of the tokenized and packed results <<<<<<

    packed_dataset.save_to_disk(args.save_path)

    # Dataset statistics
    original_rows = len(raw_datasets)
    packed_rows = len(packed_dataset)

    if filter_dropped_tokens > 0:
        print(
            f"\n\033[1;41;97mDropped {filter_dropped_tokens} tokens\033[0m during filtering "
            f"(samples shorter than min_length={args.min_length})"
        )
    # Insert pack_dropped_tokens print into statistics section
    if pack_dropped_tokens > 0:
        print(
            f"\n\033[1;41;97mDropped {pack_dropped_tokens} tokens\033[0m from final incomplete chunk "
            f"(length {pack_dropped_tokens} < context_length={args.context_length})"
        )

    print(f"\nOriginal dataset rows: {original_rows}")
    print(f"Packed dataset rows: {packed_rows}")

    if packed_rows > 0:
        # Check for any samples that do not match the expected context length
        wrong_indices = [
            i for i, sample in enumerate(packed_dataset)
            if len(sample["input_ids"]) != args.context_length
        ]
        if wrong_indices:
            print(
                f"\033[1;41;97mWarning: Found {len(wrong_indices)} samples "
                f"with incorrect length (expected {args.context_length}). "
                f"Indices: {wrong_indices}\033[0m"
            )
        else:
            print("All packed samples have the correct context length.")

    print(
        f"\nTotal elapsed time: {(time.time() - total_start_time)/60:.2f} minutes")
    # report peak memory usage
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak memory usage: {mem_kb/1024:.2f} MB")
