import re
from typing import Any
from datasets import Dataset, DatasetDict


def inspect_pretrain_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer: Any,
    show_example_rows_limit: int = 3
):
    for i in range(min(show_example_rows_limit, len(dataset))):
        sample = dataset[i]

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
