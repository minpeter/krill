"""
Module to train tokenizer based on Krill config.
"""
import os
from krill.utils.dataset_utils import load_and_prepare_raw_datasets
from tokenizers import (
    Tokenizer,
    AddedToken,
    Regex,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from krill.config import load_config


def train_and_save_huggingface_tokenizer(
    dataset: Dataset,
    output_dir: str,
    target_vocab_size: int,
    huggingface_hub_id: str,
):
    # Define additional tokens
    additional_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        AddedToken("<tool_call>", special=False, normalized=False),
        AddedToken("</tool_call>", special=False, normalized=False),
        AddedToken("<think>", special=False, normalized=False),
        AddedToken("</think>", special=False, normalized=False),
        AddedToken("<|unused_special_token_0|>",
                   special=True, normalized=False),
        AddedToken("<|unused_special_token_1|>",
                   special=True, normalized=False),
        AddedToken("<|unused_special_token_2|>",
                   special=True, normalized=False),
        AddedToken("<|unused_special_token_3|>",
                   special=True, normalized=False),
    ]

    vocab_size = target_vocab_size - len(additional_tokens)

    def get_training_corpus():
        for i in range(len(dataset)):
            text = dataset[i].get("text")
            yield text or ""

    tokenizer = Tokenizer(
        models.BPE(
            byte_fallback=True,
        )
    )
    tokenizer.normalizer = normalizers.NFKC()
    tiktoken_pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(tiktoken_pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    byte_level_alphabet = pre_tokenizers.ByteLevel.alphabet()
    chatml_reserved_words = ["system", "user", "assistant", "tool"]
    initial_alphabet = sorted(
        list(set(byte_level_alphabet + chatml_reserved_words)))

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=initial_alphabet,
        min_frequency=2,
        max_token_length=30,
    )

    print("⏳ Training started...")
    tokenizer.train_from_iterator(
        get_training_corpus(), trainer=trainer, length=len(dataset)
    )
    print("✅ Training completed!")

    print("\n✅ Adding extra tokens to the trained tokenizer...")
    tokenizer.add_tokens(additional_tokens)
    print(f"✅ {len(additional_tokens)} tokens have been added.")

    print("\n✅ Saving tokenizer in AutoTokenizer compatible format...")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        bos_token=None,
        add_bos_token=False,
        add_prefix_space=False,
        split_special_tokens=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)
    print(f"✅ Tokenizer saved to {output_dir}")
    fast_tokenizer.push_to_hub(huggingface_hub_id, private=False)
    print(f"✅ Tokenizer pushed to Hugging Face Hub: {huggingface_hub_id}")

    print(f"✅ Tokenizer and config files have been saved to '{output_dir}'")
    print("Generated files:", os.listdir(output_dir))

    print("\n--- Final validation ---")
    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print("✅ AutoTokenizer loaded successfully!")

        text_to_test = "This is a <tool_call> test."
        encoded_output = loaded_tokenizer(text_to_test)
        decoded_text = loaded_tokenizer.decode(encoded_output["input_ids"])

        print(f"\nTest sentence: {text_to_test}")
        print(f"Encoded IDs: {encoded_output['input_ids']}")
        print(f"Decoded sentence: {decoded_text}")

        if "<tool_call>" in decoded_text:
            print("✅ Success: '<tool_call>' token is preserved after decoding.")
        else:
            print("❌ Failure: '<tool_call>' token disappeared after decoding.")

    except Exception as e:
        print(f"❌ Failed to load AutoTokenizer: {e}")

    vocab_size = len(loaded_tokenizer)
    print("Tokenizer total vocab size:", vocab_size)
    if vocab_size % 64 != 0:
        print(
            f"\n⚠️ Warning: Tokenizer vocab size ({vocab_size}) is not divisible by 64.\n"
            "This may lead to slight performance degradation during model quantization or training."
        )


def main_train_tokenizer(config_path: str):
    """Main entry point for training tokenizer via CLI."""
    cfg = load_config(config_path)
    # Load and prepare dataset for tokenizer training
    dataset = load_and_prepare_raw_datasets(cfg.datasets)
    # Determine output directory for tokenizer
    output_dir = f"./artifacts/tknz/{cfg.hub_tokenizer_id}"
    train_and_save_huggingface_tokenizer(
        dataset=dataset,
        output_dir=output_dir,
        target_vocab_size=cfg.vocab_size,
        huggingface_hub_id=cfg.hub_tokenizer_id,
    )
