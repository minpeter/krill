import argparse
import os
import math
import logging
import torch

from datasets import load_from_disk
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithFlattening,
    DataCollatorForLanguageModeling
)

from krill.utils.optimizer import get_optimizer
from krill.utils.config import load_config
from krill.utils.resume import determine_resume_checkpoint
from krill import HAS_FLASH_ATTENTION, SUPPORTS_BFLOAT16


def build_llama_config(preset: str) -> LlamaConfig:
    llama_presets = {
        "pico": LlamaConfig(
            initializer_range=0.02,
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=64,
            tie_word_embeddings=True,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        "micro": LlamaConfig(
            initializer_range=(1 / math.sqrt(256)),
            hidden_size=256,
            num_hidden_layers=12,
            intermediate_size=1024,
            tie_word_embeddings=True,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        "small": LlamaConfig(
            initializer_range=(1 / math.sqrt(768)),
            hidden_size=768,
            num_hidden_layers=27,
            intermediate_size=1920,
            tie_word_embeddings=True,
            num_attention_heads=12,
            num_key_value_heads=4,
        ),
    }
    return llama_presets[preset]


def build_qwen3_config(preset: str):
    qwen_presets = {
        "pico": Qwen3Config(
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=True,
        ),
        "micro": Qwen3Config(
            hidden_size=256,
            num_hidden_layers=12,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        ),
        "small": Qwen3Config(
            hidden_size=768,
            num_hidden_layers=27,
            intermediate_size=1920,
            num_attention_heads=12,
            num_key_value_heads=4,
            tie_word_embeddings=True,
        ),
    }
    return qwen_presets[preset]


def do_train(config_path: str):
    """Trains the model using the given YAML config file."""
    print(f"🦐 Krill: Starting training with config: {config_path}")

    # Load config centrally
    config = load_config(config_path)
    # Extract settings from Pydantic model
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Tokenizer
    logger.info(f"Loading tokenizer {config.tokenizer.hub_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.hub_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model config presets per arch
    arch = getattr(config.train, "arch", "llama").lower()

    preset_name = config.train.model_config_name
    if arch == "llama":
        cfg = build_llama_config(preset_name)
    elif arch == "qwen3":
        cfg = build_qwen3_config(preset_name)
    else:
        raise ValueError(f"Unsupported arch: {arch}. Supported: llama, qwen3")

    # Common config tweaks
    cfg.torch_dtype = torch.bfloat16
    cfg.vocab_size = len(tokenizer)
    cfg.max_position_embeddings = config.preprocess.sequence_len
    cfg.use_cache = False

    # Tokens
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.bos_token_id = tokenizer.eos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    # Attention implementation
    if HAS_FLASH_ATTENTION:
        # Set attention implementation to flash attention
        setattr(cfg, "_attn_implementation", "flash_attention_2")

    # RoPE theta
    if getattr(cfg, "max_position_embeddings", 0) >= 8192:
        setattr(cfg, "rope_theta", 1_000_000.0)
    else:
        setattr(cfg, "rope_theta", 10_000.0)

    # Model
    logger.info(f"Initializing model '{preset_name}' for arch '{arch}'")
    if arch == "llama":
        model = LlamaForCausalLM(cfg)
    else:
        model = Qwen3ForCausalLM(cfg)
    model.to(torch.bfloat16).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Optimizer
    optimizer = get_optimizer(
        config.train.optimizer,
        model,
        lr=config.train.learning_rate,
        wd=config.train.weight_decay,
        muon_implementation=config.train.muon_implementation)

    # Dataset
    logger.info(f"Loading dataset from {config.preprocess.prepared_path}")
    ds = load_from_disk(config.preprocess.prepared_path)
    ds = ds.train_test_split(test_size=0.001, shuffle=True)

    # Data collator
    if HAS_FLASH_ATTENTION:
        data_collator = DataCollatorWithFlattening(
            return_flash_attn_kwargs=True,
            return_position_ids=True,
            return_seq_idx=True)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                        mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=config.train.output_dir,
        do_train=True,
        do_eval=True,
        logging_dir=f"{config.train.output_dir}/logs",
        overwrite_output_dir=True,
        push_to_hub=True,
        hub_model_id=config.train.hub_model_id,
        hub_strategy="checkpoint",
        eval_strategy="steps",
        save_strategy="steps",
        # eval_steps=1_000,
        # save_steps=1_000,
        eval_steps=100,
        save_steps=100,
        logging_steps=1,

        # auto_find_batch_size=True,
        per_device_train_batch_size=config.train.micro_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        num_train_epochs=config.train.num_epochs,
        warmup_ratio=0.05,
        # Muon handles weight decay internally, so set to 0 in Trainer
        weight_decay=0.0,
        lr_scheduler_type="cosine",  # warmup_stable_decay
        learning_rate=config.train.learning_rate,
        bf16=SUPPORTS_BFLOAT16,
        ddp_find_unused_parameters=True,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,

        # https://huggingface.co/docs/transformers/v4.53.3/en/trainer#optimizations
        use_liger_kernel=torch.cuda.is_available(),
        # neftune_noise_alpha=0.1,

        # torch_compile=True,
        # # "default", "max-autotune", "reduce-overhead"
        # torch_compile_mode="reduce-overhead",
        save_total_limit=3,

        # metric_for_best_model="eval_loss",
        # load_best_model_at_end=True,
        # greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    # Save tokenizer
    os.makedirs(config.train.output_dir, exist_ok=True)
    tokenizer.save_pretrained(config.train.output_dir)
    tokenizer.push_to_hub(config.train.hub_model_id)

    # Handle resume functionality
    resume_from_checkpoint = determine_resume_checkpoint(
        config.train.resume,
        config.train.output_dir,
        config.train.hub_model_id
    )

    # Train
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
        
    print(f"🚀 [Train] Finished. Model saved to {config.train.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a model with Krill.")
    parser.add_argument("config",
                        type=str,
                        help="Path to the YAML config file.")
    args = parser.parse_args()

    do_train(args.config)


if __name__ == "__main__":
    main()
