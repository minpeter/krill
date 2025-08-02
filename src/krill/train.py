import argparse
from .config import load_config
import os
import math
import logging
import torch
from datasets import load_from_disk
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithFlattening,
    DataCollatorForLanguageModeling
)
from pytorch_optimizer import Muon


def do_train(config_path: str):
    """Trains the model using the given YAML config file."""
    print(f"🚀 [Train] Starting training with config: {config_path}")
    # Load config centrally
    config = load_config(config_path)
    # Extract settings from Pydantic model
    # ...access settings directly from config...
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Tokenizer
    logger.info(f"Loading tokenizer {config.hub_tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.hub_tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Model config
    model_configs = {
        "micro": LlamaConfig(initializer_range=(1 / math.sqrt(256)), hidden_size=256, num_hidden_layers=12, intermediate_size=1024, tie_word_embeddings=True, num_attention_heads=4, num_key_value_heads=2),
        "small": LlamaConfig(initializer_range=(1 / math.sqrt(768)), hidden_size=768, num_hidden_layers=27, intermediate_size=1920, tie_word_embeddings=True, num_attention_heads=12, num_key_value_heads=4),
    }

    cfg = model_configs.get(config.model_config_name)
    cfg.torch_dtype = torch.bfloat16
    cfg.vocab_size = len(tokenizer)
    cfg.max_position_embeddings = config.sequence_len
    cfg.use_cache = False

    cfg.pad_token_id = tokenizer.pad_token_id
    # Following Qwen style: only the BOS is set in the model config; not actually used
    cfg.bos_token_id = tokenizer.eos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    cfg._attn_implementation = "flash_attention_2"

    # Set rope_theta
    if cfg.max_position_embeddings >= 8192:
        cfg.rope_theta = 1_000_000.0  # or optionally 500_000.0
    else:
        cfg.rope_theta = 10_000.0  # default

    # Model
    logger.info(f"Initializing model '{config.model_config_name}'")
    model = LlamaForCausalLM(cfg)
    model.to(torch.bfloat16).to(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    # Optimizer
    if config.optimizer == "muon":
        optimizer = Muon(model.parameters(), lr=config.learning_rate,
                         weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Dataset
    logger.info(f"Loading dataset from {config.dataset_prepared_path}")
    ds = load_from_disk(config.dataset_prepared_path)
    ds = ds.train_test_split(test_size=0.001, shuffle=True)

    # Data collator
    data_collator = DataCollatorWithFlattening(
        return_flash_attn_kwargs=True, return_position_ids=True, return_seq_idx=True)
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        do_train=True,
        do_eval=True,
        logging_dir=f"{config.output_dir}/logs",
        overwrite_output_dir=True,

        push_to_hub=True,
        hub_model_id=config.hub_model_id,
        hub_strategy="checkpoint",

        eval_strategy="steps",
        save_strategy="steps",
        # eval_steps=1_000,
        # save_steps=1_000,
        eval_steps=100,
        save_steps=100,
        logging_steps=1,


        # https://huggingface.co/docs/transformers/v4.53.3/en/trainer#optimizations

        auto_find_batch_size=True,
        # per_device_train_batch_size=4,
        gradient_accumulation_steps=2,


        num_train_epochs=config.num_epochs,

        warmup_ratio=0.05,
        # --- Muon이 weight decay를 자체 처리하므로 Trainer에서는 0으로 설정 ---
        weight_decay=0.0,
        lr_scheduler_type="cosine",  # warmup_stable_decay
        learning_rate=config.learning_rate,
        bf16=True,


        # torch_compile=True,
        # # "default", "max-autotune", "reduce-overhead"
        # torch_compile_mode="reduce-overhead",

        ddp_find_unused_parameters=True,

        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,

        remove_unused_columns=False,

        use_liger_kernel=True,
        # neftune_noise_alpha= 0.1,

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
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer.save_pretrained(config.output_dir)
    tokenizer.push_to_hub(config.hub_model_id)
    # Train
    trainer.train(
        # resume_from_checkpoint=True
        # resume_from_checkpoint="last-checkpoint" # resume from the huggingface_hub last checkpoint
    )
    print(f"🚀 [Train] Finished. Model saved to {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Krill Training Script")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()
    do_train(args.config)


if __name__ == "__main__":
    main()
