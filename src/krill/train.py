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
)
from pytorch_optimizer import Muon


def do_train(config_path: str):
    """Trains the model using the given YAML config file."""
    print(f"🚀 [Train] Starting training with config: {config_path}")
    # Load config centrally
    config = load_config(config_path)
    # Extract settings from Pydantic model
    dataset_path = config.dataset_prepared_path
    tokenizer_id = config.hub_tokenizer_id
    output_dir = config.output_dir
    model_cfg_name = config.model_config_name
    num_epochs = config.num_epochs
    lr = config.learning_rate
    weight_decay = config.weight_decay
    optimizer_choice = config.optimizer
    hf_model_id = config.hub_model_id
    max_seq = config.sequence_len
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Tokenizer
    logger.info(f"Loading tokenizer {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Model config
    model_configs = {
        "small": LlamaConfig(initializer_range=(1 / math.sqrt(768)), hidden_size=768, num_hidden_layers=27, intermediate_size=1920, tie_word_embeddings=True, num_attention_heads=12, num_key_value_heads=4),
    }
    cfg = model_configs.get(model_cfg_name)
    cfg.torch_dtype = torch.bfloat16
    cfg.vocab_size = len(tokenizer)
    cfg.max_position_embeddings = max_seq
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    # Model
    logger.info(f"Initializing model '{model_cfg_name}'")
    model = LlamaForCausalLM(cfg)
    model.to(torch.bfloat16).to(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    # Optimizer
    if optimizer_choice == "muon":
        optimizer = Muon(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    # Dataset
    logger.info(f"Loading dataset from {dataset_path}")
    ds = load_from_disk(dataset_path)
    ds = ds.train_test_split(test_size=0.001, shuffle=True)
    # Data collator
    data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        weight_decay=0.0,
        bf16=True,
        push_to_hub=True,
        hub_model_id=hf_model_id,
        logging_steps=1,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        save_strategy="steps",
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
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    tokenizer.push_to_hub(hf_model_id)
    # Train
    trainer.train()
    print(f"🚀 [Train] Finished. Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Krill Training Script")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()
    do_train(args.config)


if __name__ == "__main__":
    main()
