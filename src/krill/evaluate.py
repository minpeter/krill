import lm_eval
from lm_eval.loggers import WandbLogger
import wandb


def do_evaluate(model: str):
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model},trust_remote_code=True",
        tasks=[
            "hellaswag",
            "piqa"
        ],
        log_samples=True,
        batch_size="auto"
    )

    print(results)

    wandb.init(
        project="krill-eval",
        job_type="eval"
    )
    wandb_logger = WandbLogger()
    wandb_logger.post_init(results)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(results["samples"])  # if log_samples
