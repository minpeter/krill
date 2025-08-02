from torch.nn.functional import nll_loss
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
import torch


def inspect_token_prediction_info():
    print("🔍 Inspect mode enabled (experimental)")
    print("\033[48;2;255;100;100mEntropy (red background)\033[0m: token prediction uncertainty (red=high uncertainty, grey=low).")
    print("\033[48;2;100;100;255mLoss (blue background)\033[0m: token prediction error (blue=high error, grey=low).")


def inspect_token_predictions(tokenizer, model, text):
    """Compute per-token loss, entropy, top-5 predictions and display colored shading."""
    # Tokenize input and move tensors to the model's device
    tokenized = tokenizer(text, return_tensors='pt',
                          return_offsets_mapping=True)
    input_ids = tokenized['input_ids'].to(model.device)
    # Extract tokens using offset mapping
    offsets = tokenized['offset_mapping'][0]
    tokens = [text[s: e] for s, e in offsets]

    with torch.no_grad():
        # Forward pass; input_ids already on correct device
        res = model(input_ids=input_ids, output_attentions=True)
    logits = res['logits']

    losses, entropies, token_info = [], [], []
    for i in range(len(tokens)):
        loss = nll_loss(logits[0, i], input_ids[0, i])
        losses.append(loss.item())
        entropy = -torch.sum(
            torch.nn.functional.softmax(logits[0, i], dim=-1)
            * torch.nn.functional.log_softmax(logits[0, i], dim=-1)
        )
        entropies.append(entropy.item())
        top5 = torch.argsort(logits[0, i])[:5]
        preds = [tokenizer.decode([idx]) for idx in top5]
        token_info.append(
            "\n".join(f"pred {j+1}: {preds[j]}" for j in range(5)))

    min_e, max_e = min(entropies), max(entropies)
    range_e = max_e - min_e if max_e > min_e else 1.0
    min_l, max_l = min(losses), max(losses)
    range_l = max_l - min_l if max_l > min_l else 1.0

    entropy_tokens, loss_tokens = [], []
    for i, tok in enumerate(tokens):
        ne = (entropies[i] - min_e) / range_e
        gbe = int((1 - ne) * 255)
        entropy_tokens.append(f"\033[48;2;255;{gbe};{gbe}m{tok}\033[0m")
        nl = (losses[i] - min_l) / range_l
        rgl = int((1 - nl) * 255)
        loss_tokens.append(f"\033[48;2;{rgl};{rgl};255m{tok}\033[0m")

    print("".join(entropy_tokens))
    print("".join(loss_tokens))
    return {
        'tokens': tokens,
        'losses': losses,
        'entropies': entropies,
        'token_info': token_info,
        'entropy_tokens': entropy_tokens,
        'loss_tokens': loss_tokens,
    }


if __name__ == "__main__":
    context_length = 128
    tokenizer = AutoTokenizer.from_pretrained(
        "huggingface-course/code-search-net-tokenizer"
    )
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    sample_text = (
        "Contrary to popular belief, Lorem Ipsum is not simply random text."
    )
    inspect_token_predictions(tokenizer, model, sample_text)
