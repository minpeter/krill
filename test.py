from torch.nn.functional import nll_loss
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
import torch

context_length = 128
tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer")

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

text = 'Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32. The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et Malorum" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.'
tokenized = tokenizer(
    text,
    return_tensors='pt',
    return_offsets_mapping=True
)
input_ids = tokenized['input_ids']

tokens = [text[s: e] for s, e in tokenized['offset_mapping'][0]]

with torch.no_grad():
    res = model(input_ids=input_ids.to(model.device), output_attentions=True)


logits = res['logits']
losses = []
entropies = []
token_info = []
for i in range(len(tokens)):
    loss = nll_loss(logits[0, i], input_ids[0, i])
    losses.append(loss.item())

    # compute entropy with explicit dimension for softmax and log_softmax
    entropy = -torch.sum(
        torch.nn.functional.softmax(logits[0, i], dim=-1)
        * torch.nn.functional.log_softmax(logits[0, i], dim=-1)
    )
    entropies.append(entropy.item())

    pred_token_indices = torch.argsort(logits[0, i])[:5]
    pred_tokens = [tokenizer.decode([idx]) for idx in pred_token_indices]
    token_info.append(
        f"pred 1: {pred_tokens[0]}\npred 2: {pred_tokens[1]}\npred 3: {pred_tokens[2]}\npred 4: {pred_tokens[3]}\npred 5: {pred_tokens[4]}")

# Build separate shading: entropy (red hue) and loss (blue hue)
# normalize entropy
min_e, max_e = min(entropies), max(entropies)
range_e = max_e - min_e if max_e > min_e else 1.0
# normalize loss
min_l, max_l = min(losses), max(losses)
range_l = max_l - min_l if max_l > min_l else 1.0

entropy_tokens = []
loss_tokens = []
for i, tok in enumerate(tokens):
    # entropy shading: red base
    ne = (entropies[i] - min_e) / range_e
    gbe = int((1 - ne) * 255)
    entropy_tokens.append(f"\033[48;2;255;{gbe};{gbe}m{tok}\033[0m")
    # loss shading: blue base
    nl = (losses[i] - min_l) / range_l
    rgl = int((1 - nl) * 255)
    loss_tokens.append(f"\033[48;2;{rgl};{rgl};255m{tok}\033[0m")

print("Entropy: 토큰 예측 분포의 불확실성을 나타냅니다. 빨간색일수록 불확실성이 높아 다양한 후보가 존재하며, 회색일수록 불확실성이 낮아 모델이 자신 있는 예측을 함을 의미합니다.")
print("Loss: 모델 예측의 오류 크기(loss)를 나타냅니다. 파란색일수록 loss가 높아 예측이 어려웠음을, 회색일수록 loss가 낮아 예측이 정확했음을 의미합니다.")

print("Entropy shading:")
print("".join(entropy_tokens))
print("Loss shading:")
print("".join(loss_tokens))
