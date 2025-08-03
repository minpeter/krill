import math
import torch
from typing import Any, Iterable, Union
from typing_extensions import TypeAlias

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor],
    Iterable[dict[str, Any]],
    Iterable[tuple[str, torch.Tensor]]
]


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Supports batched inputs: last two dimensions are treated as the matrix to orthogonalize.
    """
    assert G.ndim >= 2, "Input must have at least 2 dimensions"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # If more rows than columns, transpose for stability
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize so spectral norm ≤ 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Quintic Newton–Schulz steps
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # Transpose back if we flipped earlier
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    buf: torch.Tensor,
    momentum: float,
    nesterov: bool,
    ns_steps: int
) -> torch.Tensor:
    """
    Apply momentum, orthogonalize the update via Newton–Schulz, then scale.
    """
    # momentum buffer update
    buf.mul_(momentum).add_(grad)
    # Nesterov if requested
    update = grad.add(buf, alpha=momentum) if nesterov else buf

    # collapse conv filters (4D) into 2D if needed
    if update.ndim > 2:
        update = update.view(update.size(0), -1)

    # orthogonalize
    u = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # scale by 0.2 * sqrt(max(rows, cols))
    A, B = grad.size(-2), grad.size(-1)
    scale = 0.2 * math.sqrt(max(A, B))
    return u.mul(scale)


class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton–Schulz.

    Internally runs an SGD-momentum-like update on 2D weight matrices with an
    orthogonalization post‐step. All other parameters fall back to AdamW.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        defaults: dict[str, Any] = dict(
            lr=lr,
            wd=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        # Split into Muon‐eligible 2D parameters vs AdamW params
        muon_params: list[torch.Tensor] = []
        adamw_params: list[torch.Tensor] = []
        for item in params:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, p = item
            elif isinstance(item, torch.Tensor):
                name, p = "", item
            else:
                continue
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        super().__init__(muon_params + adamw_params, defaults)

        for p in muon_params:
            assert p.ndim == 2, f"Expected 2D tensor but got ndim={p.ndim}"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ----- Muon updates for 2D weights -----
            for p in group["params"]:
                if not self.state[p].get("use_muon", False):
                    continue
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                u = muon_update(
                    grad=g,
                    buf=buf,
                    momentum=group["momentum"],
                    nesterov=group["nesterov"],
                    ns_steps=group["ns_steps"],
                )
                # weight decay (AdamW‐style)
                p.data.mul_(1 - group["lr"] * group["wd"])
                # apply update
                p.data.add_(u.reshape(p.shape), alpha=-group["lr"])

            # ----- AdamW fallback for scalar / embedding params -----
            for p in group["params"]:
                if self.state[p].get("use_muon", False):
                    continue
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                wd = group["wd"]
                lr = group["lr"]

                m1 = state["moment1"]
                m2 = state["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)

                g_hat = m1 / (m2.sqrt() + eps)
                bias_corr1 = 1 - beta1**step
                bias_corr2 = 1 - beta2**step
                scale = bias_corr1 / (bias_corr2**0.5)

                p.data.mul_(1 - lr * wd)
                p.data.add_(g_hat, alpha=-lr / scale)

        return loss
