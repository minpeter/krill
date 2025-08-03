import math
import os
import torch
from typing import Any, Iterable, Union, Optional, Tuple
from typing_extensions import TypeAlias

# Import from pytorch_optimizer
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.exception import NoSparseGradientError, NoComplexParameterError

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]
                                     ], Iterable[tuple[str, torch.Tensor]]
]

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py


def zero_power_via_newton_schulz_5(
    g: torch.Tensor, num_steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This is based on the pytorch_optimizer implementation which uses bfloat16 tensors
    and is optimized for distributed training.
    """
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')

    # Coefficients selected to maximize slope at zero
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    x = g.bfloat16()
    x.div_(x.norm().add_(eps))

    if g.size(0) > g.size(1):
        x = x.T

    for _ in range(num_steps):
        mat_a = x @ x.T
        mat_b = b * mat_a + c * mat_a @ mat_a
        x = a * x + mat_b @ x

    if g.size(0) > g.size(1):
        x = x.T

    return x


class Muon(BaseOptimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    This is a fully featured, distributed-aware implementation based on BaseOptimizer
    from pytorch_optimizer. It automatically splits parameters into Muon (≥2D) and 
    AdamW sets, supports multi-GPU training, and includes advanced features like
    decoupled weight decay, Nesterov momentum, bias-correction, and adjusted learning rates.

    Arguments:
      params: Iterable of parameters to optimize or dicts defining parameter groups.
      lr: Base learning rate for both Muon and AdamW updates (default: 0.02).
      momentum: Momentum factor for the Muon (SGD) update (default: 0.95).
      weight_decay: L2 regularization coefficient applied to all parameters (default: 0.01).
      weight_decouple: Use decoupled weight decay as in AdamW (default: True).
      betas: Tuple (beta1, beta2) for the internal AdamW optimizer (default: (0.9, 0.95)).
      nesterov: If True, use Nesterov momentum in the Muon update (default: True).
      ns_steps: Number of Newton–Schulz iterations for orthogonalizing each Muon update (default: 5).
      use_adjusted_lr: Use adjusted learning rate scaling for Muon parameters (default: False).
      adamw_params: Separate parameters for AdamW (default: None, auto-detected).
      adamw_lr: Separate learning rate for AdamW parameters (default: 0.0003).
      adamw_wd: Weight decay for AdamW parameters (default: 0.0).
      adamw_eps: Epsilon value for numerical stability in AdamW (default: 1e-8).
      maximize: Maximize the objective with respect to the params (default: False).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        weight_decouple: bool = True,
        betas: Tuple[float, float] = (0.9, 0.95),
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_params: Optional[ParamsT] = None,
        adamw_lr: float = 0.0003,
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-8,
        maximize: bool = False,
        **kwargs
    ) -> None:
        
        # Distributed training setup
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        defaults: dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            adamw_betas=betas,
            nesterov=nesterov,
            ns_steps=ns_steps,
            use_adjusted_lr=use_adjusted_lr,
            adamw_lr_ratio=adamw_lr / lr if lr > 0 else 0.015,  # ratio for scaling
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
            maximize=maximize,
        )

        # Get parameters and split into Muon vs AdamW
        muon_params, adamw_params_auto = self.get_parameters(params)
        
        # Use provided adamw_params if given, otherwise use auto-detected
        if adamw_params is not None:
            _, adamw_params_provided = self.get_parameters(adamw_params)
            all_params = muon_params + adamw_params_provided
        else:
            all_params = muon_params + adamw_params_auto
            adamw_params = adamw_params_auto

        super().__init__(all_params, defaults)
        
        # Set Muon state for parameter classification
        self.set_muon_state(muon_params, adamw_params)

    def get_parameters(self, params: ParamsT) -> Tuple[list, list]:
        """Split parameters into Muon (≥2D) and AdamW sets."""
        muon_params: list[torch.Tensor] = []
        adamw_params: list[torch.Tensor] = []
        
        for item in params:
            # Unpack named parameters if provided
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], torch.Tensor):
                name, p = item
            # Handle plain parameter tensors
            elif isinstance(item, torch.Tensor):
                name, p = "", item
            else:
                # Skip unsupported entries
                continue
                
            if not p.requires_grad:
                continue
                
            # Determine if using Muon (2D+ and not embedding or LM head)
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
                
        return muon_params, adamw_params

    def set_muon_state(self, muon_params: list, adamw_params: list) -> None:
        """Set use_muon flag for parameter classification."""
        for p in muon_params:
            self.state[p]['use_muon'] = True
        for p in adamw_params:
            self.state[p]['use_muon'] = False

    def init_group(self, group: dict, **kwargs) -> None:
        """Initialize group state."""
        pass

    @staticmethod
    def get_adjusted_lr(lr: float, param_shape: Tuple[int, ...], use_adjusted_lr: bool = False) -> float:
        """Get the adjusted learning rate based on parameter shape."""
        output_shape, *input_shape = param_shape
        input_shape = math.prod(input_shape)

        ratio: float = (
            math.pow(max(1.0, output_shape / input_shape), 0.5)
            if use_adjusted_lr
            else 0.2 * math.sqrt(max(output_shape, input_shape))
        )

        return lr * ratio

    @staticmethod
    def debias(beta: float, step: int) -> float:
        """Compute bias correction term."""
        return 1.0 - beta ** step

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            ############################
            #           Muon           #
            ############################

            params = []
            for p in group['params']:
                if p.grad is not None and self.state[p]['use_muon']:
                    if p.grad.is_sparse:
                        raise NoSparseGradientError(str(self))
                    if torch.is_complex(p):
                        raise NoComplexParameterError(str(self))
                    params.append(p)

            if len(params) == 0:
                continue

            momentum = group['momentum']

            # Distributed computation: prepare updates for all parameters
            total_params: int = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr_idx: int = 0

            # Process parameters distributed across ranks
            for i, p in enumerate(params):
                if i % self.world_size != self.rank:
                    curr_idx += p.numel()
                    continue

                grad = p.grad
                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.lerp_(grad, weight=1.0 - momentum)

                if group['nesterov']:
                    grad = grad.lerp_(buf, momentum)
                else:
                    grad = buf

                # Apply Newton-Schulz orthogonalization
                grad = zero_power_via_newton_schulz_5(grad, num_steps=group['ns_steps']).flatten()

                updates_flat[curr_idx:curr_idx + p.numel()] = grad

                curr_idx += p.numel()

            # Distributed all-reduce if multi-GPU
            if self.world_size > 1:
                torch.distributed.all_reduce(updates_flat, op=torch.distributed.ReduceOp.SUM)

            # Apply updates
            curr_idx: int = 0
            for p in params:
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p)

                # Apply weight decay (decoupled)
                self.apply_weight_decay(
                    p,
                    grad=g,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                # Get adjusted learning rate
                lr: float = self.get_adjusted_lr(group['lr'], p.size(), group['use_adjusted_lr'])

                # Apply maximize flag
                alpha = lr if not group['maximize'] else -lr
                p.add_(g, alpha=-alpha)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if p.grad is not None and not self.state[p]['use_muon']]

            lr: float = group['adamw_lr_ratio'] * group['lr']
            beta1, beta2 = group['adamw_betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            scale: float = bias_correction1 / bias_correction2 ** 0.5
            step_size: float = lr / scale

            for p in params:
                grad = p.grad
                state = self.state[p]
                if 'moment1' not in state:
                    state['moment1'] = torch.zeros_like(grad)
                    state['moment2'] = torch.zeros_like(grad)

                buf1, buf2 = state['moment1'], state['moment2']
                buf1.lerp_(grad, weight=1.0 - beta1)
                buf2.lerp_(grad.square(), weight=1.0 - beta2)

                update = buf1 / buf2.sqrt().add_(group['adamw_eps'])

                # Apply weight decay for AdamW parameters
                self.apply_weight_decay(
                    p,
                    grad,
                    lr=lr,
                    weight_decay=group['adamw_wd'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                # Apply maximize flag
                alpha = step_size if not group['maximize'] else -step_size
                p.add_(update, alpha=-alpha)

        return loss
