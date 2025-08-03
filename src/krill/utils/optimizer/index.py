import torch

from muon import SingleDeviceMuonWithAuxAdam
from pytorch_optimizer import Muon as PytorchOptimizerMuon
from krill.utils.optimizer.moonlight_muon import Muon as MoonlightMuon


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1, muon_implementation="moonlight"):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":

        # muon_params = [p for p in model.parameters() if p.ndim >= 2]
        # non_muon_params = [p for p in model.parameters() if p.ndim < 2]

        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        non_muon_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        if muon_implementation == "moonlight":
            # https://github.com/MoonshotAI/Moonlight
            optimizer = MoonlightMuon(
                lr=lr,
                wd=wd,
                muon_params=muon_params,
                adamw_params=non_muon_params,
                adamw_betas=(0.9, 0.95),
            )
        elif muon_implementation == "kellerjordan":
            # https://github.com/KellerJordan/Muon
            # https://kellerjordan.github.io/posts/muon/
            param_groups = [
                dict(params=muon_params, use_muon=True,
                     lr=lr, weight_decay=wd),
                dict(params=non_muon_params, use_muon=False,
                     lr=lr, betas=(0.9, 0.95), weight_decay=wd),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        elif muon_implementation == "pytorch_optimizer":
            # https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/muon.py
            optimizer = PytorchOptimizerMuon(
                muon_params, lr=lr, weight_decay=wd,
                adamw_params=non_muon_params, adamw_lr=lr, adamw_wd=wd
            )
        else:
            assert 0, f"Unknown muon implementation: {muon_implementation}. Supported: ['moonlight', 'kellerjordan', 'pytorch_optimizer']"

        return optimizer

    else:
        assert 0, "optimizer not supported"
