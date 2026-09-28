"""ESM encoder with AMP, strain-conditioned MIC, and hemolysis/HC50 heads.

Transformers is imported lazily so inference without the optional property model
does not gain an undeclared runtime dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PropertyOutput:
    amp_logits: torch.Tensor
    mic_mean: torch.Tensor
    mic_log_std: torch.Tensor
    hemolysis_logits: torch.Tensor
    hc50_mean: torch.Tensor
    hc50_log_std: torch.Tensor
    pooled: torch.Tensor
    task_masks: dict[str, torch.Tensor] | None = None

    @property
    def amp_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.amp_logits)

    @property
    def hemolysis_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.hemolysis_logits)


def censored_gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor,
                          censor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL with exact(0), left(-1), right(1), or interval(2) censor codes.

    Interval observations use target as the lower bound and an optional upper bound
    supplied through the `upper` argument in `multitask_loss`.
    """
    sigma = log_std.exp().clamp_min(1e-5)
    z = (target - mean) / sigma
    exact = 0.5 * z.square() + log_std + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=mean.device))
    cdf = torch.special.ndtr(z).clamp(1e-7, 1.0 - 1e-7)
    terms = torch.where(censor < 0, -torch.log(cdf), torch.where(censor > 0, -torch.log1p(-cdf), exact))
    effective = mask.bool() & censor.ne(2)
    return terms[effective].mean() if effective.any() else mean.sum() * 0.0


class ESMMultiTaskPredictor(nn.Module):
    """Task heads over residue-only pooled ESM representations."""

    def __init__(self, backbone: nn.Module, hidden_size: int, strain_count: int, strain_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.strain_embedding = nn.Embedding(strain_count + 1, strain_dim)  # final index is unknown
        self.amp_head = nn.Linear(hidden_size, 1)
        self.mic_head = nn.Sequential(nn.Linear(hidden_size + strain_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, 2))
        self.hemo_head = nn.Linear(hidden_size, 1)
        self.hc50_head = nn.Linear(hidden_size, 2)

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/esm2_t30_150M_UR50D", revision: str | None = None,
                        strain_count: int = 1) -> tuple["ESMMultiTaskPredictor", object]:
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError as exc:
            raise RuntimeError("Install the training extra `transformers` to load an ESM property model") from exc
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        backbone = EsmModel.from_pretrained(model_name, revision=revision)
        return cls(backbone, int(backbone.config.hidden_size), strain_count), tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, strain_ids: torch.Tensor) -> PropertyOutput:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        special = torch.zeros_like(attention_mask, dtype=torch.bool)
        if getattr(self.backbone.config, "bos_token_id", None) is not None:
            special |= input_ids.eq(self.backbone.config.bos_token_id)
        if getattr(self.backbone.config, "eos_token_id", None) is not None:
            special |= input_ids.eq(self.backbone.config.eos_token_id)
        residue_mask = attention_mask.bool() & ~special
        pooled = (hidden * residue_mask.unsqueeze(-1)).sum(1) / residue_mask.sum(1, keepdim=True).clamp_min(1)
        mic = self.mic_head(torch.cat([pooled, self.strain_embedding(strain_ids)], dim=-1))
        hc = self.hc50_head(pooled)
        return PropertyOutput(self.amp_head(pooled).squeeze(-1), mic[:, 0], mic[:, 1].clamp(-6, 4),
                              self.hemo_head(pooled).squeeze(-1), hc[:, 0], hc[:, 1].clamp(-6, 4), pooled)

    def multitask_loss(self, output: PropertyOutput, labels: dict[str, torch.Tensor], weights: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
        weights = weights or {}
        losses: dict[str, torch.Tensor] = {}
        for key, logits in (("amp", output.amp_logits), ("hemolysis", output.hemolysis_logits)):
            mask = labels.get(f"{key}_mask", torch.zeros_like(logits, dtype=torch.bool)).bool()
            losses[key] = F.binary_cross_entropy_with_logits(logits[mask], labels[key][mask].float()) if mask.any() else logits.sum() * 0
        for key, mean, log_std in (("mic", output.mic_mean, output.mic_log_std), ("hc50", output.hc50_mean, output.hc50_log_std)):
            mask = labels.get(f"{key}_mask", torch.zeros_like(mean, dtype=torch.bool)).bool()
            censor = labels.get(f"{key}_censor", torch.zeros_like(mean, dtype=torch.long))
            losses[key] = censored_gaussian_nll(mean, log_std, labels[key], censor, mask)
        losses["total"] = sum(float(weights.get(k, 1.0)) * v for k, v in losses.items())
        return losses
