"""Core model components: denoiser, schedule, sampling."""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder

CONDITION_NAMES = (
    "amp_probability",
    "broad_spectrum_probability",
    "mean_activity_probability",
    "mean_gram_negative_activity_probability",
    "mean_gram_positive_activity_probability",
    "mean_mdr_activity_probability",
    "hemolysis_probability",
    "predicted_log2_mic_summary",
    "predicted_log2_hc50",
)


@dataclass(frozen=True)
class ConditionVector:
    """Soft generation targets, observation mask, and optional per-row provenance."""

    values: torch.Tensor
    observed: torch.Tensor
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.values.shape != self.observed.shape or self.values.ndim != 2:
            raise ValueError(
                "condition values and observed mask must have matching [B,C] shapes"
            )
        if self.values.shape[1] != len(CONDITION_NAMES):
            raise ValueError(f"conditions must contain {len(CONDITION_NAMES)} fields")


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal scalar-time embedding projected to model width."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return shape [B, d_model] time embeddings."""
        if t.ndim != 1:
            raise ValueError("t must have shape [B]")
        half = self.d_model // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.d_model:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
        return self.proj(emb)


class NoiseSchedule(abc.ABC):
    """Abstract noise schedule for masked CTMC training and sampling."""

    @abc.abstractmethod
    def kappa(self, t: float) -> float:
        """Return reveal probability at time t."""

    @abc.abstractmethod
    def kappa_dot(self, t: float) -> float:
        """Return derivative of reveal probability at time t."""


class SinSquaredSchedule(NoiseSchedule):
    """kappa(t)=sin^2(pi*t/2)."""

    def kappa(self, t: float) -> float:
        return math.sin(math.pi * float(t) / 2.0) ** 2

    def kappa_dot(self, t: float) -> float:
        return 0.5 * math.pi * math.sin(math.pi * float(t))


def build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin cache."""
    half = head_dim // 2
    freqs = 1.0 / (10_000 ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # [L, half]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding."""
    # x: [B, n_heads, L, head_dim]
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated


class AdaLNModulation(nn.Module):
    """Produces per-layer scale/shift/gate from a time embedding."""

    def __init__(self, d_model: int, time_dim: int, n_outputs: int = 6) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.to_params = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, n_outputs * d_model),
        )
        nn.init.zeros_(self.to_params[-1].weight)
        nn.init.zeros_(self.to_params[-1].bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        params = self.to_params(t_emb)
        return params.chunk(self.n_outputs, dim=-1)


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, l, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, n_heads, L, head_dim]

        cos, sin = build_rope_cache(l, self.head_dim, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = ~key_padding_mask[:, None, None, :]  # SDPA expects True = attend
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.out_proj(out)


class AdaLNEncoderLayer(nn.Module):
    """Transformer encoder layer with AdaLN time conditioning (DiT-style)."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, time_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.modulation = AdaLNModulation(d_model, time_dim, n_outputs=6)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(t_emb)

        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out = self.attn(h, key_padding_mask)
        x = x + gate1.unsqueeze(1) * self.dropout(attn_out)

        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        ff_out = self.ff(h)
        x = x + gate2.unsqueeze(1) * self.dropout(ff_out)
        return x


class AdaLNEncoder(nn.Module):
    """Stack of AdaLN-modulated encoder layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
        time_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaLNEncoderLayer(d_model, n_heads, d_ff, dropout, time_dim)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, t_emb, key_padding_mask)
        return x


class ConvStem(nn.Module):
    """Depthwise-separable Conv1d stem for local motif capture."""

    def __init__(
        self, d_model: int, kernel_size: int = 7, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]; zero out PAD positions before conv
        x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        h = x_masked.transpose(1, 2)  # [B, D, L]
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = h.transpose(1, 2)  # [B, L, D]
        return self.norm(x + self.dropout(h))


class CTMCDenoiser(nn.Module):
    """Time-conditioned bidirectional Transformer denoiser with AdaLN."""

    def __init__(self, config: AMPConfig, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(
            vocab_size, config.d_model, padding_idx=pad_idx
        )
        self.time_embedding = SinusoidalTimeEmbedding(config.d_model)
        self.condition_embedding = nn.Sequential(
            nn.Linear(config.condition_dim * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.conv_stem = ConvStem(config.d_model, kernel_size=7, dropout=config.dropout)

        self.encoder = AdaLNEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            dropout=config.dropout,
            time_dim=config.d_model,
        )
        self.output_head = nn.Linear(config.d_model, vocab_size)

        # Length prediction head
        self.length_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.max_length - config.min_length + 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].zero_()

    def encode_hidden(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditions: ConditionVector | None = None,
    ) -> torch.Tensor:
        """Return hidden states with shape [B, L, d_model]."""
        hidden = self.token_embedding(x_t)
        pad_mask = x_t.eq(self.pad_idx)
        hidden = self.conv_stem(hidden, pad_mask)
        t_emb = self.time_embedding(t)
        if conditions is not None:
            if conditions.values.shape[0] != x_t.shape[0]:
                raise ValueError("condition batch size must match token batch size")
            values = conditions.values.to(device=x_t.device, dtype=hidden.dtype)
            observed = conditions.observed.to(device=x_t.device, dtype=hidden.dtype)
            t_emb = t_emb + self.condition_embedding(
                torch.cat([values * observed, observed], dim=-1)
            )
        return self.encoder(hidden, t_emb, pad_mask)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditions: ConditionVector | None = None,
    ) -> torch.Tensor:
        """Return per-position vocabulary logits [B, L, V]."""
        hidden = self.encode_hidden(x_t, t, conditions)
        return self.output_head(hidden)

    def predict_length_logits(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditions: ConditionVector | None = None,
    ) -> torch.Tensor:
        """Predict target length logits [B, num_lengths]."""
        hidden = self.encode_hidden(x_t, t, conditions)
        pad_mask = x_t.eq(self.pad_idx)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.length_head(pooled)

    def guided_logits(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditions: ConditionVector | None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Classifier-free guidance using the documented conditional/unconditional formula."""
        if conditions is None or cfg_scale == 0.0:
            return self(x_t, t)
        conditional = self(x_t, t, conditions)
        if cfg_scale == 1.0:
            return conditional
        unconditional = self(x_t, t, None)
        return unconditional + float(cfg_scale) * (conditional - unconditional)

    def save(self, path: Path) -> None:
        """Save model state dict atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "checkpoint_version": 2,
                "canvas_semantics": "residues_plus_eos",
                "state_dict": self.state_dict(),
            },
            tmp,
        )
        tmp.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        config: AMPConfig,
        vocab_size: int,
        pad_idx: int,
        device: torch.device,
    ) -> CTMCDenoiser:
        """Load a model from checkpoint."""
        model = cls(config=config, vocab_size=vocab_size, pad_idx=pad_idx)
        state = torch.load(path, map_location=device, weights_only=False)
        if (
            not isinstance(state, dict)
            or state.get("checkpoint_version") != 2
            or state.get("canvas_semantics") != "residues_plus_eos"
        ):
            raise ValueError(
                "Unsupported legacy CTMC checkpoint; canvas semantics changed and explicit migration is required"
            )
        model.load_state_dict(state["state_dict"])
        model.to(device)
        return model


def initialize_amino_acid_embeddings(
    model: CTMCDenoiser,
    esm_embedding_weight: torch.Tensor,
    esm_token_ids: dict[str, int],
    encoder: AMPCanvasEncoder,
    source_revision: str | None = None,
) -> dict:
    """Copy semantically matched ESM amino-acid rows into the student embedding.

    Special-token rows remain independently initialized. A deterministic truncated
    identity/zero-padded projection is used when hidden widths differ, avoiding an
    unseeded randomly trained adapter during initialization.
    """
    source = esm_embedding_weight.detach().float()
    target_width = model.token_embedding.embedding_dim
    if source.ndim != 2:
        raise ValueError("ESM embedding weight must be [vocabulary, hidden]")
    if source.shape[1] >= target_width:
        projection = torch.eye(source.shape[1], device=source.device)[:, :target_width]
    else:
        projection = torch.zeros((source.shape[1], target_width), device=source.device)
        projection[:, : source.shape[1]] = torch.eye(
            source.shape[1], device=source.device
        )
    copied: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for aa in encoder.vocab:
            if aa not in esm_token_ids:
                raise ValueError(
                    f"ESM tokenizer does not expose canonical residue {aa}"
                )
            source_id = int(esm_token_ids[aa])
            target_id = encoder.token_to_idx[aa]
            model.token_embedding.weight[target_id].copy_(
                source[source_id] @ projection.to(source.device)
            )
            copied[aa] = {"source_token_id": source_id, "student_token_id": target_id}
    return {
        "source_revision": source_revision,
        "mapping": copied,
        "projection": "deterministic_prefix_truncate_or_zero_pad",
    }


@dataclass(frozen=True)
class ReverseGenerationConfig:
    """Configuration for reverse CTMC tau-leaping generation."""

    steps: int
    temperature: float
    seed: int
    confidence_reveal_fraction: float = 0.0
    conditions: ConditionVector | None = None
    cfg_scale: float = 1.0


class TauLeapingSampler:
    """Constraint-aware batched reverse generator with optional DFM stochastic corrector."""

    def __init__(
        self,
        model: CTMCDenoiser,
        encoder: AMPCanvasEncoder,
        schedule: NoiseSchedule,
        min_length: int,
        length_prior: list[float] | None = None,
        dfm_stochasticity: float = 0.0,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.schedule = schedule
        self.min_length = min_length
        self.length_prior = length_prior
        self.dfm_stochasticity = dfm_stochasticity  # eta parameter from DFM paper
        self.device = next(model.parameters()).device

    def _allowed_indices(self, position: int) -> list[int]:
        if position < self.min_length:
            return list(self.encoder.amino_indices)
        return list(self.encoder.amino_indices) + [self.encoder.eos_idx]

    def _sample_target_lengths(
        self, n: int, torch_rng: torch.Generator
    ) -> torch.Tensor:
        """Sample peptide lengths in [min_length, max_length]. EOS is placed at that index."""
        length_values = torch.arange(
            self.min_length,
            self.encoder.max_length + 1,
            device=self.device,
            dtype=torch.long,
        )
        if self.length_prior is None:
            probs = torch.ones(
                (length_values.numel(),), device=self.device, dtype=torch.float32
            )
        else:
            probs = torch.as_tensor(
                self.length_prior, device=self.device, dtype=torch.float32
            )
            if probs.numel() != length_values.numel():
                raise ValueError(
                    f"length_prior must have {length_values.numel()} entries "
                    f"for lengths {self.min_length}..{self.encoder.max_length}"
                )
        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum().clamp_min(1e-8)
        return length_values[
            torch.multinomial(probs, n, replacement=True, generator=torch_rng)
        ]

    def _terminate(self, canvases: torch.Tensor, batch_idx: int, pos: int) -> None:
        canvases[batch_idx, pos] = self.encoder.eos_idx
        if pos + 1 < self.encoder.canvas_length:
            canvases[batch_idx, pos + 1 :] = self.encoder.pad_idx

    def _apply_stochastic_corrector(
        self, canvases: torch.Tensor, torch_rng: torch.Generator
    ) -> None:
        """Apply DFM-style stochastic corrector: re-mask some revealed positions with probability eta.

        This implements the "detailed-balance-respecting stochastic component" from
        Campbell et al. (DFM/Multiflow paper, arXiv:2402.04997), Section 2.1.
        At eta=0, this is a no-op (pure unmasking). At eta>0, allows the sampler to
        revisit and correct earlier commitments before terminal time.
        """
        if self.dfm_stochasticity <= 0.0:
            return

        # Probability of re-masking any non-EOS, non-PAD revealed position
        remask_prob = min(self.dfm_stochasticity * 0.01, 0.5)  # scale eta to [0, 0.5]

        for batch_idx in range(canvases.shape[0]):
            # Find positions that can be re-masked: revealed (not MASK), not EOS, not PAD
            is_mask = canvases[batch_idx] == self.encoder.mask_idx
            is_eos = canvases[batch_idx] == self.encoder.eos_idx
            is_pad = canvases[batch_idx] == self.encoder.pad_idx
            can_remask = ~(is_mask | is_eos | is_pad)

            remaskable_positions = torch.nonzero(can_remask, as_tuple=False).flatten()
            if remaskable_positions.numel() == 0:
                continue

            # Randomly select which remaskable positions to re-mask
            remask_mask = (
                torch.rand(
                    remaskable_positions.numel(),
                    generator=torch_rng,
                    device=self.device,
                )
                < remask_prob
            )
            positions_to_remask = remaskable_positions[remask_mask]
            canvases[batch_idx, positions_to_remask] = self.encoder.mask_idx

    def sample(self, gen_config: ReverseGenerationConfig) -> str:
        """Sample one peptide string."""
        return self.sample_batch(gen_config, 1)[0]

    def sample_batch(self, gen_config: ReverseGenerationConfig, n: int) -> list[str]:
        """Sample a batch of peptide strings in parallel."""
        if n <= 0:
            return []

        torch_rng = torch.Generator(device=self.device).manual_seed(gen_config.seed)
        canvases = torch.full(
            (n, self.encoder.canvas_length),
            fill_value=self.encoder.mask_idx,
            dtype=torch.long,
            device=self.device,
        )
        target_len = self._sample_target_lengths(n, torch_rng)

        self.model.eval()
        with torch.inference_mode():
            for step in range(gen_config.steps):
                if step > 0 and self.dfm_stochasticity > 0:
                    self._apply_stochastic_corrector(canvases, torch_rng)
                masked = canvases.eq(self.encoder.mask_idx)
                if not torch.any(masked):
                    break

                t_value = min((step + 1) / gen_config.steps, 1.0 - 1e-6)
                t = torch.full((n,), float(t_value), device=self.device)
                logits = self.model.guided_logits(
                    canvases, t, gen_config.conditions, gen_config.cfg_scale
                )

                for batch_idx in range(n):
                    mask_positions = torch.nonzero(
                        masked[batch_idx], as_tuple=False
                    ).flatten()
                    if mask_positions.numel() == 0:
                        continue

                    stop_at = int(target_len[batch_idx].item())
                    reveal_count = max(
                        1, math.ceil(mask_positions.numel() / (gen_config.steps - step))
                    )
                    chosen = mask_positions
                    if reveal_count < mask_positions.numel():
                        conf_take = round(
                            reveal_count * gen_config.confidence_reveal_fraction
                        )
                        conf_take = min(conf_take, reveal_count)
                        if conf_take > 0:
                            conf = logits[batch_idx, mask_positions].max(dim=-1).values
                            top_idx = torch.topk(
                                conf, k=conf_take, largest=True
                            ).indices
                            chosen_conf = mask_positions[top_idx]
                        else:
                            chosen_conf = torch.empty(
                                0, device=self.device, dtype=torch.long
                            )
                        remaining = mask_positions[
                            ~torch.isin(mask_positions, chosen_conf)
                        ]
                        rem_take = reveal_count - conf_take
                        if rem_take > 0 and remaining.numel() > 0:
                            perm = torch.randperm(
                                remaining.numel(),
                                generator=torch_rng,
                                device=self.device,
                            )
                            chosen_rand = remaining[perm[:rem_take]]
                            chosen = torch.cat([chosen_conf, chosen_rand])
                        else:
                            chosen = chosen_conf
                        chosen, _ = torch.sort(chosen)

                    for pos in chosen.tolist():
                        if canvases[batch_idx, pos].item() != self.encoder.mask_idx:
                            continue
                        if pos > stop_at:
                            canvases[batch_idx, pos] = self.encoder.pad_idx
                            continue
                        if pos == stop_at:
                            if stop_at < self.encoder.canvas_length:
                                self._terminate(canvases, batch_idx, pos)
                            break

                        allowed = list(self.encoder.amino_indices)
                        local_logits = logits[batch_idx, pos, allowed] / max(
                            gen_config.temperature, 1e-6
                        )
                        probs = torch.softmax(local_logits, dim=-1)
                        pick_idx = torch.multinomial(
                            probs, num_samples=1, generator=torch_rng
                        ).item()
                        canvases[batch_idx, pos] = allowed[pick_idx]

            unresolved = canvases.eq(self.encoder.mask_idx)
            if torch.any(unresolved):
                logits = self.model.guided_logits(
                    canvases,
                    torch.ones((n,), device=self.device),
                    gen_config.conditions,
                    gen_config.cfg_scale,
                )
                for batch_idx in range(n):
                    stop_at = int(target_len[batch_idx].item())
                    positions = (
                        torch.nonzero(unresolved[batch_idx], as_tuple=False)
                        .flatten()
                        .tolist()
                    )
                    for pos in positions:
                        if canvases[batch_idx, pos].item() != self.encoder.mask_idx:
                            continue
                        if pos > stop_at:
                            canvases[batch_idx, pos] = self.encoder.pad_idx
                            continue
                        if pos == stop_at:
                            if stop_at < self.encoder.canvas_length:
                                self._terminate(canvases, batch_idx, pos)
                            break

                        allowed = list(self.encoder.amino_indices)
                        local_logits = logits[batch_idx, pos, allowed]
                        token = allowed[torch.argmax(local_logits).item()]
                        canvases[batch_idx, pos] = token

        sequences: list[str] = []
        for canvas in canvases.detach().cpu():
            if (canvas == self.encoder.mask_idx).any():
                canvas = torch.where(
                    canvas == self.encoder.mask_idx,
                    torch.tensor(self.encoder.pad_idx),
                    canvas,
                )
            sequences.append(self.encoder.decode(canvas))
        return sequences


class GuidedTauLeapingSampler(TauLeapingSampler):
    def __init__(
        self,
        model,
        encoder,
        schedule,
        min_length,
        discriminator=None,
        length_prior=None,
        guidance_every=8,
        guidance_keep_frac=0.5,
        dfm_stochasticity=0.0,
    ):
        super().__init__(
            model=model,
            encoder=encoder,
            schedule=schedule,
            min_length=min_length,
            length_prior=length_prior,
            dfm_stochasticity=dfm_stochasticity,
        )
        self.discriminator = discriminator
        self.guidance_every = guidance_every
        self.guidance_keep_frac = guidance_keep_frac

    def _rerank_and_resample(self, canvases, torch_rng):
        if self.discriminator is None:
            return canvases
        import numpy as np

        decoded = [
            self.encoder.decode(
                torch.where(
                    canvas == self.encoder.mask_idx,
                    torch.tensor(self.encoder.amino_indices[0]),
                    canvas,
                )
            )
            for canvas in canvases.detach().cpu()
        ]
        probs = self.discriminator.predict_proba(decoded)
        n = canvases.shape[0]
        keep_n = max(1, round(n * self.guidance_keep_frac))
        top_idx = np.argsort(-probs)[:keep_n]
        resample_idx = torch.from_numpy(
            np.random.default_rng().choice(top_idx, size=n, replace=True)
        ).to(canvases.device)
        return canvases[resample_idx]
