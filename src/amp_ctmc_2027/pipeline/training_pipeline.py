from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from amp_ctmc_2027.config import AMPConfig, set_global_determinism
from amp_ctmc_2027.core import (
    CONDITION_NAMES,
    ConditionVector,
    CTMCDenoiser,
    SinSquaredSchedule,
    initialize_amino_acid_embeddings,
)
from amp_ctmc_2027.data.dataset import (
    AMPCanvasEncoder,
    AMPDataset,
    ConditionedAMPDataset,
    collate_conditioned,
)
from amp_ctmc_2027.data.conditions import (
    fit_condition_normalization,
    load_condition_table,
    panel_semantic_definition,
)
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.data.manifests import sha256_file, write_manifest
from amp_ctmc_2027.discriminator import DiscriminatorEnsemble, PeptideFeatureExtractor
from amp_ctmc_2027.infra import WandbTracker, log_device, resolve_device

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Pipeline that trains the denoiser and discriminator and writes checkpoints."""

    def __init__(self, config: AMPConfig, fasta_repo: FastaRepository) -> None:
        self.config = config
        self.repo = fasta_repo

    def _corrupt_batch(
        self, clean: torch.Tensor, t: torch.Tensor, encoder: AMPCanvasEncoder
    ) -> torch.Tensor:
        kappa = torch.sin(torch.pi * t / 2.0) ** 2
        random_draws = torch.rand(clean.shape, device=clean.device)
        non_pad = clean.ne(encoder.pad_idx)
        keep = random_draws < kappa.unsqueeze(1)
        corrupted = clean.clone()
        to_mask = non_pad & ~keep
        needs_mask = ~to_mask.any(dim=1) & non_pad.any(dim=1)
        if needs_mask.any():
            valid = non_pad.float()
            valid = valid.masked_fill(~needs_mask.unsqueeze(1), 0.0)
            probs = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            sampled = torch.multinomial(probs[needs_mask], num_samples=1).squeeze(1)
            rows = torch.nonzero(needs_mask, as_tuple=False).flatten()
            to_mask = to_mask.clone()
            to_mask[rows, sampled] = True
        corrupted[to_mask] = encoder.mask_idx
        return corrupted

    def _sample_t_curriculum(
        self, batch_size: int, epoch: int, total_epochs: int, device: torch.device
    ) -> torch.Tensor:
        """Sample t where 0 is fully masked and 1 is clean; early samples favor clean t."""
        warmup_frac = min(1.0, epoch / max(1, total_epochs // 4))
        base = torch.rand((batch_size,), device=device)
        if warmup_frac >= 1.0:
            return base
        power = 2.0 + 3.0 * (1.0 - warmup_frac)
        return 1.0 - (1.0 - base).pow(power)

    def _load_condition_rows(self, sequences: list[str]) -> tuple[list[dict], dict]:
        if (
            self.config.condition_table_path is None
            or self.config.condition_manifest_path is None
        ):
            raise ValueError(
                "property_conditioned mode requires a condition table and manifest"
            )
        return load_condition_table(
            self.repo.resolve(self.config.condition_table_path),
            self.repo.resolve(self.config.condition_manifest_path),
            sequences,
        )

    def _loss_for_batch(
        self,
        model: CTMCDenoiser,
        clean: torch.Tensor,
        encoder: AMPCanvasEncoder,
        epoch: int = 0,
        total_epochs: int = 1,
        validation: bool = False,
        condition_values: torch.Tensor | None = None,
        condition_observed: torch.Tensor | None = None,
        force_unconditional: bool = False,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        batch_size = clean.size(0)
        if validation:
            # Fixed stratified evaluation distribution, independent of training epoch.
            t = torch.linspace(0.0, 1.0, batch_size + 2, device=clean.device)[1:-1]
        else:
            t = self._sample_t_curriculum(batch_size, epoch, total_epochs, clean.device)
        x_t = self._corrupt_batch(clean, t, encoder)
        valid = clean.ne(encoder.pad_idx)
        supervised = x_t.eq(encoder.mask_idx) & valid
        stats = {
            "valid_tokens": float(valid.sum().item()),
            "masked_tokens": float(supervised.sum().item()),
            "supervised_tokens": float(supervised.sum().item()),
            "aa_total": 0.0,
            "aa_correct": 0.0,
            "pad_targets": 0.0,
            "eos_targets": 0.0,
            "length_loss": 0.0,
            "length_acc": 0.0,
            "condition_drop_fraction": 0.0,
        }
        if not torch.any(supervised):
            return None, stats

        conditions = None
        if (
            condition_values is not None
            and self.config.training_mode == "property_conditioned"
            and not force_unconditional
        ):
            observed = condition_observed.bool().clone()
            drop = (
                torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
                if validation
                else torch.rand(batch_size, device=clean.device)
                < self.config.condition_dropout
            )
            observed[drop] = False
            conditions = ConditionVector(condition_values, observed)
            stats["condition_drop_fraction"] = float(drop.float().mean().item())
        use_amp = (
            self.config.mixed_precision
            and clean.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x_t, t, conditions)
            length_logits = model.predict_length_logits(x_t, t, conditions)
        masked_logits = logits[supervised].float()
        masked_target = clean[supervised]
        loss = F.cross_entropy(masked_logits, masked_target, label_smoothing=0.05)

        pred = masked_logits.argmax(dim=-1)
        aa = masked_target < 20
        stats["aa_total"] = float(aa.sum().item())
        stats["aa_correct"] = (
            float((pred[aa] == masked_target[aa]).sum().item()) if aa.any() else 0.0
        )
        stats["pad_targets"] = float(masked_target.eq(encoder.pad_idx).sum().item())
        stats["eos_targets"] = float(masked_target.eq(encoder.eos_idx).sum().item())

        true_lengths = clean.ne(encoder.pad_idx).sum(dim=1) - 1
        length_target = (true_lengths - self.config.min_length).clamp(
            0, self.config.max_length - self.config.min_length
        )
        length_loss = F.cross_entropy(length_logits.float(), length_target)
        total_loss = loss + 0.1 * length_loss

        length_pred = length_logits.argmax(dim=-1)
        length_acc = (length_pred == length_target).float().mean()
        stats["length_loss"] = float(length_loss.item())
        stats["length_acc"] = float(length_acc.item())

        stats["conditional_denoising_loss"] = (
            float(total_loss.item()) if conditions is not None else 0.0
        )
        stats["unconditional_denoising_loss"] = (
            float(total_loss.item()) if conditions is None else 0.0
        )
        return total_loss, stats

    @staticmethod
    def _accumulate(total: dict[str, float], stats: dict[str, float]) -> None:
        for key, value in stats.items():
            total[key] = total.get(key, 0.0) + value

    def run(self) -> None:
        """Execute end-to-end training and checkpointing."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
        set_global_determinism(self.config.seed)
        logger.info(
            "Training config batch_size=%d lr=%.3e dropout=%.3f d_model=%d n_layers=%d n_heads=%d",
            self.config.batch_size,
            self.config.learning_rate,
            self.config.dropout,
            self.config.d_model,
            self.config.n_layers,
            self.config.n_heads,
        )

        split_path = self.repo.resolve(self.config.cluster_split_manifest_path)
        split_hash = sha256_file(split_path) if split_path.is_file() else None

        training_sequences = self.repo.read_sequences(self.config.training_fasta_path)
        background_sequences = self.repo.read_sequences(
            self.config.background_fasta_path
        )
        if not training_sequences:
            raise RuntimeError("Training FASTA is empty")
        if not background_sequences:
            raise RuntimeError("Background FASTA is empty")

        encoder = AMPCanvasEncoder(
            max_length=self.config.max_length, vocab=self.config.vocab
        )
        logger.info(
            "Tokenizer vocab_size=%d eos=%d mask=%d pad=%d",
            encoder.vocab_size,
            encoder.eos_idx,
            encoder.mask_idx,
            encoder.pad_idx,
        )

        if self.config.debug_overfit_samples is not None:
            n = min(self.config.debug_overfit_samples, len(training_sequences))
            train_sequences = training_sequences[:n]
            val_sequences = training_sequences[:n]
            logger.warning(
                "OVERFIT DEBUG MODE: train and val use the same %d sequences", n
            )
        else:
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Production CTMC training requires a cluster-level split manifest: {split_path}"
                )
            with split_path.open(newline="", encoding="utf-8") as stream:
                split_rows = list(csv.DictReader(stream))
            sequence_split: dict[str, tuple[str, str]] = {}
            for row in split_rows:
                seq = "".join(row["sequence"].split()).upper()
                if seq in sequence_split and sequence_split[seq][1] != row["split"]:
                    raise ValueError(f"Exact sequence crosses split assignments: {seq}")
                sequence_split[seq] = (row["split_cluster_id"], row["split"])
            cluster_split: dict[str, str] = {}
            for cluster_id, split in sequence_split.values():
                if cluster_id in cluster_split and cluster_split[cluster_id] != split:
                    raise ValueError(f"Homology cluster crosses splits: {cluster_id}")
                cluster_split[cluster_id] = split
            missing = sorted(set(training_sequences) - set(sequence_split))
            if missing:
                raise ValueError(
                    f"Training FASTA has {len(missing)} sequences absent from split manifest"
                )
            train_sequences = [
                s for s in training_sequences if sequence_split[s][1] == "train"
            ]
            val_sequences = [
                s for s in training_sequences if sequence_split[s][1] == "val"
            ]
            missing_background = sorted(set(background_sequences) - set(sequence_split))
            if missing_background:
                raise ValueError(
                    f"Background FASTA has {len(missing_background)} sequences absent from split manifest"
                )
            background_sequences = [
                sequence
                for sequence in background_sequences
                if sequence_split[sequence][1] == "train"
            ]
            if not train_sequences or not val_sequences:
                raise ValueError(
                    "Cluster split manifest requires nonempty train and val splits"
                )
            if not background_sequences:
                raise ValueError(
                    "Cluster split has no train-split background sequences"
                )

        if split_hash is None:
            split_hash = hashlib.sha256(
                ("debug-train-split\n" + "\n".join(train_sequences)).encode()
            ).hexdigest()

        device = resolve_device(self.config)
        log_device(device)
        pin_memory = device.type == "cuda"

        condition_rows: list[dict] | None = None
        condition_manifest: dict = {}
        condition_semantics: dict | None = None
        condition_normalization: dict | None = None
        if self.config.training_mode == "property_conditioned":
            if split_hash is None:
                raise ValueError(
                    "Property-conditioned training requires the hashed CTMC split manifest"
                )
            condition_rows, condition_manifest = self._load_condition_rows(
                training_sequences
            )
            condition_semantics = condition_manifest.get("semantic_definition")
            if not isinstance(condition_semantics, dict):
                raise ValueError("Condition cache is missing panel semantic definition")
            property_metadata_path = self.repo.resolve(
                self.config.property_metadata_path
            )
            property_metadata = json.loads(
                property_metadata_path.read_text(encoding="utf-8")
            )
            if condition_manifest.get("property_metadata_sha256") != sha256_file(
                property_metadata_path
            ) or condition_manifest.get("property_checkpoint_sha256") != sha256_file(
                property_metadata_path.parent / "model.pt"
            ):
                raise ValueError(
                    "Condition cache property checkpoint/metadata hashes differ"
                )
            if condition_semantics != property_metadata.get("semantic_definition"):
                raise ValueError(
                    "Condition cache semantics differ from property-model metadata"
                )
            condition_normalization = fit_condition_normalization(
                condition_rows, train_sequences, split_hash
            )
            if not any(any(row["observed"].values()) for row in condition_rows):
                raise ValueError("Condition table contains no observations")
            train_dataset = ConditionedAMPDataset(
                train_sequences, encoder, condition_rows, condition_normalization
            )
            val_dataset = ConditionedAMPDataset(
                val_sequences, encoder, condition_rows, condition_normalization
            )
            collate_fn = collate_conditioned
        else:
            property_metadata_path = self.repo.resolve(
                self.config.property_metadata_path
            )
            if property_metadata_path.is_file():
                property_metadata = json.loads(
                    property_metadata_path.read_text(encoding="utf-8")
                )
                condition_semantics = property_metadata.get("semantic_definition")
                if condition_semantics is not None:
                    condition_semantics = panel_semantic_definition(
                        condition_semantics["ranking_strains"],
                        condition_semantics["strain_groups"],
                        condition_semantics["activity_threshold_log2_mic"],
                        condition_semantics["activity_temperature"],
                        condition_semantics["broad_spectrum_aggregation"],
                    )
            train_dataset, val_dataset = (
                AMPDataset(train_sequences, encoder),
                AMPDataset(val_sequences, encoder),
            )
            collate_fn = None
            empty_rows = [
                {
                    "sequence": sequence,
                    "values": {name: 0.0 for name in CONDITION_NAMES},
                    "observed": {name: False for name in CONDITION_NAMES},
                }
                for sequence in train_sequences
            ]
            condition_normalization = fit_condition_normalization(
                empty_rows, train_sequences, split_hash
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        model = CTMCDenoiser(
            config=self.config,
            vocab_size=encoder.vocab_size,
            pad_idx=encoder.pad_idx,
        ).to(device)
        transfer_metadata = {"mode": self.config.ctmc_pretraining_mode}
        if "distill" in self.config.ctmc_pretraining_mode:
            raise RuntimeError(
                "Teacher distillation is selected but teacher-cache distillation is not connected to this trainer"
            )
        if self.config.ctmc_pretraining_mode == "embedding_init":
            try:
                from transformers import AutoTokenizer, EsmModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install the training extra to initialize from ESM embeddings"
                ) from exc
            source = self.config.teacher_model_name
            tokenizer = AutoTokenizer.from_pretrained(
                source, revision=self.config.teacher_revision
            )
            if self.config.teacher_checkpoint_dir is not None:
                teacher_dir = self.repo.resolve(self.config.teacher_checkpoint_dir)
                from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor

                metadata = json.loads(
                    (teacher_dir / "model_metadata.json").read_text(encoding="utf-8")
                )
                teacher_model, _ = ESMMultiTaskPredictor.from_pretrained(
                    metadata["model_name"],
                    metadata.get("revision"),
                    len(metadata["strain_ids"]),
                )
                teacher_model.load_state_dict(
                    torch.load(
                        teacher_dir / "model.pt", map_location="cpu", weights_only=True
                    )
                )
                teacher = teacher_model.backbone
                source_revision = metadata.get("revision") or metadata["model_name"]
            else:
                teacher = EsmModel.from_pretrained(
                    source, revision=self.config.teacher_revision
                )
                source_revision = self.config.teacher_revision or source
            residue_ids = {
                aa: int(tokenizer.convert_tokens_to_ids(aa)) for aa in self.config.vocab
            }
            transfer_metadata.update(
                initialize_amino_acid_embeddings(
                    model,
                    teacher.embeddings.word_embeddings.weight,
                    residue_ids,
                    encoder,
                    source_revision=source_revision,
                )
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.n_epochs,
                eta_min=self.config.learning_rate_min,
            )
        elif self.config.scheduler_type == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None

        schedule = SinSquaredSchedule()
        logger.info("Using schedule %s", schedule.__class__.__name__)
        tracker = WandbTracker(self.config, job_type="train")
        if tracker.enabled:
            tracker.log(
                {
                    "runtime/device": device.type,
                    "dataset/train_size": len(train_dataset),
                    "dataset/val_size": len(val_dataset),
                }
            )

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience = 0
        history: list[dict[str, float]] = []
        start_epoch = 0
        if self.config.resume_from is not None:
            resume_path = self.repo.resolve(self.config.resume_from)
            resume = torch.load(resume_path, map_location=device, weights_only=False)
            if (
                resume.get("checkpoint_version") != 2
                or resume.get("canvas_semantics") != "residues_plus_eos"
            ):
                raise ValueError(
                    "Resume checkpoint has incompatible canvas semantics/version"
                )
            model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            if scheduler is not None and resume.get("scheduler") is not None:
                scheduler.load_state_dict(resume["scheduler"])
            start_epoch = int(resume["epoch"])
            best_val_loss = float(resume["best_val_loss"])
            best_state = resume["best_state"]
            patience = int(resume["patience"])
            history = list(resume["history"])
            logger.info("Resuming optimizer/model state at epoch %d", start_epoch)

        for epoch in range(start_epoch, self.config.n_epochs):
            model.train()
            train_loss_total = 0.0
            train_batches = 0
            train_stats: dict[str, float] = {}
            for batch in tqdm(
                train_loader, desc=f"train-epoch-{epoch + 1}", leave=False
            ):
                if isinstance(batch, dict):
                    clean = batch["tokens"].to(device)
                    values = batch["condition_values"].to(device)
                    observed = batch["condition_observed"].to(device)
                else:
                    clean, values, observed = batch.to(device), None, None
                optimizer.zero_grad(set_to_none=True)
                loss, stats = self._loss_for_batch(
                    model,
                    clean,
                    encoder,
                    epoch,
                    self.config.n_epochs,
                    condition_values=values,
                    condition_observed=observed,
                )
                self._accumulate(train_stats, stats)
                if loss is not None and loss.requires_grad:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
                    optimizer.step()
                    train_loss_total += float(loss.item())
                    train_batches += 1

            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            val_stats: dict[str, float] = {}
            with torch.inference_mode():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        clean = batch["tokens"].to(device)
                        values = batch["condition_values"].to(device)
                        observed = batch["condition_observed"].to(device)
                    else:
                        clean, values, observed = batch.to(device), None, None
                    loss, stats = self._loss_for_batch(
                        model,
                        clean,
                        encoder,
                        validation=True,
                        condition_values=values,
                        condition_observed=observed,
                    )
                    self._accumulate(val_stats, stats)
                    if loss is not None:
                        val_loss_total += float(loss.item())
                        val_batches += 1
                    if values is not None:
                        provenance = batch["condition_provenance"]
                        for label in ("measured", "teacher"):
                            selected = [
                                idx
                                for idx, row_sources in enumerate(provenance)
                                if any(
                                    source == label and bool(observed[idx, j])
                                    for j, source in enumerate(row_sources)
                                )
                            ]
                            if selected:
                                subset_loss, _ = self._loss_for_batch(
                                    model,
                                    clean[selected],
                                    encoder,
                                    validation=True,
                                    condition_values=values[selected],
                                    condition_observed=observed[selected],
                                )
                                if subset_loss is not None:
                                    key = f"{label}_conditional_loss"
                                    val_stats[key] = val_stats.get(key, 0.0) + float(
                                        subset_loss.item()
                                    )
                                    val_stats[f"{label}_conditional_batches"] = (
                                        val_stats.get(
                                            f"{label}_conditional_batches", 0.0
                                        )
                                        + 1.0
                                    )
                        uncond_loss, uncond_stats = self._loss_for_batch(
                            model,
                            clean,
                            encoder,
                            validation=True,
                            condition_values=values,
                            condition_observed=observed,
                            force_unconditional=True,
                        )
                        if uncond_loss is not None:
                            val_stats["unconditional_denoising_loss"] = val_stats.get(
                                "unconditional_denoising_loss", 0.0
                            ) + float(uncond_loss.item())

            train_loss = train_loss_total / max(train_batches, 1)
            val_loss = val_loss_total / max(val_batches, 1)
            if condition_rows is not None:
                val_stats["conditional_denoising_loss"] = val_loss
                val_stats["unconditional_denoising_loss"] = val_stats.get(
                    "unconditional_denoising_loss", 0.0
                ) / max(val_batches, 1)
                for provenance in ("measured", "teacher"):
                    batches = val_stats.get(f"{provenance}_conditional_batches", 0.0)
                    val_stats[f"{provenance}_conditional_loss"] = val_stats.get(
                        f"{provenance}_conditional_loss", 0.0
                    ) / max(batches, 1.0)
            masked_frac = train_stats.get("masked_tokens", 0.0) / max(
                train_stats.get("valid_tokens", 0.0), 1.0
            )
            aa_acc = train_stats.get("aa_correct", 0.0) / max(
                train_stats.get("aa_total", 0.0), 1.0
            )
            pad_tgt = train_stats.get("pad_targets", 0.0) / max(
                train_stats.get("supervised_tokens", 0.0), 1.0
            )
            eos_tgt = train_stats.get("eos_targets", 0.0) / max(
                train_stats.get("supervised_tokens", 0.0), 1.0
            )
            length_loss = train_stats.get("length_loss", 0.0) / max(train_batches, 1)
            length_acc = train_stats.get("length_acc", 0.0) / max(train_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "condition_drop_fraction": train_stats.get(
                        "condition_drop_fraction", 0.0
                    )
                    / max(len(train_loader), 1),
                }
            )
            logger.info(
                "Epoch %d train_loss=%.6f val_loss=%.6f masked_frac=%.3f aa_acc=%.3f pad_tgt=%.6f eos_tgt=%.3f length_loss=%.6f length_acc=%.3f lr=%.3e",
                epoch + 1,
                train_loss,
                val_loss,
                masked_frac,
                aa_acc,
                pad_tgt,
                eos_tgt,
                length_loss,
                length_acc,
                current_lr,
            )
            tracker.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/masked_frac": masked_frac,
                    "train/aa_acc": aa_acc,
                    "train/pad_tgt": pad_tgt,
                    "train/eos_tgt": eos_tgt,
                    "train/length_loss": length_loss,
                    "train/length_acc": length_acc,
                    "train/lr": current_lr,
                    "epoch": epoch + 1,
                },
                step=epoch + 1,
            )

            if scheduler is not None:
                scheduler.step()

            checkpoint_dir = self.repo.resolve(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            progress_path = checkpoint_dir / "training_state.pt"
            tmp_progress = progress_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "checkpoint_version": 2,
                    "canvas_semantics": "residues_plus_eos",
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "best_val_loss": best_val_loss,
                    "best_state": best_state,
                    "patience": patience,
                    "history": history,
                },
                tmp_progress,
            )
            tmp_progress.replace(progress_path)

            if val_loss < best_val_loss - self.config.early_stopping_threshold:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint")
        model.load_state_dict(best_state)

        feature_extractor = PeptideFeatureExtractor(alphabet=self.config.vocab)
        feature_stats = feature_extractor.fit_stats(train_sequences)
        discriminator = DiscriminatorEnsemble(
            extractor=feature_extractor,
            seeds=[
                self.config.seed + i
                for i in range(self.config.discriminator_ensemble_size)
            ],
        )
        discriminator.fit(train_sequences, background_sequences)

        checkpoint_dir = self.repo.resolve(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        condition_contract = {
            "training_mode": self.config.training_mode,
            "condition_names": list(CONDITION_NAMES),
            "condition_normalization": {
                **condition_normalization,
            },
            "semantic_definition": condition_semantics,
            "condition_table_sha256": condition_manifest.get("table_sha256"),
            "condition_manifest_sha256": sha256_file(
                self.repo.resolve(self.config.condition_manifest_path)
            )
            if condition_rows is not None
            else None,
            "property_model_checkpoint_sha256": condition_manifest.get(
                "property_checkpoint_sha256"
            ),
            "measured_count": sum(
                p == "measured"
                for row in (condition_rows or [])
                for p in row["provenance"].values()
            ),
            "pseudo_label_count": sum(
                p == "teacher"
                for row in (condition_rows or [])
                for p in row["provenance"].values()
            ),
            "condition_dropout": self.config.condition_dropout,
            "training_split_sha256": split_hash,
        }
        model.save(
            checkpoint_dir / "model.pt", encoder=encoder, contract=condition_contract
        )
        self.repo.write_json_atomic(
            checkpoint_dir / "config.json",
            {
                **self.config.model_dump(mode="json"),
                "condition_schema": list(CONDITION_NAMES),
                "condition_normalization": condition_contract[
                    "condition_normalization"
                ],
                "condition_semantics": condition_contract["semantic_definition"],
                "condition_table_sha256": condition_contract["condition_table_sha256"],
                "condition_manifest_sha256": condition_contract[
                    "condition_manifest_sha256"
                ],
                "property_model_checkpoint_sha256": condition_contract[
                    "property_model_checkpoint_sha256"
                ],
                "measured_count": condition_contract["measured_count"],
                "pseudo_label_count": condition_contract["pseudo_label_count"],
                "training_split_sha256": condition_contract["training_split_sha256"],
            },
        )
        self.repo.write_json_atomic(
            checkpoint_dir / "tokenizer.json", encoder.to_metadata()
        )
        self.repo.write_json_atomic(
            checkpoint_dir / "training_stats.json",
            {
                "checkpoint_version": 2,
                "canvas_semantics": "residues_plus_eos",
                "conditioning_contract": condition_contract,
                "validation_condition_metrics": {
                    k: v
                    for k, v in val_stats.items()
                    if k
                    in {
                        "conditional_denoising_loss",
                        "unconditional_denoising_loss",
                        "condition_drop_fraction",
                        "measured_conditional_loss",
                        "teacher_conditional_loss",
                    }
                },
                "best_val_loss": best_val_loss,
                "history": history,
                "feature_stats": asdict(feature_stats),
            },
        )
        self.repo.write_json_atomic(
            checkpoint_dir / "pretrained_transfer.json", transfer_metadata
        )
        write_manifest(
            checkpoint_dir / "dataset_manifest.json",
            {
                "training_fasta": str(
                    self.repo.resolve(self.config.training_fasta_path)
                ),
                "training_fasta_sha256": sha256_file(
                    self.repo.resolve(self.config.training_fasta_path)
                ),
                "background_fasta": str(
                    self.repo.resolve(self.config.background_fasta_path)
                ),
                "background_fasta_sha256": sha256_file(
                    self.repo.resolve(self.config.background_fasta_path)
                ),
                "cluster_split_manifest": str(
                    self.repo.resolve(self.config.cluster_split_manifest_path)
                )
                if self.config.cluster_split_manifest_path.exists()
                else None,
                "cluster_split_sha256": sha256_file(
                    self.repo.resolve(self.config.cluster_split_manifest_path)
                )
                if self.config.cluster_split_manifest_path.exists()
                else None,
                "train_count": len(train_sequences),
                "validation_count": len(val_sequences),
                "config": self.config.model_dump(mode="json"),
                "checkpoint_version": 2,
                "canvas_semantics": "residues_plus_eos",
            },
        )
        production_manifest = self.repo.resolve(
            Path("data/training/dataset_manifest.json")
        )
        existing_manifest = {}
        if production_manifest.is_file():
            existing_manifest = json.loads(
                production_manifest.read_text(encoding="utf-8")
            )
        existing_manifest.update(
            {
                "ctmc_training_fasta": str(
                    self.repo.resolve(self.config.training_fasta_path)
                ),
                "ctmc_training_fasta_sha256": sha256_file(
                    self.repo.resolve(self.config.training_fasta_path)
                ),
                "ctmc_cluster_split_manifest": str(
                    self.repo.resolve(self.config.cluster_split_manifest_path)
                )
                if self.config.cluster_split_manifest_path.exists()
                else None,
                "ctmc_cluster_split_sha256": sha256_file(
                    self.repo.resolve(self.config.cluster_split_manifest_path)
                )
                if self.config.cluster_split_manifest_path.exists()
                else None,
                "ctmc_train_count": len(train_sequences),
                "ctmc_validation_count": len(val_sequences),
                "ctmc_checkpoint_format_version": 2,
            }
        )
        self.repo.write_json_atomic(production_manifest, existing_manifest)
        discriminator.save(checkpoint_dir / "discriminator.json")
        tracker.log({"val/best_loss": best_val_loss})
        tracker.finish()
        logger.info("Checkpoint written to %s", checkpoint_dir)
