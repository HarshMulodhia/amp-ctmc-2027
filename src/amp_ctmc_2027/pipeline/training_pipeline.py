from __future__ import annotations

import logging
import csv
import json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from amp_ctmc_2027.config import AMPConfig, set_global_determinism
from amp_ctmc_2027.core import CTMCDenoiser, SinSquaredSchedule
from amp_ctmc_2027.core import initialize_amino_acid_embeddings
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder, AMPDataset
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.data.manifests import sha256_file, write_manifest
from amp_ctmc_2027.discriminator import DiscriminatorEnsemble, PeptideFeatureExtractor
from amp_ctmc_2027.infra import log_device, resolve_device, WandbTracker

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Pipeline that trains the denoiser and discriminator and writes checkpoints."""

    def __init__(self, config: AMPConfig, fasta_repo: FastaRepository) -> None:
        self.config = config
        self.repo = fasta_repo

    def _corrupt_batch(self, clean: torch.Tensor, t: torch.Tensor, encoder: AMPCanvasEncoder) -> torch.Tensor:
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

    def _sample_t_curriculum(self, batch_size: int, epoch: int, total_epochs: int, device: torch.device) -> torch.Tensor:
        """Sample t where 0 is fully masked and 1 is clean; early samples favor clean t."""
        warmup_frac = min(1.0, epoch / max(1, total_epochs // 4))
        base = torch.rand((batch_size,), device=device)
        if warmup_frac >= 1.0:
            return base
        power = 2.0 + 3.0 * (1.0 - warmup_frac)
        return 1.0 - (1.0 - base).pow(power)

    def _loss_for_batch(
        self,
        model: CTMCDenoiser,
        clean: torch.Tensor,
        encoder: AMPCanvasEncoder,
        epoch: int = 0,
        total_epochs: int = 1,
        validation: bool = False,
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
        }
        if not torch.any(supervised):
            return None, stats

        logits = model(x_t, t)
        masked_logits = logits[supervised]
        masked_target = clean[supervised]
        loss = F.cross_entropy(masked_logits, masked_target, label_smoothing=0.05)

        pred = masked_logits.argmax(dim=-1)
        aa = masked_target < 20
        stats["aa_total"] = float(aa.sum().item())
        stats["aa_correct"] = float((pred[aa] == masked_target[aa]).sum().item()) if aa.any() else 0.0
        stats["pad_targets"] = float(masked_target.eq(encoder.pad_idx).sum().item())
        stats["eos_targets"] = float(masked_target.eq(encoder.eos_idx).sum().item())

        true_lengths = clean.ne(encoder.pad_idx).sum(dim=1) - 1
        length_target = (true_lengths - self.config.min_length).clamp(0, self.config.max_length - self.config.min_length)
        length_logits = model.predict_length_logits(x_t, t)
        length_loss = F.cross_entropy(length_logits, length_target)
        total_loss = loss + 0.1 * length_loss

        length_pred = length_logits.argmax(dim=-1)
        length_acc = (length_pred == length_target).float().mean()
        stats["length_loss"] = float(length_loss.item())
        stats["length_acc"] = float(length_acc.item())

        return total_loss, stats

    @staticmethod
    def _accumulate(total: dict[str, float], stats: dict[str, float]) -> None:
        for key, value in stats.items():
            total[key] = total.get(key, 0.0) + value

    def run(self) -> None:
        """Execute end-to-end training and checkpointing."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
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

        training_sequences = self.repo.read_sequences(self.config.training_fasta_path)
        background_sequences = self.repo.read_sequences(self.config.background_fasta_path)
        if not training_sequences:
            raise RuntimeError("Training FASTA is empty")
        if not background_sequences:
            raise RuntimeError("Background FASTA is empty")

        encoder = AMPCanvasEncoder(max_length=self.config.max_length, vocab=self.config.vocab)
        logger.info(
            "Tokenizer vocab_size=%d eos=%d mask=%d pad=%d",
            encoder.vocab_size,
            encoder.eos_idx,
            encoder.mask_idx,
            encoder.pad_idx,
        )

        if self.config.debug_overfit_samples is not None:
            rng = np.random.default_rng(self.config.seed)
            n = min(self.config.debug_overfit_samples, len(training_sequences))
            train_sequences = training_sequences[:n]
            val_sequences = training_sequences[:n]
            logger.warning("OVERFIT DEBUG MODE: train and val use the same %d sequences", n)
        else:
            split_path = self.repo.resolve(self.config.cluster_split_manifest_path)
            if not split_path.exists():
                raise FileNotFoundError(f"Production CTMC training requires a cluster-level split manifest: {split_path}")
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
                raise ValueError(f"Training FASTA has {len(missing)} sequences absent from split manifest")
            train_sequences = [s for s in training_sequences if sequence_split[s][1] == "train"]
            val_sequences = [s for s in training_sequences if sequence_split[s][1] == "val"]
            if not train_sequences or not val_sequences:
                raise ValueError("Cluster split manifest requires nonempty train and val splits")

        device = resolve_device(self.config)
        log_device(device)
        pin_memory = device.type == "cuda"

        train_dataset = AMPDataset(train_sequences, encoder)
        val_dataset = AMPDataset(val_sequences, encoder)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, pin_memory=pin_memory)

        model = CTMCDenoiser(
            config=self.config,
            vocab_size=encoder.vocab_size,
            pad_idx=encoder.pad_idx,
        ).to(device)
        transfer_metadata = {"mode": self.config.ctmc_pretraining_mode}
        if "distill" in self.config.ctmc_pretraining_mode:
            raise RuntimeError("Teacher distillation is selected but teacher-cache distillation is not connected to this trainer")
        if self.config.ctmc_pretraining_mode == "embedding_init":
            try:
                from transformers import AutoTokenizer, EsmModel
            except ImportError as exc:
                raise RuntimeError("Install the training extra to initialize from ESM embeddings") from exc
            source = self.config.teacher_model_name
            tokenizer = AutoTokenizer.from_pretrained(source, revision=self.config.teacher_revision)
            if self.config.teacher_checkpoint_dir is not None:
                teacher_dir = self.repo.resolve(self.config.teacher_checkpoint_dir)
                from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor
                metadata = json.loads((teacher_dir / "model_metadata.json").read_text(encoding="utf-8"))
                teacher_model, _ = ESMMultiTaskPredictor.from_pretrained(
                    metadata["model_name"], metadata.get("revision"), len(metadata["strain_ids"])
                )
                teacher_model.load_state_dict(torch.load(teacher_dir / "model.pt", map_location="cpu", weights_only=True))
                teacher = teacher_model.backbone
                source_revision = metadata.get("revision") or metadata["model_name"]
            else:
                teacher = EsmModel.from_pretrained(source, revision=self.config.teacher_revision)
                source_revision = self.config.teacher_revision or source
            residue_ids = {aa: int(tokenizer.convert_tokens_to_ids(aa)) for aa in self.config.vocab}
            transfer_metadata.update(initialize_amino_acid_embeddings(
                model, teacher.embeddings.word_embeddings.weight, residue_ids, encoder,
                source_revision=source_revision,
            ))
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
            if resume.get("checkpoint_version") != 2 or resume.get("canvas_semantics") != "residues_plus_eos":
                raise ValueError("Resume checkpoint has incompatible canvas semantics/version")
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
            for batch in tqdm(train_loader, desc=f"train-epoch-{epoch+1}", leave=False):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, stats = self._loss_for_batch(model, batch, encoder, epoch, self.config.n_epochs)
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
                    batch = batch.to(device)
                    loss, stats = self._loss_for_batch(model, batch, encoder, validation=True)
                    self._accumulate(val_stats, stats)
                    if loss is not None:
                        val_loss_total += float(loss.item())
                        val_batches += 1

            train_loss = train_loss_total / max(train_batches, 1)
            val_loss = val_loss_total / max(val_batches, 1)
            masked_frac = train_stats.get("masked_tokens", 0.0) / max(train_stats.get("valid_tokens", 0.0), 1.0)
            aa_acc = train_stats.get("aa_correct", 0.0) / max(train_stats.get("aa_total", 0.0), 1.0)
            pad_tgt = train_stats.get("pad_targets", 0.0) / max(train_stats.get("supervised_tokens", 0.0), 1.0)
            eos_tgt = train_stats.get("eos_targets", 0.0) / max(train_stats.get("supervised_tokens", 0.0), 1.0)
            length_loss = train_stats.get("length_loss", 0.0) / max(train_batches, 1)
            length_acc = train_stats.get("length_acc", 0.0) / max(train_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})
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
            torch.save({
                "checkpoint_version": 2, "canvas_semantics": "residues_plus_eos",
                "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val_loss": best_val_loss, "best_state": best_state,
                "patience": patience, "history": history,
            }, tmp_progress)
            tmp_progress.replace(progress_path)

            if val_loss < best_val_loss - self.config.early_stopping_threshold:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
            seeds=[self.config.seed + i for i in range(self.config.discriminator_ensemble_size)],
        )
        discriminator.fit(train_sequences, background_sequences)

        checkpoint_dir = self.repo.resolve(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save(checkpoint_dir / "model.pt")
        self.config.to_json_file(checkpoint_dir / "config.json")
        self.repo.write_json_atomic(checkpoint_dir / "tokenizer.json", encoder.to_metadata())
        self.repo.write_json_atomic(
            checkpoint_dir / "training_stats.json",
            {
                "checkpoint_version": 2,
                "canvas_semantics": "residues_plus_eos",
                "best_val_loss": best_val_loss,
                "history": history,
                "feature_stats": asdict(feature_stats),
            },
        )
        self.repo.write_json_atomic(checkpoint_dir / "pretrained_transfer.json", transfer_metadata)
        write_manifest(checkpoint_dir / "dataset_manifest.json", {
            "training_fasta": str(self.repo.resolve(self.config.training_fasta_path)),
            "training_fasta_sha256": sha256_file(self.repo.resolve(self.config.training_fasta_path)),
            "background_fasta": str(self.repo.resolve(self.config.background_fasta_path)),
            "background_fasta_sha256": sha256_file(self.repo.resolve(self.config.background_fasta_path)),
            "cluster_split_manifest": str(self.repo.resolve(self.config.cluster_split_manifest_path)) if self.config.cluster_split_manifest_path.exists() else None,
            "cluster_split_sha256": sha256_file(self.repo.resolve(self.config.cluster_split_manifest_path)) if self.config.cluster_split_manifest_path.exists() else None,
            "train_count": len(train_sequences), "validation_count": len(val_sequences),
            "config": self.config.model_dump(mode="json"), "checkpoint_version": 2,
            "canvas_semantics": "residues_plus_eos",
        })
        discriminator.save(checkpoint_dir / "discriminator.json")
        tracker.log({"val/best_loss": best_val_loss})
        tracker.finish()
        logger.info("Checkpoint written to %s", checkpoint_dir)
