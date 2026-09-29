"""Cluster-split supervised training for the multitask ESM property predictor."""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor
from amp_ctmc_2027.data.conditions import panel_semantic_definition


class _Rows(Dataset):
    def __init__(self, rows, tokenizer, strain_ids, max_length):
        self.rows, self.tokenizer, self.strain_ids, self.max_length = (
            rows,
            tokenizer,
            strain_ids,
            max_length,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        result = {key: value.squeeze(0) for key, value in encoded.items()}
        result["strain_id"] = torch.tensor(
            self.strain_ids.get(row.get("strain") or "", len(self.strain_ids))
        )
        for task in ("amp", "mic", "hemolysis", "hc50"):
            value = row.get(
                {
                    "amp": "is_amp",
                    "mic": "mic_uM",
                    "hemolysis": "is_hemolytic",
                    "hc50": "hc50_uM",
                }[task]
            )
            numeric = float(value or 0.0)
            if task in ("mic", "hc50") and numeric > 0:
                import math

                numeric = math.log2(numeric)
            result[task] = torch.tensor(numeric, dtype=torch.float32)
            result[f"{task}_mask"] = torch.tensor(
                value is not None and value != "", dtype=torch.bool
            )
            if task in ("mic", "hc50"):
                censor = row.get(f"{task}_censor") or "none"
                result[f"{task}_censor"] = torch.tensor(
                    {"left": -1, "none": 0, "right": 1, "interval": 2}.get(censor, 0)
                )
                upper_field = "mic_upper_uM" if task == "mic" else "hc50_upper_uM"
                upper = row.get(upper_field)
                upper_value = float(upper) if upper not in (None, "") else numeric
                if upper_value > 0:
                    import math

                    upper_value = math.log2(upper_value)
                result[f"{task}_upper"] = torch.tensor(upper_value, dtype=torch.float32)
        return result


class PropertyTrainingPipeline:
    """Train against an existing cluster-assigned Parquet table (no random row split)."""

    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        model_name: str,
        revision: str | None,
        epochs: int = 10,
        batch_size: int = 16,
        max_length: int = 64,
        seed: int = 42,
        unfreeze_last_n_layers: int = 6,
        gradient_accumulation_steps: int = 1,
        repository_root: Path | None = None,
        ranking_strains: list[str] | None = None,
        strain_groups: dict[str, list[str]] | None = None,
        sequence_clustering_threshold: float = 0.4,
        activity_threshold_log2_mic: float = 4.0,
        activity_temperature: float = 1.0,
        broad_spectrum_aggregation: dict | None = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 2,
        mixed_precision: bool = True,
        pretrained_model_id: str | None = None,
        pretrained_revision: str | None = None,
    ):
        self.data_path, self.output_dir, self.model_name, self.revision = (
            data_path,
            output_dir,
            model_name,
            revision,
        )
        self.epochs, self.batch_size, self.max_length, self.seed = (
            epochs,
            batch_size,
            max_length,
            seed,
        )
        self.unfreeze_last_n_layers, self.gradient_accumulation_steps = (
            unfreeze_last_n_layers,
            gradient_accumulation_steps,
        )
        self.repository_root = (repository_root or Path.cwd()).resolve()
        self.ranking_strains = ranking_strains
        self.strain_groups = strain_groups or {
            "gram_positive": [],
            "gram_negative": [],
            "mdr": [],
        }
        self.sequence_clustering_threshold = float(sequence_clustering_threshold)
        self.activity_threshold_log2_mic = float(activity_threshold_log2_mic)
        self.activity_temperature = float(activity_temperature)
        self.broad_spectrum_aggregation = broad_spectrum_aggregation or {
            "method": "minimum"
        }
        self.early_stopping = bool(early_stopping)
        self.early_stopping_patience = max(1, int(early_stopping_patience))
        self.mixed_precision = bool(mixed_precision)
        self.pretrained_model_id = pretrained_model_id
        self.pretrained_revision = pretrained_revision

    def run(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to load the prepared Parquet dataset"
            ) from exc
        data_path = (
            self.data_path
            if self.data_path.is_absolute()
            else self.repository_root / self.data_path
        )
        table = pq.read_table(data_path).to_pylist()
        if not table or not all(row.get("split") for row in table):
            raise ValueError(
                "Property training requires prepared train/val/test cluster assignments"
            )
        train_rows = [r for r in table if r["split"] == "train"]
        val_rows = [r for r in table if r["split"] == "val"]
        if not train_rows or not val_rows:
            raise ValueError("Training and validation split must both be nonempty")
        strain_ids = {
            name: i
            for i, name in enumerate(
                sorted({r.get("strain") for r in train_rows if r.get("strain")})
            )
        }
        ranking_strains = self.ranking_strains
        if not ranking_strains:
            raise ValueError(
                "Configure explicit challenge-panel ranking_strains before property export"
            )
        semantic_definition = panel_semantic_definition(
            ranking_strains,
            self.strain_groups,
            self.activity_threshold_log2_mic,
            self.activity_temperature,
            self.broad_spectrum_aggregation,
        )
        missing_panel = set(ranking_strains).difference(strain_ids)
        if missing_panel:
            raise ValueError(
                f"Ranking strains are absent from the training split: {sorted(missing_panel)}"
            )
        for group, members in self.strain_groups.items():
            if set(members).difference(strain_ids):
                raise ValueError(
                    f"Strain group {group!r} contains unknown training strain IDs"
                )
        model_path = Path(self.model_name)
        model, tokenizer = ESMMultiTaskPredictor.from_pretrained(
            self.model_name,
            self.revision,
            len(strain_ids),
            local_files_only=model_path.exists(),
        )
        unknown = len(strain_ids)
        train_ds = _Rows(train_rows, tokenizer, strain_ids, self.max_length)
        val_ds = _Rows(val_rows, tokenizer, strain_ids, self.max_length)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = self.mixed_precision and device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_amp and amp_dtype == torch.float16
        )
        model.to(device)
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4
        )
        best = float("inf")
        best_state = None
        patience = 0
        for epoch in range(self.epochs):
            if epoch == 1:
                layers = getattr(model.backbone.encoder, "layer", [])
                for layer in list(layers)[-self.unfreeze_last_n_layers :]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "backbone" not in n and p.requires_grad
                            ],
                            "lr": 2e-4,
                        },
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "backbone" in n and p.requires_grad
                            ],
                            "lr": 2e-5,
                        },
                    ]
                )
            model.train()
            for batch_i, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    out = model(
                        batch["input_ids"], batch["attention_mask"], batch["strain_id"]
                    )
                loss = (
                    model.multitask_loss(out, batch)["total"]
                    / self.gradient_accumulation_steps
                )
                scaler.scale(loss).backward()
                if (
                    batch_i + 1
                ) % self.gradient_accumulation_steps == 0 or batch_i + 1 == len(
                    train_loader
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            model.eval()
            total = 0.0
            count = 0
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(
                        device_type="cuda", dtype=amp_dtype, enabled=use_amp
                    ):
                        out = model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["strain_id"],
                        )
                    total += float(model.multitask_loss(out, batch)["total"])
                    count += 1
            val_loss = total / max(count, 1)
            if val_loss < best:
                best = val_loss
                patience = 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience += 1
                if self.early_stopping and patience >= self.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("No property-model checkpoint was produced")
        output_dir = (
            self.output_dir
            if self.output_dir.is_absolute()
            else self.repository_root / self.output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
        property_checkpoint = {
            "checkpoint_version": 1,
            "architecture": {
                "hidden_size": int(model.backbone.config.hidden_size),
                "strain_count": len(strain_ids),
                "strain_dim": model.strain_embedding.embedding_dim,
            },
            "state_dict": best_state,
        }
        temporary_checkpoint = output_dir / "model.pt.tmp"
        torch.save(property_checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(output_dir / "model.pt")
        backbone_dir = output_dir / "backbone"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model.backbone.save_pretrained(backbone_dir, safe_serialization=True)
        model_revision = self.revision or getattr(
            model.backbone.config, "_commit_hash", None
        )
        if not model_revision:
            model_fingerprint = hashlib.sha256()
            for artifact in sorted(
                path for path in backbone_dir.rglob("*") if path.is_file()
            ):
                model_fingerprint.update(
                    artifact.relative_to(backbone_dir).as_posix().encode()
                )
                model_fingerprint.update(hashlib.sha256(artifact.read_bytes()).digest())
            model_revision = f"local-sha256:{model_fingerprint.hexdigest()}"
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        split_payload = [
            {
                "sequence": row.get("sequence"),
                "split": row.get("split"),
                "cluster_id": row.get("cluster_id"),
            }
            for row in sorted(
                table,
                key=lambda item: (str(item.get("sequence")), str(item.get("split"))),
            )
        ]
        split_hash = hashlib.sha256(
            json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        training_dir = self.repository_root / "data/training"
        training_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = training_dir / "training.fasta"
        if not fasta_path.is_file():
            train_sequences = list(dict.fromkeys(row["sequence"] for row in train_rows))
            fasta_temp = fasta_path.with_suffix(".fasta.tmp")
            fasta_temp.write_text(
                "".join(
                    f">train_{i}\n{sequence}\n"
                    for i, sequence in enumerate(train_sequences, 1)
                ),
                encoding="utf-8",
            )
            fasta_temp.replace(fasta_path)
        fasta_hash = hashlib.sha256(fasta_path.read_bytes()).hexdigest()
        dataset_manifest = {
            "source_parquet": str(data_path.relative_to(self.repository_root))
            if data_path.is_relative_to(self.repository_root)
            else str(data_path),
            "source_parquet_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "training_fasta": "data/training/training.fasta",
            "training_fasta_sha256": fasta_hash,
            "training_split_sha256": split_hash,
            "sequence_clustering_threshold": self.sequence_clustering_threshold,
            "train_count": len(train_rows),
            "validation_count": len(val_rows),
            "test_count": sum(row["split"] == "test" for row in table),
        }
        dataset_manifest_path = training_dir / "dataset_manifest.json"
        if not dataset_manifest_path.is_file():
            dataset_manifest_path.write_text(
                json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (output_dir / "model_metadata.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "revision": self.revision,
                    "pretrained_model_id": self.pretrained_model_id,
                    "pretrained_revision": self.pretrained_revision,
                    "model_revision": model_revision,
                    "strain_ids": strain_ids,
                    "strain_to_id": strain_ids,
                    "ranking_strains": ranking_strains,
                    "strain_groups": self.strain_groups,
                    **semantic_definition,
                    "semantic_definition": semantic_definition,
                    "unknown_strain_id": unknown,
                    "strain_dim": model.strain_embedding.embedding_dim,
                    "hidden_size": int(model.backbone.config.hidden_size),
                    "best_val_loss": best,
                    "train_count": len(train_rows),
                    "val_count": len(val_rows),
                    "test_count": sum(r["split"] == "test" for r in table),
                    "split_source": str(self.data_path),
                    "training_split_sha256": split_hash,
                    "sequence_clustering_threshold": self.sequence_clustering_threshold,
                    "task_definitions": {
                        "amp": "binary AMP classification probability",
                        "mic": "strain-conditioned log2 MIC with censor-aware Gaussian likelihood",
                        "hemolysis": "binary hemolysis probability",
                        "hc50": "log2 HC50 with censor-aware Gaussian likelihood",
                    },
                    "inference_batch_size": self.batch_size,
                    "max_tokenized_length": self.max_length,
                    "training_data_manifest": "data/training/dataset_manifest.json",
                    "seed": self.seed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
