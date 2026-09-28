"""Cluster-split supervised training for the multitask ESM property predictor."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor


class _Rows(Dataset):
    def __init__(self, rows, tokenizer, strain_ids, max_length):
        self.rows, self.tokenizer, self.strain_ids, self.max_length = rows, tokenizer, strain_ids, max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(row["sequence"], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        result = {key: value.squeeze(0) for key, value in encoded.items()}
        result["strain_id"] = torch.tensor(self.strain_ids.get(row.get("strain") or "", len(self.strain_ids)))
        for task in ("amp", "mic", "hemolysis", "hc50"):
            value = row.get({"amp":"is_amp", "mic":"mic_uM", "hemolysis":"is_hemolytic", "hc50":"hc50_uM"}[task])
            numeric = float(value or 0.0)
            if task in ("mic", "hc50") and numeric > 0:
                import math
                numeric = math.log2(numeric)
            result[task] = torch.tensor(numeric, dtype=torch.float32)
            result[f"{task}_mask"] = torch.tensor(value is not None and value != "", dtype=torch.bool)
            if task in ("mic", "hc50"):
                censor = row.get(f"{task}_censor") or "none"
                result[f"{task}_censor"] = torch.tensor({"left":-1, "none":0, "right":1, "interval":2}.get(censor, 0))
        return result


class PropertyTrainingPipeline:
    """Train against an existing cluster-assigned Parquet table (no random row split)."""
    def __init__(self, data_path: Path, output_dir: Path, model_name: str, revision: str | None,
                 epochs: int = 10, batch_size: int = 16, max_length: int = 64, seed: int = 42,
                 unfreeze_last_n_layers: int = 6, gradient_accumulation_steps: int = 1):
        self.data_path, self.output_dir, self.model_name, self.revision = data_path, output_dir, model_name, revision
        self.epochs, self.batch_size, self.max_length, self.seed = epochs, batch_size, max_length, seed
        self.unfreeze_last_n_layers, self.gradient_accumulation_steps = unfreeze_last_n_layers, gradient_accumulation_steps

    def run(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to load the prepared Parquet dataset") from exc
        table = pq.read_table(self.data_path).to_pylist()
        if not table or not all(row.get("split") for row in table):
            raise ValueError("Property training requires prepared train/val/test cluster assignments")
        train_rows = [r for r in table if r["split"] == "train"]
        val_rows = [r for r in table if r["split"] == "val"]
        if not train_rows or not val_rows:
            raise ValueError("Training and validation split must both be nonempty")
        strain_ids = {name: i for i, name in enumerate(sorted({r.get("strain") for r in train_rows if r.get("strain")}))}
        model, tokenizer = ESMMultiTaskPredictor.from_pretrained(self.model_name, self.revision, len(strain_ids))
        unknown = len(strain_ids)
        train_ds = _Rows(train_rows, tokenizer, strain_ids, self.max_length)
        val_ds = _Rows(val_rows, tokenizer, strain_ids, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for parameter in model.backbone.parameters(): parameter.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        best = float("inf")
        best_state = None
        for epoch in range(self.epochs):
            if epoch == 1:
                layers = getattr(model.backbone.encoder, "layer", [])
                for layer in list(layers)[-self.unfreeze_last_n_layers:]:
                    for parameter in layer.parameters(): parameter.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {"params": [p for n,p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 2e-4},
                    {"params": [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 2e-5},
                ])
            model.train()
            for batch_i, batch in enumerate(train_loader):
                batch = {k:v.to(device) for k,v in batch.items()}
                out = model(batch["input_ids"], batch["attention_mask"], batch["strain_id"])
                loss = model.multitask_loss(out, batch)["total"] / self.gradient_accumulation_steps
                loss.backward()
                if (batch_i + 1) % self.gradient_accumulation_steps == 0 or batch_i + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            model.eval(); total = 0.0; count = 0
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    out = model(batch["input_ids"], batch["attention_mask"], batch["strain_id"])
                    total += float(model.multitask_loss(out, batch)["total"]); count += 1
            val_loss = total / max(count,1)
            if val_loss < best:
                best = val_loss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best_state is None: raise RuntimeError("No property-model checkpoint was produced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
        torch.save(best_state, self.output_dir / "model.pt")
        tokenizer.save_pretrained(self.output_dir)
        (self.output_dir / "model_metadata.json").write_text(json.dumps({
            "model_name": self.model_name, "revision": self.revision, "strain_ids": strain_ids,
            "unknown_strain_id": unknown, "best_val_loss": best, "train_count": len(train_rows),
            "val_count": len(val_rows), "test_count": sum(r["split"] == "test" for r in table),
            "split_source": str(self.data_path), "seed": self.seed,
        }, indent=2) + "\n", encoding="utf-8")
