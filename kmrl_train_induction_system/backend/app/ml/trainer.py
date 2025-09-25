# backend/app/ml/trainer.py
from __future__ import annotations

import io
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from app.utils.cloud_database import cloud_db_manager


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3


class RiskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_col: str):
        self.X = df[feature_cols].fillna(0.0).astype(np.float32).values
        self.y = df[label_col].astype(np.float32).values

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class RiskModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


async def load_training_frame() -> Tuple[pd.DataFrame, List[str], str]:
    """Load historical features and label from MongoDB and return (df, features, hash)."""
    # Example collections: telemetry_features, fitness_windows, job_cards, mileage_rollups, labels
    tele = await (await cloud_db_manager.get_collection("telemetry_features")).find({}).to_list(None)
    miles = await (await cloud_db_manager.get_collection("mileage_rollups")).find({}).to_list(None)
    labels = await (await cloud_db_manager.get_collection("withdrawal_labels")).find({}).to_list(None)

    df_tele = pd.DataFrame(tele)
    df_miles = pd.DataFrame(miles)
    df_lbl = pd.DataFrame(labels)
    # Minimal join by trainset_id and date proximity (placeholder)
    df = df_tele.merge(df_miles, on=["trainset_id"], how="left").merge(df_lbl, on=["trainset_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    feature_cols = [
        c for c in df.columns if c not in {"_id", "trainset_id", "label"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df[feature_cols].fillna(0)).values).hexdigest()[:16]
    return df, feature_cols, data_hash


async def train_and_register(config: TrainConfig | None = None) -> Dict[str, Any]:
    cfg = config or TrainConfig()
    df, feature_cols, data_hash = await load_training_frame()
    if len(df) < 100:
        return {"status": "insufficient_data", "rows": len(df)}

    # Save dataset snapshot (parquet) for reproducibility
    snap_col = await cloud_db_manager.get_collection("model_datasets")
    try:
        buf_ds = io.BytesIO()
        df.to_parquet(buf_ds, index=False)
        await snap_col.insert_one({
            "data_hash": data_hash,
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "feature_cols": feature_cols,
            "rows": int(len(df)),
            "blob": buf_ds.getvalue(),
        })
    except Exception:
        pass

    # Split
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    val_df = df[~msk]

    train_ds = RiskDataset(train_df, feature_cols, "label")
    val_ds = RiskDataset(val_df, feature_cols, "label")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Model
    model = RiskModel(in_dim=len(feature_cols), hidden_dim=cfg.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCELoss()

    def _eval(loader: DataLoader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for X, y in loader:
                p = model(X).squeeze(1)
                losses.append(bce(p, y).item())
        return float(np.mean(losses)) if losses else 0.0

    for _ in range(cfg.epochs):
        model.train()
        for X, y in train_loader:
            p = model(X).squeeze(1)
            loss = bce(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    val_loss = _eval(val_loader)

    # Save TorchScript for fast inference
    example = torch.randn(1, len(feature_cols))
    scripted = torch.jit.trace(model, example)
    buf = io.BytesIO()
    torch.jit.save(scripted, buf)
    buf.seek(0)

    # Register model in MongoDB
    feature_means = train_df[feature_cols].fillna(0.0).mean().to_dict()
    meta = {
        "version": hashlib.sha1(buf.getvalue()).hexdigest()[:12],
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "data_hash": data_hash,
        "val_loss": float(val_loss),
        "features": feature_cols,
        "feature_means": feature_means,
        "framework": "torchscript",
    }
    col = await cloud_db_manager.get_collection("models")
    await col.insert_one({"meta": meta, "blob": buf.getvalue()})
    return {"status": "ok", "meta": meta}


