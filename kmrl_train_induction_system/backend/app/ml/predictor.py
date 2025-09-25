# backend/app/ml/predictor.py
from __future__ import annotations

from typing import Dict, Any, List
import io
import numpy as np
import pandas as pd
import torch
import shap

from app.utils.cloud_database import cloud_db_manager


async def load_latest_model() -> Dict[str, Any]:
    col = await cloud_db_manager.get_collection("models")
    doc = await col.find_one(sort=[("meta.created_at", -1)])
    if not doc:
        raise RuntimeError("No model registered")
    meta = doc.get("meta", {})
    blob = doc.get("blob")
    if isinstance(blob, bytes):
        buf = io.BytesIO(blob)
    else:
        buf = io.BytesIO(bytes(blob))
    model = torch.jit.load(buf)
    model.eval()
    return {"model": model, "meta": meta}


async def batch_predict(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not features:
        return []
    bundle = await load_latest_model()
    model = bundle["model"]
    meta = bundle["meta"]
    feature_cols = meta.get("features", [])

    df = pd.DataFrame(features)
    # Center with feature means if available
    means = meta.get("feature_means", {})
    df_centered = df.reindex(columns=feature_cols).fillna(0.0)
    for c in feature_cols:
        if c in means:
            df_centered[c] = df_centered[c] - float(means[c])
    X = df_centered.astype(np.float32).values
    with torch.no_grad():
        p = model(torch.from_numpy(X)).squeeze(1).numpy()

    # SHAP kernel approximation for top feature attributions (lightweight)
    try:
        explainer = shap.Explainer(lambda z: model(torch.from_numpy(z.astype(np.float32))).detach().numpy(), feature_names=feature_cols)
        shap_vals = explainer(X, max_evals=100)
        contrib = np.abs(shap_vals.values)
        top_idx = np.argsort(-contrib, axis=1)[:, :3]
    except Exception:
        contrib = None
        top_idx = None
    out: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        item = {
            "trainset_id": row.get("trainset_id"),
            "risk_prob": float(p[i]),
        }
        if top_idx is not None:
            item["top_features"] = [feature_cols[j] for j in top_idx[i]]
        out.append(item)
    return out


