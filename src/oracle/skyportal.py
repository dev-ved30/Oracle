"""SkyPortal analysis-service adapter: the science half of an ORACLE analysis job.

``analyze()`` takes detections already parsed from a SkyPortal photometry payload
(mjd, fid, mag, magerr), the source redshift if SkyPortal has one, and the service
parameters, and returns the dictionary a SkyPortal analysis callback carries:
results, annotations, and an ml classification mapped onto SkyPortal's Sitewide
Taxonomy. The thin bridge in skyportal/osg-skyportal-plugin only parses the payload
and calls this, so the light-curve preprocessing is versioned here with the model.

ORACLE-2-lite is light-curve only (a GRU over the light curve); redshift and coords
are accepted for interface parity with the multimodal models but are unused here.
The light-curve tensor construction matches the model's training / real-time BOOM
deployment: days since first detection, magpsf, sigmapsf, filter mean-wavelength,
and a detection photflag of 1.

Parameters (all optional): model ("BTSv2-lite"), weights (path to a state dict;
else $ORACLE_MODEL_WEIGHTS, else the newest bundled run for the model).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

# ORACLE-2 BTS leaf classes -> the nearest label in SkyPortal's Sitewide Taxonomy.
# Rows carry ml=True and an ORACLE origin so they render as their own set, apart
# from other classifiers and from human labels.
SKYPORTAL_TAXONOMY = "Sitewide Taxonomy"
SKYPORTAL_ORIGIN = "ORACLE"
ORACLE_TO_TAXONOMY = {
    "SN-Ia": "Ia",
    "SN-II": "Type II",
    "SN-Ib/c": "Ib/c",
    "SLSN": "Ic-SLSN",
    "AGN": "AGN",
    "CV": "Cataclysmic",
    "Varstar": "Stellar variable",
}
DEFAULTS = {"model": "BTSv2-lite"}

_MODEL = None
_MODEL_KEY = None


def _default_weights(model_choice: str) -> str:
    """Newest bundled run for the model. models/ sits at the repo root, one level
    above the installed package (this file is src/oracle/skyportal.py)."""
    root = Path(__file__).resolve().parents[2]
    hits = sorted(glob.glob(str(root / "models" / model_choice / "*" / "best_model_f1.pth")))
    if not hits:
        raise FileNotFoundError(
            f"no {model_choice} weights under {root / 'models'}; pass params['weights'] "
            "or set $ORACLE_MODEL_WEIGHTS"
        )
    return hits[-1]


def _load_model(model_choice: str, weights: str):
    """Cache one loaded model per (choice, weights) for reuse across a grouped job."""
    global _MODEL, _MODEL_KEY
    key = (model_choice, weights)
    if _MODEL is None or _MODEL_KEY != key:
        import torch

        from oracle.presets import get_model

        m = get_model(model_choice)
        m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        m.eval()
        _MODEL, _MODEL_KEY = m, key
    return _MODEL


def _build_batch(rows):
    """(mjd, fid, mag, magerr) detections -> the lite model's input batch. Five
    time-series features per point: days since first detection, magpsf, sigmapsf,
    filter mean-wavelength, and a photflag left at 1 (all points are detections)."""
    import torch

    from oracle.custom_datasets.BTS import ZTF_fid_to_wavelengths

    rows = sorted(rows)
    n = len(rows)
    t0 = rows[0][0]
    ts = torch.ones((1, n, 5))
    for i, (mjd, fid, mag, err) in enumerate(rows):
        ts[0, i, 0] = mjd - t0
        ts[0, i, 1] = mag
        ts[0, i, 2] = err
        ts[0, i, 3] = ZTF_fid_to_wavelengths[int(fid)]
    return {"ts": ts, "length": torch.tensor([n])}


def classify(rows, model_choice: str, weights: str) -> dict:
    import torch

    model = _load_model(model_choice, weights)
    with torch.no_grad():
        df = model.predict_class_probabilities_df(_build_batch(rows))
    leaves = model.taxonomy.get_leaf_nodes()
    probs = {c: round(float(df[c].iloc[0]), 4) for c in leaves if c in df.columns}
    if not probs:
        raise ValueError("model returned no leaf-class probabilities")
    predicted = max(probs, key=probs.get)
    return {"predicted": predicted, "probabilities": probs}


def annotations_for(cls: dict) -> dict:
    """Flat annotation dict: the probability vector plus the top class."""
    out = {f"oracle_p_{k}": v for k, v in cls["probabilities"].items()}
    out["oracle_class"] = cls["predicted"]
    out["oracle_p_max"] = max(cls["probabilities"].values())
    return out


def skyportal_annotations(cls: dict) -> list:
    """The webhook form: a list of {origin, data} (a flat dict is dropped by SkyPortal)."""
    return [{"origin": SKYPORTAL_ORIGIN, "data": annotations_for(cls)}]


def skyportal_classifications(cls: dict) -> list:
    """One ml classification of the predicted leaf on the Sitewide Taxonomy."""
    label = ORACLE_TO_TAXONOMY.get(cls["predicted"])
    if not label:
        return []
    return [
        {
            "taxonomy": SKYPORTAL_TAXONOMY,
            "classification": label,
            "probability": cls["probabilities"][cls["predicted"]],
            "ml": True,
            "origin": SKYPORTAL_ORIGIN,
        }
    ]


def analyze(rows, redshift=None, params: dict | None = None, resource_id: str = "obj",
            work_dir: str = ".") -> dict:
    """rows: (mjd, fid, mag, magerr) ZTF detections; redshift: unused by the lite model."""
    params = {**DEFAULTS, **(params or {})}
    rows = list(rows)
    if len(rows) < 1:
        raise ValueError("no detections for ORACLE")
    model_choice = str(params["model"])
    weights = (
        params.get("weights")
        or os.environ.get("ORACLE_MODEL_WEIGHTS")
        or _default_weights(model_choice)
    )
    cls = classify(rows, model_choice, weights)
    top = cls["probabilities"][cls["predicted"]]
    message = f"ORACLE-2 ({model_choice}): {cls['predicted']} (p={top:.2f}), {len(rows)} detections"
    return {
        "status": "success",
        "message": message,
        "results": {
            "resource_id": resource_id,
            "n_detections": len(rows),
            "redshift_used": redshift,
            "model": model_choice,
            "classification": cls,
        },
        "annotations": skyportal_annotations(cls),
        "classifications": skyportal_classifications(cls),
        "annotations_flat": annotations_for(cls),
        "plot_files": [],
    }
