import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import base64
import gzip
import io
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_default_device("cpu")

import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

logger = logging.getLogger(__name__)


def _coerce(value, default, ztf_id, field, source):
    """Return value if it's a real number, else default — and log why."""
    if value is None:
        logger.warning("[%s] %s.%s is null in %s, using flag_value", ztf_id, source, field, source)
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            logger.warning("[%s] %s.%s is NaN in %s, using flag_value", ztf_id, source, field, source)
            return default
    except (TypeError, ValueError):
        pass
    return value

from oracle.architectures import GRU_MD_MM_Improved
from oracle.custom_datasets.BTS import (
    ZTF_passband_to_wavelengths,
    ZTF_fid_to_wavelengths,
    time_dependent_feature_list,
    time_independent_feature_list,
    meta_data_feature_list,
    flag_value,
)
from oracle.taxonomies import BTS_Taxonomy

MODEL_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "best_model_f1_ztf.pth"

BAND_TO_CHANNEL = {"g": 0, "r": 1, "i": 2}

_model = None


def _get_model():
    global _model
    if _model is None:
        m = GRU_MD_MM_Improved(
            BTS_Taxonomy(),
            lc_md_model_dir=None,
            image_model_dir=None,
        )
        #m.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location="cpu"), strict=True)
        m.eval()
        _model = m
    return _model

def show_batch(batch):


    jd = batch["ts"][0, :, time_dependent_feature_list.index("jd")]
    magpsf = batch["ts"][0, :, time_dependent_feature_list.index("magpsf")]
    band = batch["ts"][0, :, time_dependent_feature_list.index("fid")]
    sigmapsf = batch['ts'][0, :, time_dependent_feature_list.index("sigmapsf")]
    

    print(jd)
    print(magpsf)
    print(sigmapsf)
    print(batch["ts"][0, :, 4])

    print(band)

    for f in np.unique(band):
        mask = band == f
        plt.errorbar(jd[mask], magpsf[mask], yerr=sigmapsf[mask], label=f"band {f}")

    plt.gca().invert_yaxis()
    plt.xlabel("jd")
    plt.ylabel("magpsf")
    plt.legend()
    plt.show()

    # plot the rgb channels of the image tensor
    image = batch["postage_stamp"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.imshow(image[i], cmap="gray")
        ax.set_title(f"Channel {i}")
        ax.axis("off")
    plt.show()

def load_cutout(field):
    """Decode a ZTF cutout into a normalized 63x63 torch tensor.

    Accepts MongoDB form ``{"$binary": {"base64": "..."}}``, Avro
    ``{"stampData": <bytes>}``, raw gzipped FITS bytes, or a base64 string.
    Returns None if the input is empty or malformed.
    """
    if field is None:
        return None
    if isinstance(field, dict):
        if "$binary" in field:
            raw_bytes = base64.b64decode(field["$binary"]["base64"])
        elif "stampData" in field:
            sd = field["stampData"]
            raw_bytes = base64.b64decode(sd) if isinstance(sd, str) else bytes(sd)
        else:
            return None
    elif isinstance(field, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(field)
    elif isinstance(field, str):
        raw_bytes = base64.b64decode(field)
    else:
        return None

    raw = gzip.decompress(raw_bytes)
    with fits.open(io.BytesIO(raw)) as hdul:
        image = hdul[0].data.astype(float)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(image)
    if norm != 0:
        image = image / norm
    return torch.from_numpy(image).float()


def get_taxonomy():
    return _get_model().taxonomy


def run_oracle(ztf_id, prv_candidates, candidate, cross_matches, cutouts=None,
               max_history_days=180):
    """Run Oracle BTSv2-pro (omni) classification on a ZTF alert.

    Args:
        ztf_id: ZTF objectId string.
        prv_candidates: list of prior detection dicts (jd, band, magpsf, sigmapsf, ra, dec, ...).
        candidate: current candidate dict (sgscore1, drb, ndethist, ...).
        cross_matches: dict of catalog cross-matches (expects 'AllWISE').
        cutouts: dict containing 'cutoutTemplate' (the reference image). The
            template is placed in the channel matching the most recent
            detection's band (g=0, r=1, i=2). May be None.
        max_history_days: keep only photometry within this many days of the
            most recent point. Set to None to disable.

    Returns:
        (class_scores_df, class_scores) tuple, or None if input is empty.
    """
    if not prv_candidates:
        return None

    prv_cand = pd.DataFrame(prv_candidates)
    prv_cand.sort_values("jd", inplace=True)

    print(prv_cand['magpsf'])

    if "programid" in prv_cand.columns:
        before = len(prv_cand)
        #prv_cand = prv_cand[prv_cand["programid"].isin([1, 2])].reset_index(drop=True)
        dropped = before - len(prv_cand)
        if dropped:
            logger.info("[%s] dropped %d/%d photometry rows with programid not in {1,2}",
                        ztf_id, dropped, before)
        if prv_cand.empty:
            logger.warning("[%s] no public photometry remains after programid filter", ztf_id)
            return None
    else:
        logger.warning("[%s] photometry missing 'programid' column, no filter applied", ztf_id)

    if max_history_days is not None and "jd" in prv_cand.columns and not prv_cand.empty:
        cutoff = prv_cand["jd"].max() - max_history_days
        before = len(prv_cand)
        prv_cand = prv_cand[prv_cand["jd"] >= cutoff].reset_index(drop=True)
        dropped = before - len(prv_cand)
        if dropped:
            logger.info("[%s] dropped %d/%d photometry rows older than %d days",
                        ztf_id, dropped, before, max_history_days)
        if prv_cand.empty:
            logger.warning("[%s] no photometry remains after %d-day window filter",
                           ztf_id, max_history_days)
            return None

    #final_alert_band = prv_cand["band"].values[-1] if "band" in prv_cand.columns else None
    final_alert_band = 'i'

    for col in ("jd", "magpsf", "sigmapsf", "ra", "dec", "band"):
        if col not in prv_cand.columns:
            logger.warning("[%s] prv_candidates missing column '%s'", ztf_id, col)
            continue
        n_null = prv_cand[col].isna().sum()
        if n_null:
            logger.warning("[%s] prv_candidates['%s'] has %d/%d null values",
                           ztf_id, col, int(n_null), len(prv_cand))

    prv_cand["jd"] = prv_cand["jd"] - prv_cand["jd"].min()
    unmapped = set(prv_cand["band"].dropna().unique()) - set(ZTF_passband_to_wavelengths)
    if unmapped:
        logger.warning("[%s] unknown bands in prv_candidates: %s", ztf_id, sorted(unmapped))
    prv_cand["band"] = prv_cand["band"].map(ZTF_passband_to_wavelengths)

    coords = SkyCoord(
        ra=prv_cand["ra"].to_numpy() * u.deg,
        dec=prv_cand["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    prv_cand["l"] = coords.galactic.l
    prv_cand["b"] = coords.galactic.b

    cm = cross_matches or {}
    if "AllWISE" not in cm:
        logger.warning("[%s] cross_matches missing 'AllWISE', WISE features will use flag_value", ztf_id)
        wise0 = {}
    else:
        wise = cm["AllWISE"] or [{}]
        wise0 = wise[0] if wise else {}
    w1 = _coerce(wise0.get("w1mpro", flag_value), flag_value, ztf_id, "w1mpro", "AllWISE")
    w2 = _coerce(wise0.get("w2mpro", flag_value), flag_value, ztf_id, "w2mpro", "AllWISE")
    w3 = _coerce(wise0.get("w3mpro", flag_value), flag_value, ztf_id, "w3mpro", "AllWISE")
    w4 = _coerce(wise0.get("w4mpro", flag_value), flag_value, ztf_id, "w4mpro", "AllWISE")
    prv_cand["W1mag"] = w1
    prv_cand["W2mag"] = w2
    prv_cand["W3mag"] = w3
    prv_cand["W4mag"] = w4
    prv_cand["W1_minus_W3"] = w1 - w3 if isinstance(w1, (int, float)) and isinstance(w3, (int, float)) else flag_value
    prv_cand["W2_minus_W3"] = w2 - w3 if isinstance(w2, (int, float)) and isinstance(w3, (int, float)) else flag_value

    for key in [
        "sgscore1", "sgscore2", "distpsnr1", "distpsnr2",
        "ndethist", "nmtchps", "drb", "ncovhist",
        "sgmag1", "srmag1", "simag1", "szmag1",
        "sgmag2", "srmag2", "simag2", "szmag2",
    ]:
        if key not in candidate:
            logger.warning("[%s] candidate missing '%s', using flag_value", ztf_id, key)
        prv_cand[key] = _coerce(candidate.get(key, flag_value), flag_value, ztf_id, key, "candidate")

    ts_tensor = torch.zeros((1, len(prv_cand), len(time_dependent_feature_list) + 1))
    for i, col in enumerate(time_dependent_feature_list):
        if col not in prv_cand.columns:
            logger.warning("[%s] time-dependent feature '%s' missing from prv_candidates", ztf_id, col)
            ts_tensor[0, :, i] = flag_value
            continue
        vals = prv_cand[col].values
        n_nan = int(pd.isna(vals).sum())
        if n_nan:
            logger.warning("[%s] time-dependent feature '%s' has %d/%d NaN values",
                           ztf_id, col, n_nan, len(vals))
        if col == "fid":
            vals = [ZTF_fid_to_wavelengths[v] for v in vals]
        ts_tensor[0, :, i] = torch.tensor(vals)

    static_tensor = torch.zeros((1, len(time_independent_feature_list)))
    for i, col in enumerate(time_independent_feature_list):
        if col not in prv_cand.columns:
            logger.warning("[%s] time-independent feature '%s' missing", ztf_id, col)
            static_tensor[0, i] = flag_value
            continue
        val = prv_cand[col].values[-1]
        if pd.isna(val):
            logger.warning("[%s] time-independent feature '%s' is NaN at last row", ztf_id, col)
        static_tensor[0, i] = torch.tensor(val)

    meta_data_tensor = torch.zeros((1, len(meta_data_feature_list)))
    for i, col in enumerate(meta_data_feature_list):
        if col not in prv_cand.columns:
            logger.warning("[%s] meta-data feature '%s' missing", ztf_id, col)
            meta_data_tensor[0, i] = flag_value
            continue
        val = prv_cand[col].values[-1]
        if pd.isna(val):
            logger.warning("[%s] meta-data feature '%s' is NaN at last row", ztf_id, col)
        meta_data_tensor[0, i] = torch.tensor(val)
        print(f"{col}: {val}")

    length = torch.tensor([len(prv_cand)])
    static_tensor = torch.cat((static_tensor, meta_data_tensor), dim=1)

    image_tensor = torch.zeros((1, 3, 63, 63))
    template_field = (cutouts or {}).get("cutoutTemplate")
    if template_field is None:
        logger.warning("[%s] cutoutTemplate missing, postage_stamp will be all zeros", ztf_id)
    else:
        try:
            template = load_cutout(template_field)
        except Exception:
            logger.exception("[%s] failed to decode cutoutTemplate", ztf_id)
            template = None
        if template is None:
            logger.warning("[%s] cutoutTemplate decode returned None", ztf_id)
        elif template.shape != (63, 63):
            logger.warning("[%s] cutoutTemplate has unexpected shape %s, expected (63,63)",
                           ztf_id, tuple(template.shape))
        else:
            ch = BAND_TO_CHANNEL.get(final_alert_band)
            if ch is None:
                logger.warning("[%s] final detection band %r has no channel mapping",
                               ztf_id, final_alert_band)
            else:
                image_tensor[0, ch, :, :] = template

    batch = {
        "ts": ts_tensor,
        "static": static_tensor,
        "length": length,
        "postage_stamp": image_tensor,
    }

    show_batch(batch)
    print(final_alert_band)
    print(ztf_id)

    model = _get_model()
    with torch.no_grad():
        class_scores = model.predict_class_probabilities(batch)[0]
        class_scores_df = model.predict_conditional_probabilities_df(batch)

    return class_scores_df, class_scores

if __name__ == "__main__":

    with open("alerts_aux_long.json", "r") as f:
        alert_aux = json.load(f)

    with open("test.json", "r") as f:
        images = json.load(f)

    
    run_oracle(
        ztf_id=alert_aux['_id'],
        prv_candidates=alert_aux['prv_candidates'],
        candidate=alert_aux['prv_candidates'][-1],
        cross_matches=alert_aux['cross_matches'],
        cutouts=images,
        max_history_days=15,
    )