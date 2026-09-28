"""Classify one ZTF source from Babamul, an alert JSON file, or a detection CSV."""

import argparse
import base64
import binascii
import csv
import gzip
import io
import json
import math
import os
import re
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

from oracle.custom_datasets.BTS import (
    ZTF_fid_to_wavelengths,
    flag_value,
    meta_data_feature_list,
    time_dependent_feature_list,
    time_independent_feature_list,
)
from oracle.presets import get_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = {
    "BTSv2-pro": REPO_ROOT / "models/BTSv2-pro/morning-feather-572/best_model_f1.pth",
    "BTSv2": REPO_ROOT / "models/BTSv2/stilted-elevator-551/best_model_f1.pth",
    "BTSv2-lite": REPO_ROOT / "models/BTSv2-lite/fancy-elevator-567/best_model_f1.pth",
}
BAND_TO_FID = {"g": 1, "r": 2, "i": 3}
BABAMUL_BASE_URLS = {
    "local": "http://localhost:4000/babamul",
    "production": "https://babamul.caltech.edu/api/babamul",
    "backup": "https://babamul.umn.edu/api/babamul",
}


def _number(value, field, *, missing=None):
    if value is None or value == "":
        if missing is not None:
            return missing
        raise ValueError(f"Missing {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        if missing is not None:
            return missing
        raise ValueError(f"Invalid {field}: {value!r}")
    return flag_value if result == -999 and missing is not None else result


def _fid(row):
    value = row.get("fid")
    if value is None or value == "":
        value = row.get("band")
    if isinstance(value, str) and value.lower().removeprefix("ztf_") in BAND_TO_FID:
        return BAND_TO_FID[value.lower().removeprefix("ztf_")]
    try:
        numeric = float(value)
        fid = int(numeric)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ZTF filter {value!r}; expected fid 1, 2, 3 or band g, r, i") from exc
    if numeric != fid or fid not in ZTF_fid_to_wavelengths:
        raise ValueError(f"Invalid ZTF filter {value!r}; expected fid 1, 2, 3 or band g, r, i")
    return fid


def _babamul_get(endpoint):
    token = os.environ.get("BABAMUL_API_TOKEN")
    if not token:
        raise ValueError("Set BABAMUL_API_TOKEN to query a ZTF object ID from Babamul")
    environment = os.environ.get("BABAMUL_ENV", "production").lower()
    if environment not in BABAMUL_BASE_URLS:
        raise ValueError("BABAMUL_ENV must be production, backup, or local")
    url = BABAMUL_BASE_URLS[environment] + endpoint
    request = Request(url, headers={"Authorization": f"Bearer {token}",
                                    "Accept": "application/json",
                                    "User-Agent": "oracle-ztf-inference/2.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            raise ValueError("Babamul authentication failed; check BABAMUL_API_TOKEN") from exc
        if exc.code == 404:
            resource = "Reference cutout" if "/cutouts" in endpoint else "ZTF object ID"
            raise ValueError(f"{resource} was not found in Babamul") from exc
        raise ValueError(f"Babamul returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"Could not reach Babamul: {exc}") from exc
    return payload.get("data", payload)


def _parse_source(data, source_id):
    if isinstance(data, list):
        return data, {}, source_id
    if not isinstance(data, dict):
        raise ValueError("Source data must be a ZTF alert object or a list of detections")
    rows = data.get("prv_candidates", data.get("detections")) or []
    if not isinstance(rows, list):
        raise ValueError("prv_candidates/detections must be a list")
    rows = list(rows)
    candidate = data.get("candidate")
    if isinstance(candidate, dict):
        rows.append(candidate)
    return rows, data, data.get("objectId", data.get("_id", source_id))


def read_source(source):
    """Return (detections, context, source_id) from a Babamul ID, JSON, or CSV."""
    if re.fullmatch(r"ZTF\d{2}[a-z]+", str(source), re.IGNORECASE):
        object_id = str(source)
        data = _babamul_get(f"/surveys/ZTF/objects/{quote(object_id, safe='')}")
        return _parse_source(data, object_id)
    path = Path(source)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return rows, {}, path.stem
    if path.suffix.lower() != ".json":
        raise ValueError("Input must be a .json ZTF alert or a .csv detection table")
    with path.open() as stream:
        data = json.load(stream)
    return _parse_source(data, path.stem)


def make_batch(rows, context, model_choice="BTSv2"):
    """Match the BTS dataset's feature order, units, flags, and static selection."""
    detections = prepare_detections(rows)

    first_jd = detections[0][0]
    ts = np.ones((len(detections), len(time_dependent_feature_list) + 1), dtype=np.float32)
    for index, (jd, mag, err, fid, _) in enumerate(detections):
        ts[index, :4] = (jd - first_jd, mag, err, ZTF_fid_to_wavelengths[fid])

    batch = {
        "ts": torch.from_numpy(ts).unsqueeze(0),
        "length": torch.tensor([len(detections)], dtype=torch.int64),
    }
    if model_choice == "BTSv2-lite":
        return batch, len(detections)

    latest = detections[-1][4]
    supplied_static = context.get("static", {})
    if not isinstance(supplied_static, dict):
        raise ValueError("Top-level static must be an object")
    ra = latest.get("ra", supplied_static.get("ra"))
    dec = latest.get("dec", supplied_static.get("dec"))
    if ra is None or dec is None:
        raise ValueError("BTSv2 requires ra and dec in the latest detection or top-level static; use BTSv2-lite for light curves only")
    coord = SkyCoord(ra=_number(ra, "ra") * u.deg, dec=_number(dec, "dec") * u.deg)
    static_values = {"l": coord.galactic.l.deg, "b": coord.galactic.b.deg}

    # BOOM auxiliary alerts may include a nearby AllWISE source. The BTS data
    # preparation uses WISE features only for matches within 2.75 arcseconds.
    wise = (context.get("cross_matches") or {}).get("AllWISE", [])
    if isinstance(wise, list):
        matches = [m for m in wise if isinstance(m, dict) and
                   _number(m.get("distance_arcsec"), "distance_arcsec", missing=math.inf) <= 2.75]
        if matches:
            nearest = min(matches, key=lambda m: float(m["distance_arcsec"]))
            for band in (1, 2, 3, 4):
                static_values[f"W{band}mag"] = nearest.get(f"w{band}mpro")
    static_values.update(supplied_static)
    for band in (1, 2, 3, 4):
        key = f"W{band}mag"
        if key in latest and latest[key] is not None:
            static_values[key] = latest[key]
    for left, right, name in (("W1mag", "W3mag", "W1_minus_W3"),
                              ("W2mag", "W3mag", "W2_minus_W3")):
        a = _number(static_values.get(left), left, missing=flag_value)
        b = _number(static_values.get(right), right, missing=flag_value)
        static_values[name] = a - b if a != flag_value and b != flag_value else flag_value

    values = [_number(static_values.get(name), name, missing=flag_value)
              for name in time_independent_feature_list]
    values += [_number(latest.get(name), name, missing=flag_value)
               for name in meta_data_feature_list]
    batch["static"] = torch.tensor([values], dtype=torch.float32)
    return batch, len(detections)


def prepare_detections(rows):
    """Validate, deduplicate, and sort the detected ZTF observations."""
    detections = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every detection must be an object with named fields")
        # ZTF upper limits carry no measured magnitude and are not model inputs.
        if row.get("magpsf") is None or row.get("magpsf") == "":
            continue
        jd = _number(row.get("jd"), "jd")
        mag = _number(row.get("magpsf"), "magpsf")
        err = _number(row.get("sigmapsf"), "sigmapsf")
        if err <= 0:
            raise ValueError("sigmapsf must be positive for every detection")
        fid = _fid(row)
        key = (jd, fid, mag)
        if key in seen:
            continue
        seen.add(key)
        detections.append((jd, mag, err, fid, row))
    if not detections:
        raise ValueError("No detections with jd, magpsf, sigmapsf, and fid/band found")
    detections.sort(key=lambda item: item[0])
    return detections


def _template_image(value):
    """Decode a Babamul base64 or MongoDB extended JSON reference FITS stamp."""
    if isinstance(value, dict):
        if "stampData" in value:
            value = value["stampData"]
        if isinstance(value, dict) and "$binary" in value:
            value = value["$binary"]["base64"]
    if not value:
        raise ValueError("Omni requires a ZTF reference cutout (cutoutTemplate)")
    try:
        raw = base64.b64decode(value, validate=True) if isinstance(value, str) else value
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerifyWarning)
            with fits.open(io.BytesIO(raw), memmap=False) as hdul:
                image = np.asarray(hdul[0].data, dtype=np.float32)
    except (ValueError, TypeError, KeyError, OSError, binascii.Error) as exc:
        raise ValueError("Could not decode the ZTF reference cutout as a FITS image") from exc
    if image.shape != (63, 63) or not np.isfinite(image).all():
        raise ValueError(f"ZTF reference cutout must be a finite 63x63 image, got {image.shape}")
    norm = np.linalg.norm(image)
    return image / norm if norm else image


def _add_omni_image(batch, rows, context, cutout_path=None):
    candidate = context.get("candidate")
    if not isinstance(candidate, dict):
        detections = [row for row in rows if isinstance(row, dict) and row.get("magpsf") is not None]
        if not detections:
            raise ValueError("Omni needs a detection to identify the reference image filter")
        candidate = max(detections, key=lambda row: _number(row.get("jd"), "jd"))
    fid = _fid(candidate)

    if cutout_path:
        path = Path(cutout_path)
        if path.suffix.lower() == ".json":
            with path.open() as stream:
                value = json.load(stream).get("cutoutTemplate")
        else:
            value = path.read_bytes()
    else:
        value = context.get("cutoutTemplate")
        if not value and context.get("objectId") and context.get("candid"):
            cutouts = _babamul_get(f"/surveys/ZTF/cutouts?candid={quote(str(context['candid']), safe='')}")
            value = cutouts.get("cutoutTemplate")
    image = _template_image(value)
    channels = np.zeros((3, 63, 63), dtype=np.float32)
    channels[fid - 1] = image
    batch["postage_stamp"] = torch.from_numpy(channels).unsqueeze(0)


def _load_inference_model(model_choice, checkpoint):
    checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINTS[model_choice]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = get_model(model_choice)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model, checkpoint


def _probabilities_by_level(model, scores):
    taxonomy = model.taxonomy
    nodes = list(taxonomy.get_level_order_traversal())
    by_level = {}
    for depth, level_nodes in taxonomy.get_nodes_by_depth().items():
        if depth <= 0:
            continue
        probabilities = {node: scores[nodes.index(node)] for node in level_nodes}
        by_level[str(depth)] = dict(sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True))
    return by_level


def classify_source(rows, context, source_id, model_choice="BTSv2-pro", checkpoint=None, cutout=None):
    """Run a selected checkpoint on an already fetched source."""
    if cutout and model_choice != "BTSv2-pro":
        raise ValueError("--cutout is only used with the BTSv2-pro (Omni) model")
    # Some package modules set a process-wide CUDA default at import time.
    torch.set_default_device("cpu")
    batch, count = make_batch(rows, context, model_choice)
    if model_choice == "BTSv2-pro":
        _add_omni_image(batch, rows, context, cutout)
    missing_features = []
    if "static" in batch:
        feature_names = time_independent_feature_list + meta_data_feature_list
        missing_features = [name for name, value in zip(feature_names, batch["static"][0].tolist())
                            if value == flag_value]
    model, checkpoint = _load_inference_model(model_choice, checkpoint)
    with torch.inference_mode():
        scores = model.predict_class_probabilities(batch)[0].cpu().tolist()
    by_level = _probabilities_by_level(model, scores)
    return {"source_id": source_id, "detections": count, "model": model_choice,
            "checkpoint": str(checkpoint), "missing_context_features": missing_features,
            "probabilities_by_level": by_level}


def classify_rolling_source(rows, context, source_id, model_choice="BTSv2-pro", checkpoint=None):
    """Score every successive detected-observation prefix with one loaded checkpoint."""
    torch.set_default_device("cpu")
    detections = prepare_detections(rows)
    model, checkpoint = _load_inference_model(model_choice, checkpoint)
    fixed_image_embedding = None
    if model_choice == "BTSv2-pro":
        image_batch = {}
        _add_omni_image(image_batch, rows, context)
        with torch.inference_mode():
            fixed_image_embedding = model.image_spine.get_latent_space_embeddings(image_batch)

    points = []
    missing_features = []
    final_probabilities = None
    for start in range(0, len(detections), 32):
        prefix_batches = []
        for count in range(start + 1, min(start + 32, len(detections)) + 1):
            prefix_rows = [item[4] for item in detections[:count]]
            prefix_batch, _ = make_batch(prefix_rows, context, model_choice)
            prefix_batches.append(prefix_batch)

        batch = {
            "ts": torch.nn.utils.rnn.pad_sequence(
                [item["ts"][0] for item in prefix_batches], batch_first=True),
            "length": torch.tensor([item["length"].item() for item in prefix_batches], dtype=torch.int64),
        }
        if model_choice != "BTSv2-lite":
            batch["static"] = torch.cat([item["static"] for item in prefix_batches])
            feature_names = time_independent_feature_list + meta_data_feature_list
            missing_features = [name for name, value in zip(feature_names, batch["static"][-1].tolist())
                                if value == flag_value]

        with torch.inference_mode():
            if fixed_image_embedding is None:
                scores = model.predict_class_probabilities(batch)
            else:
                light_curve_embedding = model.lc_md_spine.get_latent_space_embeddings(batch)
                image_embedding = fixed_image_embedding.expand(light_curve_embedding.shape[0], -1)
                logits = model.final_out(model.mlp_head(torch.cat((light_curve_embedding, image_embedding), dim=1)))
                conditional = model.taxonomy.get_conditional_probabilities(logits)
                scores = model.taxonomy.get_class_probabilities(conditional)
        for offset, values in enumerate(scores.cpu().tolist()):
            count = start + offset + 1
            jd, _, _, fid, _ = detections[count - 1]
            final_probabilities = _probabilities_by_level(model, values)
            points.append({
                "observation": count,
                "jd": jd,
                "days": jd - detections[0][0],
                "band": {1: "g", 2: "r", 3: "i"}[fid],
                "probabilities": final_probabilities.get("2", {}),
            })

    classification = {
        "source_id": source_id, "detections": len(detections), "model": model_choice,
        "checkpoint": str(checkpoint), "missing_context_features": missing_features,
        "probabilities_by_level": final_probabilities,
    }
    note = ("Each step uses detections and per-alert metadata through that observation. "
            "Current broker cross-matches and the latest ZTF reference image are reused at every step."
            if model_choice == "BTSv2-pro" else
            "Each step uses detections and per-alert metadata through that observation; current broker cross-matches are reused."
            if model_choice == "BTSv2" else
            "Each step uses only detections available through that observation.")
    return {"classification": classification, "rolling": {"points": points, "note": note}}


def classify(path, model_choice="BTSv2-pro", checkpoint=None, cutout=None):
    """Return hierarchical scores for a single source on the CPU."""
    rows, context, source_id = read_source(path)
    return classify_source(rows, context, source_id, model_choice, checkpoint, cutout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="ZTF object ID, ZTF alert/BOOM JSON, or detection CSV")
    parser.add_argument("--model", choices=tuple(DEFAULT_CHECKPOINTS), default="BTSv2-pro")
    parser.add_argument("--checkpoint", type=Path, help="Override the model checkpoint")
    parser.add_argument("--cutout", type=Path, help="Local Omni reference cutout FITS or Babamul cutout JSON")
    parser.add_argument("--output", type=Path, help="Write the same result to a JSON file")
    args = parser.parse_args(argv)
    try:
        result = classify(args.source, args.model, args.checkpoint, args.cutout)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
