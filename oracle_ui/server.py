"""Run the local ORACLE source explorer and classification interface."""

import argparse
import base64
import io
import math
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import torch
from astropy.table import Table
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from oracle.custom_datasets.BTS import (
    flag_value,
    meta_data_feature_list,
    time_independent_feature_list,
)
from oracle.infer_ztf import (
    DEFAULT_CHECKPOINTS,
    _babamul_get,
    _template_image,
    classify_rolling_source,
    classify_source,
    make_batch,
    prepare_detections,
    read_source,
)


UI_DIR = Path(__file__).resolve().parent
SOURCE_ID_PATTERN = re.compile(r"ZTF\d{2}[a-z]+", re.IGNORECASE)


def _preview_png(cutout):
    """Stretch a reference FITS cutout for display without changing model input."""
    image = _template_image(cutout)
    low, high = np.percentile(image, [1, 99])
    scaled = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)
    preview = Image.fromarray(np.uint8(scaled * 255), mode="L")
    preview = preview.resize((315, 315), Image.Resampling.NEAREST)
    output = io.BytesIO()
    preview.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _fetch_template(context):
    if context.get("cutoutTemplate"):
        return context["cutoutTemplate"]
    candid = context.get("candid")
    if candid is None:
        raise ValueError("Babamul did not provide a candidate ID for the reference cutout")
    cutouts = _babamul_get(f"/surveys/ZTF/cutouts?candid={candid}")
    if not cutouts.get("cutoutTemplate"):
        raise ValueError("Babamul has no reference cutout for this source")
    return cutouts["cutoutTemplate"]


def _ps_preview(ra, dec):
    """Fetch a Pan-STARRS1 g/r/i color cutout centered on the ZTF source."""
    ra, dec = float(ra), float(dec)
    if not math.isfinite(ra) or not math.isfinite(dec) or not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError("Pan-STARRS image needs valid source coordinates")
    if dec < -30:
        raise ValueError("This source is outside Pan-STARRS1 sky coverage")

    service = "https://ps1images.stsci.edu/cgi-bin/"
    filenames_url = service + "ps1filenames.py?" + urlencode({"ra": ra, "dec": dec, "filters": "gri"})
    with urlopen(filenames_url, timeout=10) as response:
        table_bytes = response.read(262145)
    if len(table_bytes) > 262144:
        raise ValueError("Pan-STARRS image listing is too large")
    table = Table.read(io.BytesIO(table_bytes), format="ascii")
    files = {str(row["filter"]): str(row["filename"]) for row in table}
    if not all(band in files for band in ("g", "r", "i")):
        raise ValueError("Pan-STARRS g/r/i images are unavailable at this position")

    cutout_url = service + "fitscut.cgi?" + urlencode({
        "ra": ra, "dec": dec, "size": 252, "format": "jpg", "output_size": 315,
        "red": files["i"], "green": files["r"], "blue": files["g"],
    })
    with urlopen(cutout_url, timeout=15) as response:
        jpeg = response.read(5000001)
    if len(jpeg) > 5000000:
        raise ValueError("Pan-STARRS image is too large")
    with Image.open(io.BytesIO(jpeg)) as preview:
        if preview.format != "JPEG":
            raise ValueError("Pan-STARRS returned an unexpected image format")
        preview.verify()
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _source_preview(rows, context, source_id):
    detections = prepare_detections(rows)
    first_jd = detections[0][0]
    latest = detections[-1][4]
    photometry = [
        {"jd": jd, "days": jd - first_jd, "mag": mag, "error": error,
         "band": {1: "g", 2: "r", 3: "i"}[fid]}
        for jd, mag, error, fid, _ in detections
    ]

    metadata = []
    metadata_error = None
    try:
        torch.set_default_device("cpu")
        batch, _ = make_batch(rows, context, "BTSv2")
        feature_names = time_independent_feature_list + meta_data_feature_list
        metadata = [
            {"name": name, "value": None if value == flag_value else float(value),
             "group": "Static" if index < len(time_independent_feature_list) else "Alert"}
            for index, (name, value) in enumerate(zip(feature_names, batch["static"][0].tolist()))
        ]
    except ValueError as exc:
        metadata_error = str(exc)

    image = None
    image_error = None
    try:
        context["cutoutTemplate"] = _fetch_template(context)
        image = _preview_png(context["cutoutTemplate"])
    except (ValueError, KeyError) as exc:
        image_error = str(exc)

    ps_image = None
    ps_image_error = None
    try:
        ps_image = _ps_preview(latest.get("ra"), latest.get("dec"))
    except (ValueError, OSError, KeyError) as exc:
        ps_image_error = str(exc)

    return {
        "source_id": source_id,
        "detections": len(detections),
        "first_jd": first_jd,
        "last_jd": detections[-1][0],
        "ra": latest.get("ra"),
        "dec": latest.get("dec"),
        "latest_band": photometry[-1]["band"],
        "photometry": photometry,
        "metadata": metadata,
        "metadata_error": metadata_error,
        "image": image,
        "image_error": image_error,
        "ps_image": ps_image,
        "ps_image_error": ps_image_error,
    }


def create_app():
    app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="/assets")

    @app.get("/")
    def index():
        return send_from_directory(UI_DIR, "index.html")

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True) or {}
        object_id = str(payload.get("object_id", "")).strip()
        model = payload.get("model")
        rolling = payload.get("rolling", False)
        if not SOURCE_ID_PATTERN.fullmatch(object_id):
            return jsonify({"error": "Enter a ZTF object ID such as ZTF18abmrfqv."}), 400
        if model not in DEFAULT_CHECKPOINTS:
            return jsonify({"error": "Choose an ORACLE-2 model."}), 400
        if not isinstance(rolling, bool):
            return jsonify({"error": "Rolling classification must be on or off."}), 400
        try:
            rows, context, source_id = read_source(object_id)
            preview = _source_preview(rows, context, source_id)
        except (ValueError, KeyError, OSError) as exc:
            message = str(exc)
            status = 404 if "not found" in message.lower() else 400
            return jsonify({"error": message}), status
        rolling_result = None
        rolling_error = None
        if rolling:
            try:
                bundle = classify_rolling_source(rows, context, source_id, model)
                return jsonify({"source": preview, "classification": bundle["classification"],
                                "rolling": bundle["rolling"], "rolling_error": None, "error": None})
            except (ValueError, KeyError, FileNotFoundError, OSError, RuntimeError) as exc:
                rolling_error = str(exc)
        try:
            result = classify_source(rows, context, source_id, model)
        except (ValueError, KeyError, FileNotFoundError, OSError, RuntimeError) as exc:
            return jsonify({"source": preview, "classification": None, "rolling": None,
                            "rolling_error": rolling_error, "error": str(exc)})
        return jsonify({"source": preview, "classification": result, "rolling": rolling_result,
                        "rolling_error": rolling_error, "error": None})

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    create_app().run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
