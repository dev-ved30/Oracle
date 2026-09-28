"""Input contract tests for one-source ZTF inference."""

import io
import base64
import gzip
import json
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from oracle import infer_ztf


class TestZtfInferenceInputs(unittest.TestCase):
    def test_boom_detections_are_sorted_deduplicated_and_flagged(self):
        later = {"jd": 2460002, "magpsf": 19.1, "sigmapsf": 0.2,
                 "fid": 2, "ra": 25.0, "dec": -10.0, "drb": None}
        earlier = {"jd": 2460001, "magpsf": 19.4, "sigmapsf": 0.1,
                   "band": "g", "ra": 25.0, "dec": -10.0}
        batch, count = infer_ztf.make_batch([later, earlier, dict(later),
                                             {"jd": 2460000, "magpsf": None}], {})
        self.assertEqual(count, 2)
        self.assertEqual(batch["ts"].shape, (1, 2, 5))
        self.assertAlmostEqual(batch["ts"][0, 0, 0].item(), 0)
        self.assertAlmostEqual(batch["ts"][0, 1, 0].item(), 1)
        self.assertAlmostEqual(batch["ts"][0, 0, 3].item(),
                               infer_ztf.ZTF_fid_to_wavelengths[1])
        self.assertEqual(batch["static"].shape, (1, 30))
        self.assertEqual(batch["static"][0, -1].item(), infer_ztf.flag_value)

    def test_babamul_object_id_uses_object_endpoint_and_token(self):
        payload = {"data": {"objectId": "ZTF26aasktra", "prv_candidates": [
            {"jd": 2460001, "magpsf": 19, "sigmapsf": 0.1, "fid": 1}]}}
        response = io.BytesIO(json.dumps(payload).encode())
        with patch.dict("os.environ", {"BABAMUL_API_TOKEN": "test-token"}), \
             patch.object(infer_ztf, "urlopen", return_value=response) as request:
            rows, context, source_id = infer_ztf.read_source("ZTF26aasktra")
        self.assertEqual(source_id, "ZTF26aasktra")
        self.assertEqual(len(rows), 1)
        self.assertEqual(context["objectId"], source_id)
        http_request = request.call_args.args[0]
        self.assertTrue(http_request.full_url.endswith("/surveys/ZTF/objects/ZTF26aasktra"))
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-token")

    def test_missing_babamul_token_is_clear(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "BABAMUL_API_TOKEN"):
                infer_ztf.read_source("ZTF26aasktra")

    def test_omni_reference_cutout_uses_latest_filter_channel(self):
        image = np.ones((63, 63), dtype=np.float32)
        raw = io.BytesIO()
        fits.PrimaryHDU(image).writeto(raw)
        encoded = base64.b64encode(gzip.compress(raw.getvalue())).decode()
        candidate = {"jd": 2460001, "magpsf": 19, "sigmapsf": 0.1,
                     "fid": 2, "ra": 25.0, "dec": -10.0}
        context = {"candidate": candidate, "cutoutTemplate": {"$binary": {"base64": encoded}}}
        batch, _ = infer_ztf.make_batch([candidate], context, "BTSv2-pro")
        infer_ztf._add_omni_image(batch, [candidate], context)
        stamp = batch["postage_stamp"].numpy()
        self.assertEqual(stamp.shape, (1, 3, 63, 63))
        self.assertEqual(np.count_nonzero(stamp[0, 0]), 0)
        self.assertAlmostEqual(np.linalg.norm(stamp[0, 1]), 1, places=6)
        self.assertEqual(np.count_nonzero(stamp[0, 2]), 0)


if __name__ == "__main__":
    unittest.main()
