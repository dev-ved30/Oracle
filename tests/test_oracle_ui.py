"""Exercise the local one-request source classification API."""

import base64
import gzip
import io
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from PIL import Image

from oracle_ui import server


class TestOracleUi(unittest.TestCase):
    def setUp(self):
        self.client = server.create_app().test_client()
        image_file = io.BytesIO()
        fits.PrimaryHDU(np.ones((63, 63), dtype=np.float32)).writeto(image_file)
        self.template = base64.b64encode(gzip.compress(image_file.getvalue())).decode()
        earlier = {"jd": 2460000.0, "magpsf": 19.7, "sigmapsf": 0.2,
                   "fid": 1, "ra": 151.0, "dec": 12.0}
        latest = {"jd": 2460002.0, "magpsf": 18.9, "sigmapsf": 0.1,
                  "fid": 2, "ra": 151.0, "dec": 12.0}
        self.rows = [earlier, latest]
        self.context = {"objectId": "ZTF18abmrfqv", "candid": 123, "candidate": latest}
        self.result = {"source_id": "ZTF18abmrfqv", "model": "BTSv2-pro",
                       "missing_context_features": [],
                       "probabilities_by_level": {"1": {"Persistent": 0.8, "Transient": 0.2},
                                                  "2": {"AGN": 0.8}}}

    def test_fetch_and_classify_in_one_request(self):
        with patch.object(server, "read_source", return_value=(self.rows, self.context, "ZTF18abmrfqv")) as lookup, \
             patch.object(server, "_babamul_get", return_value={"cutoutTemplate": self.template}), \
             patch.object(server, "_ps_preview", return_value="data:image/jpeg;base64,example"), \
             patch.object(server, "classify_source", return_value=self.result) as classify, \
             patch.object(server, "classify_rolling_source") as rolling:
            response = self.client.post("/api/analyze", json={"object_id": "ZTF18abmrfqv", "model": "BTSv2-pro"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsNone(body["error"])
        self.assertEqual(body["source"]["detections"], 2)
        self.assertEqual(len(body["source"]["photometry"]), 2)
        self.assertEqual(len(body["source"]["metadata"]), 30)
        self.assertTrue(body["source"]["image"].startswith("data:image/png;base64,"))
        self.assertTrue(body["source"]["ps_image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(body["classification"], self.result)
        self.assertIsNone(body["rolling"])
        rolling.assert_not_called()
        lookup.assert_called_once_with("ZTF18abmrfqv")
        self.assertIs(classify.call_args.args[0], self.rows)
        self.assertEqual(classify.call_args.args[3], "BTSv2-pro")

    def test_rolling_request_returns_series_and_final_classification(self):
        series = {"points": [{"observation": 1, "days": 0, "probabilities": {"AGN": 0.8}}], "note": "Test"}
        with patch.object(server, "read_source", return_value=(self.rows, self.context, "ZTF18abmrfqv")), \
             patch.object(server, "_babamul_get", return_value={"cutoutTemplate": self.template}), \
             patch.object(server, "_ps_preview", return_value=None), \
             patch.object(server, "classify_rolling_source", return_value={"classification": self.result, "rolling": series}) as classify, \
             patch.object(server, "classify_source") as regular:
            response = self.client.post("/api/analyze", json={"object_id": "ZTF18abmrfqv", "model": "BTSv2-pro", "rolling": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["rolling"], series)
        self.assertEqual(response.get_json()["classification"], self.result)
        classify.assert_called_once()
        regular.assert_not_called()

    def test_classification_error_keeps_source_preview(self):
        with patch.object(server, "read_source", return_value=(self.rows, self.context, "ZTF18abmrfqv")), \
             patch.object(server, "_babamul_get", return_value={"cutoutTemplate": self.template}), \
             patch.object(server, "_ps_preview", side_effect=ValueError("Pan-STARRS unavailable")), \
             patch.object(server, "classify_source", side_effect=ValueError("model failed")):
            response = self.client.post("/api/analyze", json={"object_id": "ZTF18abmrfqv", "model": "BTSv2-pro"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["source"]["detections"], 2)
        self.assertIsNone(body["classification"])
        self.assertEqual(body["error"], "model failed")
        self.assertIsNone(body["source"]["ps_image"])
        self.assertEqual(body["source"]["ps_image_error"], "Pan-STARRS unavailable")

    def test_ps_color_cutout_uses_gr_i_channels(self):
        listing = b"filter filename\ng /g.fits\nr /r.fits\ni /i.fits\n"
        output = io.BytesIO()
        Image.new("RGB", (315, 315), (20, 30, 40)).save(output, format="JPEG")
        with patch.object(server, "urlopen", side_effect=[io.BytesIO(listing), io.BytesIO(output.getvalue())]) as fetch:
            image = server._ps_preview(151.0, 12.0)
        self.assertTrue(image.startswith("data:image/jpeg;base64,"))
        self.assertIn("red=%2Fi.fits", fetch.call_args_list[1].args[0])
        self.assertIn("green=%2Fr.fits", fetch.call_args_list[1].args[0])
        self.assertIn("blue=%2Fg.fits", fetch.call_args_list[1].args[0])

    def test_invalid_request(self):
        self.assertEqual(self.client.post("/api/analyze", json={"object_id": "../../etc/passwd", "model": "BTSv2-pro"}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json={"object_id": "ZTF18abmrfqv", "model": "other"}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json={"object_id": "ZTF18abmrfqv", "model": "BTSv2", "rolling": "yes"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
