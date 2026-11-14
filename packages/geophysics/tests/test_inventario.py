import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import parse_qs, urlparse

import requests
from ingemmet_peligros import inventario

LAYER = {
    "name": "Peligros Geológicos",
    "fields": [
        {"name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"name": "NM_DPTO", "type": "esriFieldTypeString"},
        {"name": "FECHA", "type": "esriFieldTypeDate"},
        {
            "name": "TIP_PELIGRO",
            "type": "esriFieldTypeString",
            "domain": {
                "type": "codedValue",
                "name": "tipo",
                "codedValues": [
                    {"name": "Flujos", "code": "1000|1500"},
                    {"name": "Caida", "code": "1000|1100"},
                ],
            },
        },
        {"name": "SHAPE", "type": "esriFieldTypeGeometry"},
    ],
}

ROWS = {
    oid: {
        "attributes": {
            "OBJECTID": oid,
            "NM_DPTO": "Ancash",
            "FECHA": 1175403600000 if oid % 2 else None,
            "TIP_PELIGRO": "1000|1500",
        },
        "geometry": {"x": -77.5 - oid / 10, "y": -9.5},
    }
    for oid in range(10, 15)
}


class FakeArcGis(BaseHTTPRequestHandler):
    """Imita la capa de INGEMMET: sin resultOffset y con maxRecordCount de 3."""

    max_record_count = 3
    failures_left = 0
    hang = False
    requests_seen: list[dict]

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        type(self).requests_seen.append(query)
        if type(self).hang:
            threading.Event().wait(3)
            return
        if type(self).failures_left:
            type(self).failures_left -= 1
            self.send_response(503)
            self.end_headers()
            return
        if "resultOffset" in query:
            body = {"error": {"code": 400, "message": "Pagination is not supported."}}
        elif url.path.endswith("/query"):
            body = self.query(query)
        else:
            body = LAYER
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def query(self, query):
        if query.get("returnIdsOnly") == "true":
            return {"objectIdFieldName": "OBJECTID", "objectIds": list(ROWS)}
        ids = [int(i) for i in query["objectIds"].split(",")]
        body = {"features": [ROWS[i] for i in ids[: self.max_record_count]]}
        if len(ids) > self.max_record_count:
            body["exceededTransferLimit"] = True
        return body

    def log_message(self, *args):
        pass


class InventarioTest(unittest.TestCase):
    def setUp(self):
        FakeArcGis.failures_left = 0
        FakeArcGis.hang = False
        FakeArcGis.requests_seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeArcGis)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}/layer/0"
        self.session = inventario.make_session(ca_file=None)

    def test_download_batches_ids_and_returns_geojson_with_domains(self):
        geojson, tables = inventario.download(
            session=self.session, layer_url=self.url, batch_size=2
        )

        batches = [q["objectIds"] for q in FakeArcGis.requests_seen if "objectIds" in q]
        self.assertEqual(batches, ["10,11", "12,13", "14"])
        self.assertTrue(all("resultOffset" not in q for q in FakeArcGis.requests_seen))
        self.assertTrue(
            all(
                q["outSR"] == "4326"
                for q in FakeArcGis.requests_seen
                if "objectIds" in q
            )
        )

        first, second = geojson["features"][:2]
        self.assertEqual(len(geojson["features"]), 5)
        self.assertEqual(
            first["geometry"], {"type": "Point", "coordinates": [-78.5, -9.5]}
        )
        self.assertEqual(first["properties"]["FECHA"], None)
        self.assertEqual(second["properties"]["FECHA"], "2007-04-01T05:00:00Z")
        self.assertEqual(
            tables, {"TIP_PELIGRO": {"1000|1500": "Flujos", "1000|1100": "Caida"}}
        )

    def test_batch_over_max_record_count_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "maxRecordCount"):
            inventario.download(session=self.session, layer_url=self.url, batch_size=5)

    def test_transient_server_errors_are_retried(self):
        FakeArcGis.failures_left = 2
        layer = inventario.fetch_layer(self.session, self.url)
        self.assertEqual(layer["name"], "Peligros Geológicos")

    def test_service_error_in_a_200_response_raises(self):
        with self.assertRaisesRegex(RuntimeError, "Pagination is not supported"):
            inventario.get_json(self.session, self.url, {"resultOffset": 0})

    def test_unresponsive_server_times_out(self):
        FakeArcGis.hang = True
        with (
            mock.patch.object(inventario, "TIMEOUT", (1, 0.3)),
            mock.patch.object(inventario, "RETRIES", 0),
        ):
            session = inventario.make_session(ca_file=None)
            with self.assertRaisesRegex(
                requests.exceptions.ConnectionError, "Read timed out"
            ):
                inventario.fetch_layer(session, self.url)


if __name__ == "__main__":
    unittest.main()
