import base64
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from local_api import MODEL_ID, make_handler


class LocalAPITest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.ready = True

        def generate(*args):
            self.calls.append(args)
            return b"test-png"

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("x" * 32, generate, lambda: self.ready))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, body=None, key="x" * 32):
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        request = urllib.request.Request(self.base + path, headers=headers,
                                         data=None if body is None else json.dumps(body).encode())
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def test_authentication_and_model_listing(self):
        self.assertEqual(self.request("/v1/models", key="wrong")[0], 401)
        status, body = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], MODEL_ID)
        self.assertEqual(self.calls, [])

    def test_generation_returns_base64_and_forwards_dimensions(self):
        status, body = self.request("/v1/images/generations", {
            "model": MODEL_ID, "prompt": "A blue vase", "size": "512x768", "seed": 42,
        })
        self.assertEqual(status, 200)
        self.assertEqual(base64.b64decode(body["data"][0]["b64_json"]), b"test-png")
        self.assertEqual(self.calls, [("A blue vase", 512, 768, 42)])

    def test_invalid_requests_do_not_generate(self):
        for body, expected in [
            ({"prompt": ""}, 400),
            ({"prompt": "test", "size": "10000x10000"}, 400),
            ({"prompt": "test", "size": "513x512"}, 400),
            ({"prompt": "test", "model": "another-model"}, 404),
            ({"prompt": "test", "n": 2}, 400),
            ({"prompt": "test", "seed": -1}, 400),
        ]:
            self.assertEqual(self.request("/v1/images/generations", body)[0], expected)
        self.assertEqual(self.calls, [])

    def test_hycanvas_widescreen_size(self):
        status, _ = self.request("/v1/images/generations", {"prompt": "Landscape", "size": "1792x1024"})
        self.assertEqual(status, 200)
        self.assertEqual(self.calls[0][1:3], (1792, 1024))

    def test_model_loading_is_reported_without_generating(self):
        self.ready = False
        self.assertEqual(self.request("/healthz")[0], 503)
        self.assertEqual(self.request("/v1/images/generations", {"prompt": "A vase"})[0], 503)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
