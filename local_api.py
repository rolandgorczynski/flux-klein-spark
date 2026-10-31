"""Optional authenticated OpenAI-compatible image API on the private Docker network."""
import base64
import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "flux.2-klein-9b"


def make_handler(api_key, generate, ready=lambda: True):
    busy = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not log prompts or credentials.

        def respond(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + api_key).encode()):
                return True
            self.respond(401, {"error": {"message": "Invalid API key"}})
            return False

        def do_GET(self):
            if self.path == "/healthz":
                loaded = ready()
                return self.respond(200 if loaded else 503, {"status": "ready" if loaded else "loading", "model": MODEL_ID})
            if not self.authorized():
                return
            if self.path == "/v1/models":
                return self.respond(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "owned_by": "local"}
                ]})
            self.respond(404, {"error": {"message": "Endpoint not supported"}})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/images/generations":
                return self.respond(404, {"error": {"message": "Endpoint not supported"}})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be at most 64 KiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if body.get("model", MODEL_ID) != MODEL_ID:
                    return self.respond(404, {"error": {"message": "Unknown model"}})
                prompt = body.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
                    raise ValueError("Prompt must contain 1 to 8000 characters")
                if body.get("n", 1) != 1 or body.get("response_format", "b64_json") != "b64_json":
                    raise ValueError("Only n=1 and b64_json output are supported")
                size = body.get("size", "1024x1024")
                if not isinstance(size, str):
                    raise ValueError("Invalid image size")
                width, height = map(int, size.split("x"))
                if any(v < 256 or v > 2048 or v % 32 for v in (width, height)):
                    raise ValueError("Dimensions must be multiples of 32 between 256 and 2048")
                seed = body.get("seed", secrets.randbelow(2**31))
                if type(seed) is not int or not 0 <= seed < 2**31:
                    raise ValueError("Invalid seed")
            except (ValueError, TypeError, UnicodeError):
                return self.respond(400, {"error": {"message": "Invalid image request"}})
            if not ready():
                return self.respond(503, {"error": {"message": "The local model is still loading"}})
            if not busy.acquire(blocking=False):
                return self.respond(429, {"error": {"message": "Image generation is busy"}})
            try:
                png = generate(prompt, width, height, seed)
                self.respond(200, {"data": [{"b64_json": base64.b64encode(png).decode()}]})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"Local image API generation failed ({type(exc).__name__})", flush=True)
                self.respond(503, {"error": {"message": "Local image generation failed"}})
            finally:
                busy.release()

    return Handler


def start_image_api(api_key, generate, ready=lambda: True):
    if not api_key:
        return
    if len(api_key) < 32:
        raise ValueError("FLUX_API_KEY must contain at least 32 characters")
    server = ThreadingHTTPServer(("0.0.0.0", 7862), make_handler(api_key, generate, ready))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Local FLUX image API listening on :7862 (authenticated)", flush=True)
