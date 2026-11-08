"""High-compatibility server for the Telecare prototype (Project Review 2).
Uses Python's standard library to guarantee flawless out-of-the-box demonstration.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DIST = ROOT / "frontend" / "dist"
DATA_DIR = BASE / "data"

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from services import analyze  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000


class Handler(BaseHTTPRequestHandler):
    server_version = "TelecareReview2/2.0"

    def _send(self, status, body, content_type="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._json({
                "status": "ok",
                "service": "telecare-prototype",
                "phase": "Project Review 2",
                "pipeline": "IndicNLP -> Transformer-CRF -> SNOMED/ICD -> Triage -> Doctor Summary"
            })
            return
        if path == "/api/demo-cases":
            p = DATA_DIR / "demo_cases.json"
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
            self._json(data)
            return
        if path == "/api/model-evaluation":
            p = DATA_DIR / "evaluation_metrics.json"
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            self._json(data)
            return
        if path == "/api/terminology":
            p = DATA_DIR / "terminology.json"
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
            self._json(data)
            return

        self._serve_static(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/analyze", "/api/intake-report"):
            self._json({"detail": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 200_000:
                raise ValueError("Request body is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            if len(text) < 2:
                raise ValueError("text must contain at least 2 characters")
            if len(text) > 2000:
                raise ValueError("text must be at most 2000 characters")

            analysis = analyze(text)
            if path == "/api/intake-report":
                self._json(analysis.get("intake_report", {}))
            else:
                self._json(analysis)
        except Exception as exc:
            self._json({"detail": str(exc)}, 400)

    def _json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, raw, "application/json; charset=utf-8")

    def _serve_static(self, path):
        if not DIST.exists():
            self._send(500, "frontend/dist is missing")
            return
        relative = path.lstrip("/")
        target = DIST / relative if relative else DIST / "index.html"
        if not target.exists() or not target.is_file():
            target = DIST / "index.html"
        try:
            target = target.resolve()
            if DIST.resolve() not in target.parents and target != DIST.resolve() / "index.html":
                self._send(403, "Forbidden")
                return
            content = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            self._send(200, content, content_type)
        except OSError as exc:
            self._send(500, str(exc))

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    print(f"Starting Telecare Project Review 2 server at http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()
