"""Where the live monitor is shown: a desktop window, or a browser page for a machine with no screen."""

import http.server
import json
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

import cv2

from watch.dashboard import WebFrame

NO_KEY = 0xFF
# Page key to monitor key; "a" answers a check-in, which the desktop window takes as the space bar.
KEYS = {"q": "q", "m": "m", "e": "e", "a": " ", "p": "p"}
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
FILES = {"/": "pi.html", "/pi.js": "pi.js", "/style.css": "style.css", "/favicon.svg": "favicon.svg", "/apple-touch-icon.png": "apple-touch-icon.png"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png", ".csv": "text/csv"}
MAX_UPLOAD = 20 * 1024 * 1024
SHUTDOWN_COMMAND = ("systemctl", "poweroff")



class _Server(http.server.ThreadingHTTPServer):
    """Listens on IPv4 and IPv6, since a name like raspberrypi.local may resolve to either."""

    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


class Window:
    """The desktop window, with keys read from the keyboard."""

    def __init__(self, title: str):
        self._title = title

    def show(self, view, wait_ms: int) -> int:
        cv2.imshow(self._title, view)
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self):
        cv2.destroyAllWindows()


class BrowserStream:
    """The same view as an MJPEG stream on a web page, with keys sent back from the page."""

    def __init__(self, port: int, status, allow_shutdown: bool = False, analyze=None, sessions: Path | None = None):
        self._view, self._count, self._state = None, 0, None
        self._allow_shutdown = allow_shutdown
        self._updated = threading.Condition()
        self._keys = queue.SimpleQueue()
        self._status = status
        self._analyze = analyze
        self._sessions = sessions
        self._server = _Server(("::", port), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"Serving the live monitor at http://{socket.gethostname()}.local:{port} (Ctrl+C to stop)", flush=True)

    def show(self, view, wait_ms: int) -> int:
        if isinstance(view, WebFrame):
            view, self._state = view.image, view.state
        with self._updated:
            self._view, self._count = view, self._count + 1
            self._updated.notify_all()
        if wait_ms > 1:
            time.sleep(wait_ms / 1000)
        try:
            return ord(self._keys.get_nowait())
        except queue.Empty:
            return NO_KEY

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def _frames(self):
        """JPEGs of each new view, encoded only while a browser is watching."""
        seen = 0
        while True:
            with self._updated:
                if not self._updated.wait_for(lambda: self._count != seen, timeout=5.0):
                    continue
                view, seen = self._view, self._count
            ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield jpeg.tobytes()

    def _handler(self):
        stream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, content_type="text/plain", body=b""):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _file(self, path: Path):
                self._send(200, TYPES.get(path.suffix, "application/octet-stream"), path.read_bytes())

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in FILES and (WEB_DIR / FILES[path]).is_file():
                    self._file(WEB_DIR / FILES[path])
                elif path == "/state":
                    state = {**stream._status(), "shutdown": stream._allow_shutdown, "view": stream._state}
                    self._send(200, "application/json", json.dumps(state).encode())
                elif path.startswith("/sessions/") and stream._sessions is not None:
                    name = Path(path.removeprefix("/sessions/")).name
                    target = stream._sessions / name
                    if name.startswith("watch-session-") and target.is_file():
                        self._file(target)
                    else:
                        self._send(404)
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        for jpeg in stream._frames():
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n")
                    except OSError:
                        pass
                else:
                    self._send(404)

            def do_POST(self):
                key = self.path.removeprefix("/key/")
                if self.path.startswith("/key/") and key in KEYS:
                    stream._keys.put(KEYS[key])
                    self._send(204)
                elif self.path.split("?")[0] == "/analyze" and stream._analyze is not None:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_UPLOAD:
                        self._send(413)
                        return
                    data = self.rfile.read(length) if length else None
                    result = stream._analyze(data)
                    if result is None:
                        self._send(400, body=b"could not read that image")
                    else:
                        self._send(200, "application/json", json.dumps(result).encode())
                elif self.path == "/shutdown" and stream._allow_shutdown:
                    result = subprocess.run(SHUTDOWN_COMMAND, capture_output=True, text=True)
                    if result.returncode == 0:
                        self._send(204)
                    else:
                        self._send(500, body=(result.stderr.strip() or "not permitted").encode())
                else:
                    self._send(404)

        return Handler
