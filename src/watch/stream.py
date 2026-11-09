"""Where the live monitor is shown: a desktop window, or a browser page for a machine with no screen."""

import http.server
import json
import queue
import socket
import subprocess
import threading
import time

import cv2

NO_KEY = 0xFF
KEYS = "qme"
SHUTDOWN_COMMAND = ("systemctl", "poweroff")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WATCH - Live Alertness Monitoring</title>
<style>
  body { margin: 0; background: #0e1014; color: #e7e9ee; font: 13px system-ui, sans-serif; text-align: center; }
  img { display: block; margin: 0 auto; max-width: 100%; height: auto; }
  nav { padding: 12px; }
  button { margin: 0 4px; padding: 8px 14px; border: 1px solid #272c36; border-radius: 8px; background: #1d212a; color: inherit; font: inherit; cursor: pointer; }
  button.on { border-color: #00c800; color: #00c800; }
  #shutdown { border-color: #dc0000; color: #ff4d4d; }
  details { max-width: 760px; margin: 0 auto 24px; padding: 0 16px; text-align: left; line-height: 1.6; }
  summary { cursor: pointer; padding: 10px 0; font-weight: 600; color: #9fb4ff; }
  details h3 { margin: 18px 0 4px; font-size: 14px; }
  details ul, details ol { margin: 4px 0; padding-left: 20px; }
  details code { color: #9fb4ff; }
  details .note { color: #a7adb8; }
</style>
</head>
<body>
<img src="/stream" alt="Live monitor">
<nav>
  <button id="sound">Enable alarm sound</button>
  <button data-key="m">Mute / unmute alarm (M)</button>
  <button data-key="e">Export report (E)</button>
  <button data-key="q">Stop monitor (Q)</button>
  <button id="shutdown" hidden>Shut down device</button>
</nav>
<p id="message"></p>
<details>
  <summary>How to use WATCH</summary>

  <h3>Starting</h3>
  <ol>
    <li>Plug in the Pi. The monitor starts by itself and this page loads after 1 to 2 minutes.</li>
    <li>Open this page from a device on the same network, or over a cable straight to the Pi.</li>
    <li>For the first 5 seconds, while it says <b>CALIBRATING</b>, look at the camera normally with your eyes open and <b>without sunglasses</b>. It learns your face from this.</li>
    <li>Then put on sunglasses or a visor if you need them.</li>
  </ol>
  <p class="note">If it starts while your eyes are hidden, it keeps safe default settings instead of learning wrong ones. Step out of view for 3 seconds and come back to calibrate again.</p>

  <h3>Alert levels</h3>
  <ul>
    <li><b>NORMAL</b>: eyes, mouth and head look alert.</li>
    <li><b>WARNING</b>: one sign showed up, such as eyes closed, a yawn or the head dropping. It watches for a while.</li>
    <li><b>CRITICAL</b>: the sign lasted or came back. The buzzer beeps until it clears.</li>
    <li><b>NO FACE DETECTED</b>: nobody is in view. <b>CAMERA LOST</b>: the camera stopped, and the buzzer sounds after 3 seconds.</li>
  </ul>

  <h3>Buttons</h3>
  <ul>
    <li><b>Enable alarm sound</b>: also beeps on this device. Browsers need one click before they can play sound.</li>
    <li><b>Mute / unmute alarm (M)</b>: silences the buzzer and this page.</li>
    <li><b>Export report (E)</b>: saves a report of the session on the Pi.</li>
    <li><b>Stop monitor (Q)</b>: restarts the monitor, which starts again by itself.</li>
    <li><b>Shut down device</b>: always use this before unplugging. Pulling the power while it runs can damage the SD card.</li>
  </ul>

  <h3>Best results</h3>
  <ul>
    <li>Use a lit room. In a fully dark room one IR light is not enough to see through tinted lenses.</li>
    <li>Keep the whole face in view, chin included, at about arm's length.</li>
    <li>Put the IR light beside the camera, aimed at the face, and cover its small light sensor with black tape so it stays on instead of flickering.</li>
    <li>Keep the Pi on a hard surface, not on cloth, so it does not overheat.</li>
  </ul>

  <h3>What's new</h3>
  <ul>
    <li><b>Buzzer</b> on the Pi. It beeps at CRITICAL and goes quiet when the monitor stops. Wiring on the top pin row, counted from the corner hole: VCC on the 1st pin, I/O on the 4th, GND on the 5th.</li>
    <li><b>Sunglasses</b>: the camera is brighter and a contrast filter lifts the eye area, so eyes show through tinted lenses under IR.</li>
    <li><b>Sunglasses mode</b> (<code>eye_decision: cnn</code>): the eye model decides open or closed, since behind lenses the eye outline is a guess.</li>
    <li><b>Training on your own recordings</b> with <code>watch record</code> and <code>watch finetune</code>, run on the Pi over SSH. The buzzer cues each part: one beep open, two beeps closed, a long beep done. See <code>docs/raspberry-pi.md</code>.</li>
  </ul>
</details>
<script>
  const send = (key) => fetch("/key/" + key, { method: "POST" });
  document.querySelectorAll("[data-key]").forEach((button) => button.addEventListener("click", () => send(button.dataset.key)));
  document.addEventListener("keydown", (event) => { if ("qme".includes(event.key.toLowerCase())) send(event.key.toLowerCase()); });

  // Browsers only play sound after a click, so the alarm has to be switched on once.
  let audio = null;
  const sound = document.getElementById("sound");
  sound.addEventListener("click", () => {
    audio = audio || new AudioContext();
    audio.resume();
    sound.textContent = "Alarm sound on";
    sound.classList.add("on");
  });
  function beep() {
    const tone = audio.createOscillator();
    tone.frequency.value = 880;
    tone.connect(audio.destination);
    tone.start();
    tone.stop(audio.currentTime + 0.25);
  }
  const shutdown = document.getElementById("shutdown");
  const message = document.getElementById("message");
  shutdown.addEventListener("click", async () => {
    if (!confirm("Shut down the device? It has to be unplugged and plugged back in to start again.")) return;
    const response = await fetch("/shutdown", { method: "POST" });
    message.textContent = response.ok ? "Shutting down. Wait for the green light to stop blinking, then unplug the power." : "Could not shut down: " + (await response.text());
  });
  setInterval(async () => {
    try {
      const state = await (await fetch("/state")).json();
      shutdown.hidden = !state.shutdown;
      if (state.alarm && audio) beep();
    } catch (error) {}
  }, 500);
</script>
</body>
</html>
"""


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

    def __init__(self, port: int, alarm_active, allow_shutdown: bool = False):
        self._view, self._count = None, 0
        self._allow_shutdown = allow_shutdown
        self._updated = threading.Condition()
        self._keys = queue.SimpleQueue()
        self._alarm_active = alarm_active
        self._server = _Server(("::", port), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"Serving the live monitor at http://{socket.gethostname()}.local:{port} (Ctrl+C to stop)", flush=True)

    def show(self, view, wait_ms: int) -> int:
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

            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif self.path == "/state":
                    self._send(200, "application/json", json.dumps({"alarm": bool(stream._alarm_active()), "shutdown": stream._allow_shutdown}).encode())
                elif self.path == "/stream":
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
                if self.path.startswith("/key/") and len(key) == 1 and key in KEYS:
                    stream._keys.put(key)
                    self._send(204)
                elif self.path == "/shutdown" and stream._allow_shutdown:
                    result = subprocess.run(SHUTDOWN_COMMAND, capture_output=True, text=True)
                    if result.returncode == 0:
                        self._send(204)
                    else:
                        self._send(500, body=(result.stderr.strip() or "not permitted").encode())
                else:
                    self._send(404)

        return Handler
