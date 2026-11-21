import json
import socket
import urllib.error
import urllib.request

import numpy as np
import pytest

from watch import stream
from watch.stream import NO_KEY, BrowserStream


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(port, path, method="GET"):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


@pytest.fixture
def served():
    """Starts a BrowserStream and returns (port, stream) for it."""
    streams = []

    def start(**options):
        port = _free_port()
        streams.append(BrowserStream(port, lambda: {"alarm": True}, **options))
        return port, streams[-1]

    yield start
    for running in streams:
        running.close()


def test_page_and_state(served):
    port, _ = served()
    status, body = _request(port, "/")
    assert status == 200 and b"pi.js" in body and b"style.css" in body
    assert b"shutdownBtn" in body
    assert b"/stream" in _request(port, "/pi.js")[1]
    assert _request(port, "/style.css")[0] == 200
    assert json.loads(_request(port, "/state")[1]) == {"alarm": True, "shutdown": False, "view": None}


def test_state_carries_the_dashboard(served):
    from watch.dashboard import GREEN, Level, Row, WebView

    port, running = served()
    frame = WebView().render(np.zeros((4, 4, 3), np.uint8), Level("NORMAL", GREEN), [Row(GREEN, "Eyes", "open")], [], [], [("12:00:00 NORMAL", GREEN)], live=True, muted=False)
    running.show(frame, 1)
    view = json.loads(_request(port, "/state")[1])["view"]
    assert view["level"]["name"] == "NORMAL" and view["level"]["color"] == "#00c800"
    assert view["signs"][0]["label"] == "Eyes" and view["events"][0]["text"] == "12:00:00 NORMAL"


def test_photo_goes_to_the_monitor(served):
    received = []
    port, _ = served(analyze=lambda data: received.append(data) or {"face": False})
    request = urllib.request.Request(f"http://127.0.0.1:{port}/analyze", data=b"jpeg", method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        assert json.loads(response.read()) == {"face": False}
    assert _request(port, "/analyze", "POST")[0] == 200
    assert received == [b"jpeg", None]


def test_exported_files_download_but_nothing_else(served, tmp_path):
    (tmp_path / "watch-session-1.csv").write_text("a,b")
    (tmp_path / "secret.txt").write_text("no")
    port, _ = served(sessions=tmp_path)
    assert _request(port, "/sessions/watch-session-1.csv")[1] == b"a,b"
    for path in ("/sessions/secret.txt", "/sessions/../secret.txt", "/sessions/watch-session-2.csv"):
        with pytest.raises(urllib.error.HTTPError):
            _request(port, path)


def test_keys_reach_the_monitor(served):
    port, running = served()
    assert _request(port, "/key/m", "POST")[0] == 204
    assert running.show(None, 1) == ord("m")
    assert running.show(None, 1) == NO_KEY
    with pytest.raises(urllib.error.HTTPError):
        _request(port, "/key/x", "POST")


def test_check_in_answer_arrives_as_the_space_bar(served):
    port, running = served()
    assert _request(port, "/key/a", "POST")[0] == 204
    assert running.show(None, 1) == ord(" ")


def test_shutdown_is_off_unless_allowed(served, monkeypatch):
    ran = []
    monkeypatch.setattr(stream.subprocess, "run", lambda *args, **kwargs: ran.append(args))
    port, _ = served()
    with pytest.raises(urllib.error.HTTPError):
        _request(port, "/shutdown", "POST")
    assert not ran


def test_shutdown_runs_the_command_when_allowed(served, monkeypatch):
    ran = []

    def fake_run(command, **kwargs):
        ran.append(command)
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(stream.subprocess, "run", fake_run)
    port, _ = served(allow_shutdown=True)
    assert json.loads(_request(port, "/state")[1])["shutdown"] is True
    assert _request(port, "/shutdown", "POST")[0] == 204
    assert ran == [stream.SHUTDOWN_COMMAND]
