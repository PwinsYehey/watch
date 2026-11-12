import json
import socket
import urllib.error
import urllib.request

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
        streams.append(BrowserStream(port, lambda: True, **options))
        return port, streams[-1]

    yield start
    for running in streams:
        running.close()


def test_page_and_state(served):
    port, _ = served()
    status, body = _request(port, "/")
    assert status == 200 and b"/stream" in body
    assert b"How to use WATCH" in body
    assert json.loads(_request(port, "/state")[1]) == {"alarm": True, "shutdown": False}


def test_keys_reach_the_monitor(served):
    port, running = served()
    assert _request(port, "/key/m", "POST")[0] == 204
    assert running.show(None, 1) == ord("m")
    assert running.show(None, 1) == NO_KEY
    with pytest.raises(urllib.error.HTTPError):
        _request(port, "/key/x", "POST")


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
