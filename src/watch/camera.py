"""Camera sources for the live monitor: an OpenCV camera index, or a Raspberry Pi camera module."""

import shutil
import subprocess
import sys
import threading

import cv2
import numpy as np


def camera_source(value: str):
    """Parses --camera: an OpenCV index, or "pi" for a Raspberry Pi camera module."""
    return value if value == "pi" else int(value)


class PiCamera:
    """A Raspberry Pi camera module read as raw YUV frames from rpicam-vid, since OpenCV cannot open it by index."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, open_timeout: float = 8.0):
        self._shape = (height * 3 // 2, width)
        self._frame = None
        self._fresh = threading.Event()
        self._process = None
        command = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if command is None:
            return
        # A camera with no IR filter turns skin pink and washes out a face close to its IR LEDs, so meter on the centre, expose lower and drop the colour.
        self._process = subprocess.Popen(
            [command, "--nopreview", "--timeout", "0", "--codec", "yuv420", "--width", str(width), "--height", str(height), "--framerate", str(fps), "--metering", "spot", "--ev", "-1", "--saturation", "0", "--output", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=self._pump, args=(self._process,), daemon=True).start()
        # rpicam-vid starts even with no camera attached, so it counts as open only once a frame arrives.
        self._fresh.wait(open_timeout)
        if self._frame is None:
            self.release()

    def _pump(self, process):
        """Keeps only the newest frame, so a slow consumer never falls behind the camera."""
        size = self._shape[0] * self._shape[1]
        while len(data := process.stdout.read(size)) == size:
            self._frame = np.frombuffer(data, np.uint8).reshape(self._shape)
            self._fresh.set()
        self._fresh.set()

    def isOpened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def read(self):
        if not self.isOpened() or not self._fresh.wait(2.0):
            return False, None
        self._fresh.clear()
        if (frame := self._frame) is None or not self.isOpened():
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


def open_camera(source):
    if source == "pi":
        return PiCamera()
    return cv2.VideoCapture(source, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
