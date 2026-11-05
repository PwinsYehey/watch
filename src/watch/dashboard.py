"""Window layout for the live monitor, matching the web app: a header, the video card and a two-column dashboard."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colours as RGB, the same tokens as web/index.html.
BG, SURFACE, SURFACE_2, BORDER = (14, 16, 20), (22, 25, 32), (29, 33, 42), (39, 44, 54)
TEXT, MUTED, FAINT, ACCENT = (231, 233, 238), (138, 146, 160), (91, 98, 112), (61, 139, 253)
GREEN, YELLOW, RED, GRAY = (0, 200, 0), (255, 215, 0), (220, 0, 0), (120, 120, 120)
LEVELS = {
    0: ("NORMAL", GREEN),
    1: ("WARNING", YELLOW),
    2: ("CRITICAL", RED),
    "no_face": ("NO FACE DETECTED", GRAY),
    "calibrating": ("CALIBRATING", (255, 165, 0)),
    "camera_lost": ("CAMERA LOST", RED),
}

MARGIN, GAP, PAD, HEADER = 16, 16, 12, 52
VIDEO_W, VIDEO_H, CONTROLS_H = 640, 480, 34
DASH_W, DASH_PAD, COLUMN_GAP, ROW = 560, 16, 24, 20
# Rows in the left dashboard column: 4 signs, 8 detection checks and 6 calibration values.
DASH_ROWS = 18
CREDITS = ("Property of Prince Herrera, Ormin Cariaso,", "John Casili, Christenzien Lalimo")

_FONT_FILES = {
    "regular": ("C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"),
    "bold": ("C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"),
}


def _font(weight: str, size: int):
    for path in _FONT_FILES[weight]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


@dataclass
class Row:
    color: tuple | None  # dot colour, or None for a row without a dot
    label: str
    value: str
    value_color: tuple = TEXT
    extra: str = ""


@dataclass
class Level:
    name: str
    color: tuple
    reason: str = ""
    timer: str = ""


class DashboardView:
    def __init__(self):
        self.fonts = {
            "label": _font("regular", 11),
            "small": _font("regular", 12),
            "body": _font("regular", 13),
            "brand": _font("bold", 15),
            "level": _font("bold", 20),
            "key": _font("bold", 12),
        }
        self._text_cache = {}
        # The cards are as tall as the taller of the video stack and the dashboard's left column.
        card_height = max(PAD * 2 + VIDEO_H + 10 + CONTROLS_H, 14 + 96 + 3 * 42 + DASH_ROWS * ROW + 14)
        self.size = (MARGIN * 2 + PAD * 2 + VIDEO_W + GAP + DASH_W, HEADER + MARGIN * 2 + card_height)
        self._card_top, self._card_bottom = HEADER + MARGIN, self.size[1] - MARGIN
        self._dash_left = MARGIN + PAD * 2 + VIDEO_W + GAP
        # The header and empty cards never change, so they are drawn once and copied each frame.
        self._base = self._canvas = Image.new("RGB", self.size, BG)
        draw = ImageDraw.Draw(self._base)
        self._header(draw, self.size[0])
        draw.rounded_rectangle((MARGIN, self._card_top, self._dash_left - GAP, self._card_bottom), 10, SURFACE, BORDER)
        draw.rounded_rectangle((self._dash_left, self._card_top, self._dash_left + DASH_W, self._card_bottom), 10, SURFACE, BORDER)
        draw.rounded_rectangle((MARGIN + PAD, self._card_top + PAD, MARGIN + PAD + VIDEO_W - 1, self._card_top + PAD + VIDEO_H - 1), 8, (0, 0, 0))
        self._video_mask = Image.new("L", (VIDEO_W, VIDEO_H), 0)
        ImageDraw.Draw(self._video_mask).rounded_rectangle((0, 0, VIDEO_W - 1, VIDEO_H - 1), 8, 255)

    def _text(self, xy, text, color, font, anchor="la"):
        """Draws text from a cache of rendered strings, since most of the dashboard text repeats every frame."""
        if not text:
            return
        key = (text, color, font, anchor)
        if (cached := self._text_cache.get(key)) is None:
            if len(self._text_cache) > 2000:
                self._text_cache.clear()
            left, top, right, bottom = self.fonts[font].getbbox(text, anchor=anchor)
            glyphs = Image.new("RGBA", (max(right - left, 1), max(bottom - top, 1)), (0, 0, 0, 0))
            ImageDraw.Draw(glyphs).text((-left, -top), text, color, self.fonts[font], anchor=anchor)
            cached = self._text_cache[key] = (glyphs, left, top)
        glyphs, left, top = cached
        self._canvas.paste(glyphs, (int(xy[0] + left), int(xy[1] + top)), glyphs)

    def render(self, frame_bgr, level: Level, signs, checks, calibration, events, live: bool, muted: bool, readings=(), notice="") -> np.ndarray:
        """Returns the whole window as a BGR image."""
        image = self._canvas = self._base.copy()
        draw = ImageDraw.Draw(image)
        card_top, card_bottom = self._card_top, self._card_bottom

        # Video card with the controls row under it.
        self._video(image, frame_bgr, (MARGIN + PAD, card_top + PAD))
        self._controls(draw, MARGIN + PAD, card_bottom - PAD - CONTROLS_H, live, muted, notice)

        # Dashboard: alert level, signs, detection and calibration on the left; recent events on the right.
        left = self._dash_left
        column = (DASH_W - DASH_PAD * 2 - COLUMN_GAP) // 2
        x, y = left + DASH_PAD, card_top + 14
        y = self._level(draw, x, y, column, level)
        for title, rows in (("Signs", signs), ("Detection", checks), ("Calibration", calibration)):
            y = self._divider(draw, x, y, column)
            section_top = y
            y = self._rows(draw, x, y, column, title, rows)
        # Live values sit at the bottom of the right column, beside Calibration; recent events fill the space above.
        right = x + column + COLUMN_GAP
        live_top = section_top
        self._events(draw, right, card_top + 14, column, live_top - 22, events)
        self._divider(draw, right, live_top - 22, column)
        self._rows(draw, right, live_top, column, "Live values", readings)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _header(self, draw, width):
        draw.rectangle((0, 0, width, HEADER - 1), SURFACE)
        draw.line((0, HEADER - 1, width, HEADER - 1), BORDER)
        cx, cy = MARGIN + 13, HEADER // 2
        draw.ellipse((cx - 13, cy - 8, cx + 13, cy + 8), outline=ACCENT, width=2)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=ACCENT, width=2)
        self._text((MARGIN + 38, 9), "W.A.T.C.H", TEXT, "brand")
        self._text((MARGIN + 38, 28), "Wakefulness Assessment through Tracking of Cues & Head", MUTED, "small")
        for i, line in enumerate(CREDITS):
            self._text((width - MARGIN, 12 + i * 15), line, FAINT, "label", "ra")

    def _video(self, image, frame_bgr, origin):
        h, w = frame_bgr.shape[:2]
        scale = min(VIDEO_W / w, VIDEO_H / h)
        resized = frame_bgr if scale == 1 else cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        slot = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        y, x = (VIDEO_H - resized.shape[0]) // 2, (VIDEO_W - resized.shape[1]) // 2
        slot[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[:, :, ::-1]
        image.paste(Image.fromarray(slot), origin, self._video_mask)

    def _controls(self, draw, x, y, live, muted, notice=""):
        """Key hints styled like the web buttons, since the window has no clickable controls."""
        keys = [("Q", "Quit")]
        if live:
            keys += [("M", "Unmute alarm" if muted else "Mute alarm"), ("E", "Export")]
        for key, action in keys:
            width = 46 + int(draw.textlength(action, self.fonts["body"]))
            draw.rounded_rectangle((x, y, x + width, y + CONTROLS_H), 8, SURFACE_2, BORDER)
            draw.rounded_rectangle((x + 8, y + 8, x + 26, y + 26), 4, BORDER)
            self._text((x + 17, y + 17), key, TEXT, "key", "mm")
            self._text((x + 34, y + 17), action, TEXT, "body", "lm")
            x += width + 10
        if notice:
            self._text((x + 4, y + 17), notice, GREEN, "small", "lm")
        elif live:
            draw.ellipse((x + 4, y + 13, x + 12, y + 21), (255, 77, 77))
            self._text((x + 18, y + 17), "Live", MUTED, "body", "lm")

    def _label(self, draw, x, y, text):
        self._text((x, y), text.upper(), MUTED, "label")
        return y + 20

    def _divider(self, draw, x, y, width):
        draw.line((x, y + 12, x + width, y + 12), BORDER)
        return y + 22

    def _level(self, draw, x, y, width, level: Level):
        y = self._label(draw, x, y, "Alert level")
        draw.rounded_rectangle((x, y, x + width, y + 76), 8, level.color)
        ink = (0, 0, 0) if level.color == YELLOW else (255, 255, 255)
        self._text((x + 12, y + 8), level.name, ink, "level")
        self._text((x + 12, y + 36), level.reason, ink, "body")
        self._text((x + 12, y + 55), level.timer, ink, "small")
        return y + 76

    def _rows(self, draw, x, y, width, title, rows):
        y = self._label(draw, x, y, title)
        if not rows:
            self._text((x, y), "—", MUTED, "body")
            return y + ROW
        for row in rows:
            if row.color:
                draw.ellipse((x + 1, y + 6, x + 9, y + 14), row.color)
            self._text((x + 18, y), row.label, TEXT, "body")
            self._text((x + 112, y), row.value, row.value_color, "body")
            if row.extra:
                self._text((x + width, y), row.extra, MUTED, "body", "ra")
            y += ROW
        return y

    def _events(self, draw, x, y, width, bottom, events):
        y = self._label(draw, x, y, "Recent events")
        if not events:
            self._text((x, y), "none yet", MUTED, "small")
        for text, color in events:
            if y + 18 > bottom:
                break
            self._text((x, y), text, color, "small")
            y += 20

