import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FIGURES, FONT_DIR, ROOT

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent / "flowchart"
SCALE = 8
CUTS = (8376, 13402)
TITLE = "W.A.T.C.H · System architecture flowchart"
DESCRIPTION = [
    "From camera input to alerts, display and reports. The Python live monitor (desktop and Raspberry Pi) follows this whole flow. The web app follows the face",
    "flow with slump alerts and the check-in; it has no body-language mode, nod counter or stillness check, and uses buttons instead of keys. Updated November 2026.",
]


def render(source: Path, output: Path):
    config = os.environ.get("WATCH_MMDC_PUPPETEER")
    command = ["npx", "mmdc", "-c", str(HERE / "config.json"), "-i", str(source), "-o", str(output), "-s", str(SCALE), "-b", "white"]
    if config:
        command[2:2] = ["-p", config]
    subprocess.run(command, check=True)


def compose(main: Image.Image, legend: Image.Image) -> Image.Image:
    width, pad = main.width, 96
    bold = lambda size: ImageFont.truetype(str(FONT_DIR / "segoeuib.ttf"), size)
    regular = lambda size: ImageFont.truetype(str(FONT_DIR / "segoeui.ttf"), size)
    legend_width = int(width * 0.66)
    legend = legend.resize((legend_width, int(legend.height * legend_width / legend.width)), Image.LANCZOS)
    box_top, box_height = pad + 260, legend.height + 80
    height = box_top + box_height + 60
    head = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(head)
    draw.text((pad, pad), TITLE, font=bold(84), fill="#0b0b0b")
    for i, line in enumerate(DESCRIPTION):
        draw.text((pad, pad + 120 + i * 50), line, font=regular(36), fill="#52514e")
    draw.rounded_rectangle((pad, box_top, width - pad, box_top + box_height), radius=24, outline="#e6e5e0", width=4)
    draw.text((pad + 40, box_top + box_height // 2 - 18), "SYMBOLS", font=bold(30), fill="#0b0b0b")
    head.paste(legend, (pad + 260, box_top + 40))
    notes_x = pad + 260 + legend_width + 60
    draw.text((notes_x, box_top + box_height // 2 - 50), "A  Connector A: back to Read camera frame (process the next frame)", font=regular(30), fill="#52514e")
    draw.text((notes_x, box_top + box_height // 2 + 6), "B  Connector B: shut down (stop alarm, release camera, End)", font=regular(30), fill="#52514e")
    full = Image.new("RGB", (width, height + main.height), "white")
    full.paste(head, (0, 0))
    full.paste(main, (0, height))
    return full


def offline_panel(full: Image.Image) -> Image.Image:
    k = 4
    rgb = np.asarray(full.crop((0, 0, full.width, full.height // 2)).resize((full.width // k, full.height // 2 // k), Image.NEAREST)).astype(np.int16)
    grey = np.abs(rgb - np.array([247, 247, 245], np.int16)).sum(2) <= 3
    labels, count = ndimage.label(grey)
    sizes = ndimage.sum(grey, labels, range(1, count + 1))
    boxes = {i + 1: ndimage.find_objects((labels == i + 1).astype(int))[0] for i, size in enumerate(sizes) if size > 2000}
    first = boxes[min(boxes, key=lambda i: (boxes[i][0].start, boxes[i][1].start))]
    x0, x1, y0 = first[1].start * k, first[1].stop * k, first[0].start * k
    band = np.asarray(full.crop((x0, y0, x1, y0 + 4000))).astype(np.int16)
    share = (np.abs(band - np.array([247, 247, 245], np.int16)).sum(2) <= 3).mean(1)
    y1 = y0 + next(y for y in range(200, len(share) - 40) if (share[y:y + 40] < 0.05).all())
    return full.crop((0, y0 - 16, x1 + 16, y1 + 12))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        render(ROOT / "docs" / "architecture-flowchart.mmd", work / "main.png")
        render(HERE / "legend.mmd", work / "legend.png")
        full = compose(Image.open(work / "main.png").convert("RGB"), Image.open(work / "legend.png").convert("RGB"))
    bounds = (0, *CUTS, full.height)
    for i in range(3):
        full.crop((0, bounds[i], full.width, bounds[i + 1])).save(FIGURES / f"flowchart_part{i + 1}.png", optimize=True)
    offline_panel(full).save(FIGURES / "flow_offline.png", optimize=True)
    print("ok")
