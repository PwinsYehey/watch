import glob
import sys
import time
from pathlib import Path

import cv2
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FIGURES, ROOT

video = ROOT / "data/raw/yawn/yawdd_videos/Mirror/Male_mirror Avi Videos/1-MaleNoGlasses-Yawning.avi"
photo, shot = FIGURES / "_web_demo_input.jpg", FIGURES / "_web_demo_full.png"
capture = cv2.VideoCapture(str(video))
capture.set(cv2.CAP_PROP_POS_FRAMES, 190)
cv2.imwrite(str(photo), capture.read()[1])
capture.release()

browsers = glob.glob(str(Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=browsers[0] if browsers else None, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
    page.goto(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765", wait_until="networkidle")
    page.set_input_files("#imageUpload", str(photo))
    time.sleep(25)
    page.screenshot(path=str(shot), full_page=True)
    browser.close()

image = Image.open(shot)
image.crop((0, 0, image.width, 1350)).save(FIGURES / "web_demo.png")
photo.unlink()
shot.unlink()
