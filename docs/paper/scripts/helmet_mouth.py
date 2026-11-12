import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HELMET_DIR, ROOT, serif_fonts
import csv, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(ROOT / "experiments" / "hidden_face"))
from helmet_frames import session_of, resize
from watch.face import FaceTracker
from watch.live import remove_glare
from watch.model import Classifier

root, out = Path(sys.argv[1]), Path(sys.argv[2])
yawn = Classifier("yawn")
images = sorted((session_of(p), p) for p in root.rglob("*.jpg"))
with out.open("w", newline="") as f, FaceTracker(static_image_mode=True) as tracker:
    w = csv.writer(f)
    w.writerow(["folder", "session", "index", "mouth_label", "mouth_conf", "mar"])
    for n, ((session, index), path) in enumerate(images):
        regions = tracker.extract(remove_glare(resize(cv2.imread(str(path)))))
        if regions and regions.mouth is not None:
            label, conf, _ = yawn.predict(regions.mouth.crop)
            w.writerow([path.parent.name, session, index, label, conf, regions.mouth.ratio])
        else:
            w.writerow([path.parent.name, session, index, "", "", ""])
        if n % 1000 == 0:
            print(n, len(images), flush=True)
print("done")
