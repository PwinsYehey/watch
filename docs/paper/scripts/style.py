import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "docs" / "paper" / "figures"
HELMET_DIR = Path(os.environ.get("WATCH_HELMET_DIR", ROOT / "data" / "raw" / "helmet" / "head_turn"))
FONT_DIR = Path(os.environ.get("WATCH_FONT_DIR", "/mnt/c/Windows/Fonts"))


def serif_fonts():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    files = [FONT_DIR / f for f in ("times.ttf", "timesbd.ttf", "timesi.ttf")]
    if all(f.exists() for f in files):
        for f in files:
            font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"
