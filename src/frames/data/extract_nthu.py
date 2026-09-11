from pathlib import Path

import cv2
import numpy as np

from frames.preprocessing.facemesh import EyeRegionTracker

_FRAME_STRIDE = 3
_MAX_FRAMES_PER_GROUP = 150
_EAR_OPEN_THRESHOLD = 0.35
_EAR_CLOSED_THRESHOLD = 0.10


def _parse_image(path: Path) -> tuple[str, str, str, str] | None:
    """Parses '<subjectId>_<glasses>_<scenario>_<frameIdx>_<coarseLabel>.jpg' (e.g. '001_glasses_yawning_1000_drowsy.jpg')."""
    parts = path.stem.split("_")
    if len(parts) != 5:
        return None
    subject_id, glasses, scenario, frame_idx, _coarse_label = parts
    return subject_id, glasses, scenario, frame_idx


def _remove_overlay(image: np.ndarray) -> np.ndarray:
    """Inpaints out the saturated green/blue/red detection boxes this dataset burns into otherwise near-grayscale IR frames."""
    saturation = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
    mask = cv2.dilate((saturation > 30).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def extract_nthu_images(source_dir: Path, output_dir: Path) -> None:
    """Cleans overlay boxes, then labels each eye open/closed by its own EAR rather than trusting the coarse per-clip drowsy label."""
    counts = {"open": 0, "closed": 0}
    group_counts: dict[tuple[str, str, str], int] = {}

    with EyeRegionTracker(static_image_mode=True) as tracker:
        for image_path in sorted(source_dir.rglob("*.jpg")):
            parsed = _parse_image(image_path)
            if parsed is None:
                continue
            subject_id, glasses, scenario, frame_idx = parsed

            group_key = (subject_id, glasses, scenario)
            group_index = group_counts.get(group_key, 0)
            group_counts[group_key] = group_index + 1
            if group_index % _FRAME_STRIDE != 0:
                continue
            if group_index // _FRAME_STRIDE >= _MAX_FRAMES_PER_GROUP:
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            cleaned = _remove_overlay(frame)
            regions = tracker.extract(cleaned)
            if regions is None:
                continue

            for side, eye in regions.eyes.items():
                if eye.ear >= _EAR_OPEN_THRESHOLD:
                    label = "open"
                elif eye.ear <= _EAR_CLOSED_THRESHOLD:
                    label = "closed"
                else:
                    continue

                out_dir = output_dir / label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{subject_id}_{glasses}_{scenario}_{frame_idx}_{side}.jpg"
                if cv2.imwrite(str(out_path), eye.crop):
                    counts[label] += 1

    print(f"\nTotal saved: {counts}")


def main():
    from frames.config import load_config

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    source_dir = raw_dir / "eye" / "nthu_ddd" / "train"
    output_dir = raw_dir / "eye" / "nthu_eye"

    extract_nthu_images(source_dir, output_dir)


if __name__ == "__main__":
    main()
