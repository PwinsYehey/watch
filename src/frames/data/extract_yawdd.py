from pathlib import Path

import cv2

from frames.preprocessing.facemesh import EyeRegionTracker

_CATEGORY_TO_LABEL = {"normal": "no_yawn", "talking": "no_yawn", "yawning": "yawn"}
_FRAME_STRIDE = {"normal": 10, "talking": 10, "yawning": 3}
_MAX_FRAMES_PER_VIDEO = 40
_YAWN_MAR_THRESHOLD = 0.6


def _parse_video(path: Path) -> tuple[str, str] | None:
    """Parses '<subjectNum>-<condition>-<category>.avi' into (subject_num, category), or None if malformed."""
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    return parts[0], parts[-1].lower()


def extract_mirror_videos(mirror_dir: Path, output_dir: Path) -> None:
    """Samples frames from each YawDD Mirror video, crops the mouth region, and saves labeled images by real subject id."""
    gender_folders = {"Female_mirror": "female", "Male_mirror Avi Videos": "male"}
    counts = {"yawn": 0, "no_yawn": 0}

    with EyeRegionTracker(static_image_mode=True) as tracker:
        for folder_name, gender in gender_folders.items():
            video_dir = mirror_dir / folder_name
            for video_path in sorted(video_dir.glob("*.avi")):
                parsed = _parse_video(video_path)
                if parsed is None:
                    print(f"skipping malformed filename: {video_path.name}")
                    continue

                subject_num, category = parsed
                label = _CATEGORY_TO_LABEL.get(category)
                if label is None:
                    print(f"skipping unknown category '{category}': {video_path.name}")
                    continue

                subject_id = f"{gender}{subject_num}"
                stride = _FRAME_STRIDE[category]
                out_dir = output_dir / label
                out_dir.mkdir(parents=True, exist_ok=True)

                cap = cv2.VideoCapture(str(video_path))
                frame_idx = 0
                saved = 0
                while saved < _MAX_FRAMES_PER_VIDEO:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % stride == 0:
                        regions = tracker.extract(frame)
                        # for "yawning" videos, only the wide-open-mouth frames actually show a yawn
                        if regions is not None and (category != "yawning" or regions.mouth.mar >= _YAWN_MAR_THRESHOLD):
                            out_path = out_dir / f"{subject_id}_{category}_{frame_idx:05d}.jpg"
                            if cv2.imwrite(str(out_path), regions.mouth.crop):
                                saved += 1
                                counts[label] += 1
                    frame_idx += 1
                cap.release()

                print(f"{video_path.name}: saved {saved} frames as {label}")

    print(f"\nTotal saved: {counts}")


def main():
    from frames.config import load_config

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    mirror_dir = raw_dir / "yawdd_videos" / "Mirror"
    output_dir = raw_dir / "yawdd"

    extract_mirror_videos(mirror_dir, output_dir)


if __name__ == "__main__":
    main()
