import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

FACE_OVAL = sorted({i for pair in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in pair})
MASK_IDS = [6, 120, 349, 50, 280, 187, 411, 214, 434, 135, 364, 150, 379, 152, 148, 377]
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10


def points(landmarks, w, h):
    return np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)


def feather(mask, size):
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), size)[..., None]


def visor_mask(shape, pts):
    h, w = shape[:2]
    oval = pts[FACE_OVAL]
    center = oval.mean(axis=0)
    oval = center + (oval - center) * 1.12
    up = pts[FOREHEAD] - pts[CHIN]
    up /= np.linalg.norm(up) + 1e-6
    top = center + up * np.linalg.norm(pts[FOREHEAD] - pts[CHIN]) * 0.35
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [cv2.convexHull(np.vstack([oval, top]).astype(np.int32))], 1)
    ys, xs = np.mgrid[0:h, 0:w]
    edge = pts[NOSE_TIP] - up * np.linalg.norm(pts[FOREHEAD] - pts[CHIN]) * 0.04
    above = ((xs - edge[0]) * up[0] + (ys - edge[1]) * up[1]) > 0
    return region * above


def mask_mask(shape, pts):
    h, w = shape[:2]
    hull = pts[MASK_IDS]
    center = hull.mean(axis=0)
    hull = center + (hull - center) * 1.1
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [cv2.convexHull(hull.astype(np.int32))], 1)
    return region


def paint_visor(frame, region, style, rng):
    alpha = feather(region, 3)
    base = frame.astype(np.float32)
    if style == "mirror":
        reflection = cv2.GaussianBlur(cv2.flip(frame, 1), (0, 0), 6).astype(np.float32)
        lens = 0.25 * base * 0.3 + 0.75 * reflection * 0.9
    else:
        lens = base * 0.12 + np.array([18, 16, 12], np.float32)
    h, w = frame.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    sheen = np.clip(1 - np.abs((xs + ys * 0.6) % 260 - 130) / 22, 0, 1)[..., None] * 60 * rng.uniform(0.6, 1.0)
    lens = np.clip(lens + sheen, 0, 255)
    return base * (1 - alpha) + lens * alpha


def paint_mask(frame, region, texture):
    alpha = feather(region, 2)
    shade = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    shade = (shade / (shade.max() + 1e-6))[..., None]
    rubber = np.array([62, 72, 66], np.float32) * (0.55 + 0.6 * shade) + texture
    return frame * (1 - alpha) + np.clip(rubber, 0, 255) * alpha


def process(src, dst, style, with_mask, seed):
    rng = np.random.default_rng(seed)
    capture = cv2.VideoCapture(str(src))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    texture = rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    smoothed, covered, total = None, 0, 0
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1) as mesh:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            total += 1
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                pts = points(result.multi_face_landmarks[0].landmark, w, h)
                smoothed = pts if smoothed is None else 0.5 * smoothed + 0.5 * pts
            out = frame.astype(np.float32)
            if smoothed is not None:
                covered += 1
                out = paint_visor(frame, visor_mask(frame.shape, smoothed), style, rng)
                if with_mask:
                    out = paint_mask(out, mask_mask(frame.shape, smoothed), texture)
            writer.write(np.clip(out, 0, 255).astype(np.uint8))
    capture.release()
    writer.release()
    return covered, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--style", choices=["dark", "mirror"], default="dark")
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, video in enumerate(args.videos):
        dst = args.out_dir / f"{video.stem}.avi"
        if dst.exists():
            continue
        partial = dst.with_suffix(".part.avi")
        covered, total = process(video, partial, args.style, args.mask, i)
        partial.rename(dst)
        print(f"done {video.name}: covered {covered}/{total} frames", flush=True)
