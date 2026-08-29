from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from frames.config import load_config
from frames.models.build_model import load_trained_model


def measure_inference_time(model, input_shape: tuple, warmup_runs: int = 10, test_runs: int = 100):
    dummy_input = np.random.rand(1, *input_shape).astype(np.float32)

    for _ in range(warmup_runs):
        model(dummy_input, training=False)

    times = []
    for _ in range(test_runs):
        start = tf.timestamp()
        model(dummy_input, training=False)
        times.append(float(tf.timestamp() - start) * 1000)

    return float(np.mean(times)), float(np.std(times))


def main():
    config = load_config()
    image_size = tuple(config["preprocessing"]["image_size"])
    input_shape = image_size + (3,)

    model = load_trained_model(config)

    total_params = model.count_params()
    size_mb = (total_params * 4) / (1024 ** 2)

    avg_ms, std_ms = measure_inference_time(model, input_shape)
    fps = 1000 / avg_ms

    # a live frame runs the model twice (one per eye), so this is the realistic per-frame cost
    per_frame_ms = avg_ms * 2
    per_frame_fps = 1000 / per_frame_ms

    result = {
        "total_params": total_params,
        "size_mb": round(size_mb, 2),
        "single_inference_ms": round(avg_ms, 2),
        "single_inference_std_ms": round(std_ms, 2),
        "single_inference_fps": round(fps, 2),
        "per_frame_ms_both_eyes": round(per_frame_ms, 2),
        "per_frame_fps_both_eyes": round(per_frame_fps, 2),
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(results_dir / "benchmark.csv", index=False)
    print(f"\nSaved to: {results_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
