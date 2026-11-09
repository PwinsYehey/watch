import argparse
import os
import sys
import warnings

from watch.camera import camera_source
from watch.config import MODEL_NAMES

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

TRAINING_ACTIONS = {"fetch-data", "prepare-data", "train", "finetune", "evaluate", "pipeline"}


def _names(target: str) -> tuple[str, ...]:
    return MODEL_NAMES if target == "all" else (target,)


def check(_):
    import glob
    import platform

    import cv2
    import tensorflow as tf

    from watch.config import model_path

    print(f"python {platform.python_version()} on {sys.platform}")
    print(f"tensorflow {tf.__version__}, GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    for name in MODEL_NAMES:
        print(f"{name} model: {'found' if model_path(name).exists() else 'MISSING'}")
    if sys.platform == "linux" and not glob.glob("/dev/video*"):
        available = False
    else:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        available = camera.isOpened()
        camera.release()
    print(f"camera 0: {'available' if available else 'not available'}")


def fetch_data(args):
    from watch.data.fetch import fetch

    fetch(args.only)


def prepare_data(_):
    from watch.data.extract import extract_nthu, extract_yawdd
    from watch.data.split import make_split

    extract_nthu()
    extract_yawdd()
    for name in MODEL_NAMES:
        make_split(name)


def train(args):
    from watch.train import sweep, train

    for name in _names(args.model):
        if args.sweep:
            sweep(name)
        else:
            train(name, force=args.force)


def record(args):
    from watch.record import record

    try:
        record(args.camera, args.subject, args.take)
    except ValueError as error:
        sys.exit(str(error))


def finetune(args):
    from watch.finetune import finetune

    try:
        finetune(force=args.force)
    except ValueError as error:
        sys.exit(str(error))


def evaluate(args):
    from watch.evaluate import evaluate

    for name in _names(args.model):
        evaluate(name)


def benchmark(args):
    from watch.evaluate import benchmark

    for name in _names(args.model):
        benchmark(name)


def export(_):
    from watch.export import export_all

    export_all()


def live(args):
    from watch.live import LiveMonitor

    LiveMonitor(args.camera, args.stream, args.allow_shutdown).run()


def web(args):
    import functools
    import http.server

    from watch.config import load_config

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(load_config()["paths"]["web_dir"]))
    print(f"Serving the web demo at http://localhost:{args.port} (Ctrl+C to stop)")
    with http.server.ThreadingHTTPServer(("", args.port), handler) as server:
        server.serve_forever()


def predict(args):
    from pathlib import Path

    from watch.predict import predict

    predict(Path(args.image))


def test(args):
    import pytest

    from watch.config import PROJECT_ROOT

    sys.exit(pytest.main([str(PROJECT_ROOT / "tests"), *args.extra]))


def pipeline(_):
    fetch_data(argparse.Namespace(only=None))
    prepare_data(None)
    train(argparse.Namespace(model="all", sweep=False, force=False))
    evaluate(argparse.Namespace(model="all"))
    export(None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch", description="WATCH - Wakefulness Assessment through Tracking of Cues & Head")
    actions = parser.add_subparsers(dest="action", required=True)
    model_choices = [*MODEL_NAMES, "all"]

    actions.add_parser("check", help="report Python, TensorFlow, GPU, models and camera").set_defaults(run=check)
    sub = actions.add_parser("fetch-data", help="download MRL, CEW, NTHU-DDD and YawDD into data/raw")
    sub.add_argument("--only", nargs="+", choices=["mrl_eye", "cew", "nthu_ddd", "yawdd"])
    sub.set_defaults(run=fetch_data)
    actions.add_parser("prepare-data", help="extract NTHU/YawDD crops and rebuild the splits").set_defaults(run=prepare_data)
    sub = actions.add_parser("train", help="train a model, keeping it only if it beats the current one")
    sub.add_argument("model", choices=model_choices)
    sub.add_argument("--sweep", action="store_true", help="run the hyperparameter grid search instead")
    sub.add_argument("--force", action="store_true", help="replace the current model even if the new one scores lower")
    sub.set_defaults(run=train)
    sub = actions.add_parser("record", help="record labelled eye crops on this camera, cued by the buzzer")
    sub.add_argument("subject", help="who is recording, letters and digits only")
    sub.add_argument("take", help="a name for this take, such as shades1 or noshades1")
    sub.add_argument("--camera", type=camera_source, default=0, help="camera index, or 'pi' for a Raspberry Pi camera module")
    sub.set_defaults(run=record)
    sub = actions.add_parser("finetune", help="fine-tune the eye model on recorded crops, keeping it only if held-out takes improve")
    sub.add_argument("--force", action="store_true", help="replace the current model even if the held-out takes score lower")
    sub.set_defaults(run=finetune)
    for action, run, help_text in (("evaluate", evaluate, "test-set metrics and plots"), ("benchmark", benchmark, "inference speed and size")):
        sub = actions.add_parser(action, help=help_text)
        sub.add_argument("model", nargs="?", default="all", choices=model_choices)
        sub.set_defaults(run=run)
    actions.add_parser("export", help="write TF.js models, web/config.js and TFLite models").set_defaults(run=export)
    sub = actions.add_parser("live", help="webcam alertness monitor")
    sub.add_argument("--camera", type=camera_source, default=0, help="camera index, or 'pi' for a Raspberry Pi camera module")
    sub.add_argument("--stream", type=int, nargs="?", const=8766, metavar="PORT", help="show the monitor in a browser on this port instead of a window")
    sub.add_argument("--allow-shutdown", action="store_true", help="with --stream, add a button to the page that powers this machine off")
    sub.set_defaults(run=live)
    sub = actions.add_parser("web", help="serve the browser demo")
    sub.add_argument("--port", type=int, default=8765)
    sub.set_defaults(run=web)
    sub = actions.add_parser("predict", help="eye, mouth and head state for one photo")
    sub.add_argument("image")
    sub.set_defaults(run=predict)
    actions.add_parser("test", help="run the unit tests, passing extra arguments to pytest").set_defaults(run=test)
    actions.add_parser("pipeline", help="fetch-data, prepare-data, train, evaluate and export").set_defaults(run=pipeline)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, args.extra = parser.parse_known_args(argv)
    if args.extra and args.action != "test":
        parser.error(f"unrecognized arguments: {' '.join(args.extra)}")
    if sys.platform == "win32" and args.action in TRAINING_ACTIONS:
        sys.exit(f"'{args.action}' runs on the Linux training machine only.")
    try:
        args.run(args)
    except (FileNotFoundError, FileExistsError, ConnectionError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
