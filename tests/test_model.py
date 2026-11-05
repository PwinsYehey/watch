import numpy as np

from watch.model import Classifier


def _classifier(row):
    classifier = object.__new__(Classifier)
    classifier.class_names = ["eyes_closed", "eyes_opened", "blocked"]
    classifier.positive_index, classifier.unreadable_index, classifier.unreadable_threshold = 0, 2, 0.9
    classifier._prepare = lambda crop: np.zeros((1,), np.float32)
    classifier._infer = lambda batch: type("T", (), {"numpy": lambda self: np.array([row] * len(batch))})()
    return classifier


def test_blocked_only_when_sure():
    assert _classifier([0.01, 0.04, 0.95]).predict(None)[0] == "blocked"


def test_unsure_block_falls_back_to_open_or_closed():
    label, confidence, closed = _classifier([0.30, 0.10, 0.60]).predict(None)
    assert label == "eyes_closed" and np.isclose(confidence, 0.75) and np.isclose(closed, 0.75)
