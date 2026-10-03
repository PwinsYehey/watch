import pytest

from watch.__main__ import build_parser
from watch.config import load_config
from watch.export import web_config_js

ACTIONS = ["check", "fetch-data", "prepare-data", "train", "evaluate", "benchmark", "export", "live", "web", "predict", "test", "pipeline"]


@pytest.mark.parametrize("action", ACTIONS)
def test_every_action_has_help(action, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([action, "--help"])
    assert exit_info.value.code == 0


def test_web_config_matches_yaml():
    web_config = load_config()["paths"]["web_dir"] / "config.js"
    assert web_config.read_text() == web_config_js(), "run ./scripts/export.sh to regenerate web/config.js"
