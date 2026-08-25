import json
from pathlib import Path

from trdizin_topic_pipeline.config import load_config, project_root, resolve_path


def test_config_paths_resolve_from_project_root():
    config = load_config(project_root() / "configs" / "final_50k.json")
    assert resolve_path(config, "output_root") == project_root() / "outputs" / "final_50k"
    assert Path(config["_project_root"]) == project_root()
