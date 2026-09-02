import json
from pathlib import Path


def test_notebook_has_four_executable_sections():
    notebook_path = Path(__file__).parents[1] / "run" / "run.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    assert len(cells) == 4
    assert all(cell["cell_type"] == "code" for cell in cells)
    sources = ["".join(cell["source"]) for cell in cells]
    for source in sources:
        compile(source, str(notebook_path), "exec")
    assert "validate(config)" in sources[0]
    assert "train_seeds(" in sources[1]
    assert "evaluate_widths(" in sources[2]
    assert "save_gif(" in sources[3]
    assert "notebook_viewer(" in sources[3]
    assert "INFERENCE_STEPS =" in sources[3]
    assert "INFERENCE_STEPS = trained_config.training.rollout_steps" not in sources[3]
