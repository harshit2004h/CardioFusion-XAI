"""Ensure the selected PanEcho notebook reloads the safe DataLoader module."""
from __future__ import annotations

import json
from pathlib import Path


PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks/03_training/echo/04_panecho_selected_finetuning_training.ipynb"
)
OLD = "payload=torch.load(OUTPUT_CHECKPOINT,map_location='cpu',weights_only=True)\n"
NEW = (
    "# Reload the dataset module so this export uses pin_memory=False, even if\n"
    "# the notebook kernel imported an older version before the layout update.\n"
    "import importlib\n"
    "import src.datasets.panecho_echonet_dataset as panecho_dataset\n"
    "importlib.reload(panecho_dataset)\n"
    "build_panecho_loaders=panecho_dataset.build_panecho_loaders\n"
    "payload=torch.load(OUTPUT_CHECKPOINT,map_location='cpu',weights_only=True)\n"
)


def main() -> None:
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if OLD in source and "importlib.reload(panecho_dataset)" not in source:
            cell["source"] = source.replace(OLD, NEW, 1).splitlines(keepends=True)
            changed = True
    if not changed:
        raise RuntimeError("Expected final PanEcho export cell was not found or was already repaired.")
    PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {PATH.name}.")


if __name__ == "__main__":
    main()
