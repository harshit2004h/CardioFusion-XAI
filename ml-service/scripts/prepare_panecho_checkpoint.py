"""Put the PanEcho weights where the ML service expects them.

The released ``panecho.pt`` (~505 MB) also stores the Adam optimizer state,
which inference never needs. This script:

  1. accepts ``panecho.pt`` *or* the ``panecho.pt.zip`` you downloaded,
  2. keeps only the model weights (~170 MB),
  3. verifies they load into the project's PanEcho implementation,
  4. writes ``checkpoints/echo/panecho/pretrained/v1/model.pt``.

Run from ml-service/:

    python scripts/prepare_panecho_checkpoint.py --source ~/Downloads/panecho.pt.zip
    # or fetch the official release directly
    python scripts/prepare_panecho_checkpoint.py --download
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ML_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_SERVICE_ROOT))

import torch  # noqa: E402

from src.models.echo.panecho import (  # noqa: E402
    extract_state_dict,
    heads_in_state_dict,
    load_panecho,
    load_panecho_payload,
)

OFFICIAL_URL = "https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt"
DEFAULT_OUTPUT = ML_SERVICE_ROOT / "checkpoints" / "echo" / "panecho" / "pretrained" / "v1" / "model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Path to panecho.pt or panecho.pt.zip")
    source.add_argument("--download", action="store_true", help=f"Download {OFFICIAL_URL}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-optimizer", action="store_true", help="Copy the full file unchanged")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.download:
        raw = args.output.parent / "panecho_release_full.pt"
        if not raw.exists():
            print(f"Downloading {OFFICIAL_URL} -> {raw}")
            torch.hub.download_url_to_file(OFFICIAL_URL, str(raw), progress=True)
        src = raw
    else:
        src = args.source.expanduser()

    payload = load_panecho_payload(src)
    state = extract_state_dict(payload)
    heads = heads_in_state_dict(state)
    print(f"Found {len(state)} tensors, {len(heads)} task heads in {payload['_resolved_path']}")

    if args.keep_optimizer:
        payload.pop("_resolved_path", None)
        torch.save(payload, args.output)
    else:
        torch.save(
            {
                "weights": state,
                "source": "PanEcho v1.0 (CarDS-Yale/PanEcho release)",
                "license": "CC BY-NC-SA 4.0 (non-commercial)",
            },
            args.output,
        )

    model = load_panecho(args.output)  # raises if anything is missing/unexpected
    with torch.inference_mode():
        ef = model(torch.zeros(1, 3, 16, 224, 224))["EF"].item()
    size_mb = args.output.stat().st_size / 1e6
    print(f"OK: {args.output} ({size_mb:.0f} MB), {len(model.tasks)} heads, EF on a blank clip = {ef:.1f}%")


if __name__ == "__main__":
    main()
