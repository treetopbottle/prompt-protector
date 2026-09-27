"""Command-line interface. Needs the `model` extra: `uv sync --extra model`."""

import argparse
from pathlib import Path

from .analyzer.probe import DEFAULT_PROBE


def download_model(probe: Path) -> None:
    """Download the model revision the probe was trained on. The only command that uses the network."""
    import numpy as np
    from huggingface_hub import snapshot_download

    data = np.load(probe)
    model_id, revision = str(data["model"]), str(data["revision"])
    path = snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print(f"{model_id} at {revision} is in {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prompt-protector")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-model", help="download the model the Role Analyzer uses")
    download.add_argument("--probe", type=Path, default=DEFAULT_PROBE, help="probe file whose model to download")
    args = parser.parse_args(argv)

    if args.command == "download-model":
        download_model(args.probe)


if __name__ == "__main__":
    main()
