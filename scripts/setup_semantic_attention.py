from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a quantized ONNX Chinese embedding model for semantic attention."
    )
    parser.add_argument("--model", default="Xenova/bge-small-zh-v1.5")
    parser.add_argument(
        "--output",
        default="assets/models/bge-small-zh-v1.5-onnx",
        help="Project-relative or absolute model directory.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path = output_path.resolve()
    model_file = output_path / "onnx" / "model_quantized.onnx"

    if model_file.is_file():
        print(f"Semantic attention model already exists: {output_path}")
        return 0
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Refusing to overwrite non-empty directory: {output_path}", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print("Missing dependency. Run: python -m pip install huggingface-hub", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model} -> {output_path}")
    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_path),
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "onnx/model_quantized.onnx",
        ],
    )
    if not model_file.is_file():
        print(f"Downloaded repository does not contain {model_file.name}", file=sys.stderr)
        return 4

    metadata = {
        "model_id": args.model,
        "runtime": "onnxruntime",
        "model_file": "onnx/model_quantized.onnx",
        "pooling": "cls",
    }
    (output_path / "desktop_pet_semantic_attention.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Local semantic attention model is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
