from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert M2M100 for offline Chinese-to-Japanese TTS translation."
    )
    parser.add_argument("--model", default="facebook/m2m100_418M")
    parser.add_argument(
        "--output",
        default="assets/models/m2m100_418M_ct2",
        help="Project-relative or absolute CTranslate2 model directory.",
    )
    parser.add_argument(
        "--quantization",
        default="int8_float16",
        choices=("int8", "int8_float16", "float16", "float32"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path = output_path.resolve()

    if (output_path / "model.bin").exists():
        print(f"Local translation model already exists: {output_path}")
        return 0
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Refusing to overwrite non-empty directory: {output_path}", file=sys.stderr)
        return 2

    try:
        from ctranslate2.converters import TransformersConverter
        from transformers import AutoTokenizer
    except Exception as exc:
        print(
            "Missing setup dependencies. Run: "
            "python -m pip install 'transformers>=4.48,<5' huggingface-hub sentencepiece ctranslate2",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading and converting {args.model} -> {output_path}")
    converter = TransformersConverter(args.model)
    converter.convert(
        str(output_path),
        quantization=args.quantization,
        force=False,
    )

    tokenizer_path = output_path / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(tokenizer_path)
    metadata = {
        "model_id": args.model,
        "quantization": args.quantization,
        "source_language": "zh",
        "target_language": "ja",
    }
    (output_path / "desktop_pet_translation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del tokenizer
    del converter
    gc.collect()
    print("Local translation model is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
