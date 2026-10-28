"""Apply this project's cooperative cancellation patch to the local TTS runtime."""
import subprocess
import os
import shutil
from pathlib import Path


def prepare() -> None:
    root = Path(__file__).resolve().parent.parent
    runtime = root / "tools" / "gpt_sovits"
    command = ["git", "-C", str(runtime), "apply", "--check"]
    for name in ("gpt_sovits_windows_compat.patch", "gpt_sovits_cancel.patch"):
        patch = root / "scripts" / name
        if subprocess.run(command + [str(patch)], capture_output=True).returncode == 0:
            subprocess.run(["git", "-C", str(runtime), "apply", str(patch)], check=True)
            print(f"Installed {name}.")
        elif subprocess.run(command + ["--reverse", str(patch)], capture_output=True).returncode == 0:
            print(f"Already installed: {name}.")
        else:
            raise RuntimeError(f"GPT-SoVITS source does not match {name}; install the documented source revision.")
    models = root / "assets" / "models" / "gpt-sovits-pretrained"
    for source in models.rglob("*"):
        if not source.is_file() or source.suffix not in {".bin", ".ckpt", ".json"}:
            continue
        with source.open("rb") as f:
            if f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError("Speech models are Git LFS pointers; download them with git lfs pull first.")
        target = runtime / "GPT_SoVITS" / "pretrained_models" / source.relative_to(models)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


if __name__ == "__main__":
    prepare()
