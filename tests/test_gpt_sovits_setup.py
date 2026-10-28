import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def installer(root):
    source = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gpt_sovits.py"
    spec = importlib.util.spec_from_file_location("tts_setup_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(root / "scripts" / "prepare_gpt_sovits.py")
    return module


def test_model_setup_restores_missing_files_and_preserves_existing_models():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "assets/models/gpt-sovits-pretrained/base"
        source.mkdir(parents=True)
        (source / "pytorch_model.bin").write_bytes(b"model weights")
        (source / "config.json").write_text("{}", encoding="utf-8")
        module = installer(root)
        target = root / "tools/gpt_sovits/GPT_SoVITS/pretrained_models/base"
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            module.prepare()
            assert (target / "pytorch_model.bin").read_bytes() == b"model weights"
            assert (target / "config.json").read_text(encoding="utf-8") == "{}"
            (target / "pytorch_model.bin").unlink()
            (target / "pytorch_model.bin").write_bytes(b"existing custom model")
            module.prepare()
            assert (target / "pytorch_model.bin").read_bytes() == b"existing custom model"


def test_model_setup_rejects_lfs_pointers_before_installing_them():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "assets/models/gpt-sovits-pretrained/base"
        source.mkdir(parents=True)
        (source / "pytorch_model.bin").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        module = installer(root)
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            try:
                module.prepare()
            except RuntimeError as exc:
                assert "git lfs pull" in str(exc)
            else:
                raise AssertionError("LFS pointer installed as model weights")
        assert not (root / "tools/gpt_sovits/GPT_SoVITS/pretrained_models/base/pytorch_model.bin").exists()
