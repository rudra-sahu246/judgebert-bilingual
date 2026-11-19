"""Package a completed experiment with its pinned tokenizer and integrity manifest."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from huggingface_hub import snapshot_download

from lmp import write_json


def package(root):
    artifact = json.loads((root / "model.json").read_text(encoding="utf-8"))
    snapshot = Path(snapshot_download(artifact["encoder"], revision=artifact["revision"],
                                      local_files_only=True))
    configs = root / "encoder-config"
    configs.mkdir(exist_ok=True)
    for name in ("config.json", "modules.json", "sentence_bert_config.json",
                 "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        source = snapshot / name
        if source.is_file():
            shutil.copyfile(source, configs / name)
    if not (configs / "tokenizer_config.json").is_file():
        raise ValueError("The cached encoder is missing tokenizer configuration")
    write_json(root / "environment.json", {"python": platform.python_version(),
                                          "platform": platform.platform(),
                                          "dependencies": {
        name: importlib.metadata.version(name) for name in (
            "torch", "scikit-learn", "scipy", "numpy", "sentence-transformers",
            "transformers", "tokenizers", "fastapi", "uvicorn")
    }})
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and not {"presentation-build", "paper-build"} & set(path.relative_to(root).parts)
                   and path.name not in ("run-manifest.json", "research-run.zip"))
    manifest = {str(path.relative_to(root).as_posix()): sha256(path.read_bytes()).hexdigest()
                for path in files}
    write_json(root / "run-manifest.json", {
        "synthetic": True, "encoder_revision": artifact["revision"], "files": manifest})
    with ZipFile(root / "research-run.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in [*files, root / "run-manifest.json"]:
            archive.write(path, arcname=path.relative_to(root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/synthetic"))
    package(parser.parse_args().out)
