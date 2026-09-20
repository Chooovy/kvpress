# SPDX-FileCopyrightText: 2026 Xintong Yang
# SPDX-License-Identifier: Apache-2.0

"""Create a portable run directory after checking the published FA2 inputs."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_prepared(root):
    """Refuse to mix changed inputs, source or packages into an existing run."""
    record = json.loads((root / "preflight.json").read_text())
    for package, expected in record["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, found {actual}")
    for name, expected in record["files"].items():
        if sha(name) != expected:
            raise ValueError(f"Changed since preparation: {name}; prepare a fresh run directory")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="New, empty run directory")
    parser.add_argument("--prompts", type=Path, required=True, help="Published prompts256k.json")
    for size in ("4b", "8b"):
        parser.add_argument(f"--model-{size}", type=Path, required=True)
        parser.add_argument(f"--checkpoint-{size}", type=Path, required=True, help="IndexMem final.pt")
    args = parser.parse_args()
    reference = json.loads((CODE / "reference.json").read_text())
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("--root must be empty; previous results are never overwritten")

    files = {}

    def check(path, expected):
        path = path.resolve()
        digest = sha(path)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
        files[str(path)] = digest

    for package, expected in reference["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, found {actual}")
    check(args.prompts, reference["prompts_sha256"])
    models = {}
    for size in ("4b", "8b"):
        model = getattr(args, f"model_{size}").resolve()
        models[size] = str(model)
        for name, item in reference["models"][size]["files"].items():
            check(model / name, item["sha256"])
        check(getattr(args, f"checkpoint_{size}"), reference["indexmem_checkpoints"][size]["sha256"])

    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    assets.mkdir()
    (assets / "prompts.json").symlink_to(args.prompts.resolve())
    for size in ("4b", "8b"):
        folder = assets / f"indexmem{size}"
        folder.mkdir()
        (folder / "final.pt").symlink_to(getattr(args, f"checkpoint_{size}").resolve())
    write_json(root / "config.json", {"models": models})
    paths = sorted((REPO / "kvpress").rglob("*.py")) + sorted(CODE.glob("*.py"))
    paths += [CODE / "reference.json", root / "config.json", assets / "prompts.json"]
    paths += [assets / f"indexmem{size}" / "final.pt" for size in ("4b", "8b")]
    files.update({str(path): sha(path) for path in paths})
    write_json(
        root / "preflight.json",
        {
            "packages": reference["packages"],
            "files": files,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "runtime_diff": subprocess.check_output(["git", "diff", "HEAD", "--", "kvpress"], cwd=REPO, text=True),
            "protocol": {"documents": 3, "warmups": 2, "repeats": 5, "steps": 256},
        },
    )
    print(f"Prepared {root}; checked weights, token streams, FA2 packages and source hashes")


if __name__ == "__main__":
    main()
