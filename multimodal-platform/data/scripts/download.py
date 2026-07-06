"""Deterministic dataset fetcher.

Downloads every dataset in data/manifest.yaml, verifies SHA256, and unpacks
into data/raw/<name>/. On first run with a placeholder checksum, it prints
the computed hash so you can pin it in the manifest.

Usage:
    uv run python data/scripts/download.py [name ...]
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def sha256_of(path: Path) -> str:
    """Stream a file's SHA256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, spec: dict[str, str]) -> None:
    """Download, verify, and unpack one dataset."""
    url = spec["url"]
    if not url.startswith("http") or "download.html" in url or spec["sha256"] == "manual-download":
        print(f"[{name}] manual step required: {url}")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / f"{name}.zip"
    if not archive.exists():
        print(f"[{name}] downloading {url}")
        urllib.request.urlretrieve(url, archive)  # noqa: S310
    digest = sha256_of(archive)
    expected = spec["sha256"]
    if expected.startswith("REPLACE"):
        print(f"[{name}] pin this checksum in data/manifest.yaml: {digest}")
    elif digest != expected:
        raise SystemExit(f"[{name}] checksum mismatch: {digest} != {expected}")
    out = RAW / name
    if not out.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    print(f"[{name}] ready at {out}")


def main() -> None:
    """Entry point."""
    manifest = yaml.safe_load((ROOT / "data" / "manifest.yaml").read_text())["datasets"]
    names = sys.argv[1:] or list(manifest)
    for name in names:
        fetch(name, manifest[name])


if __name__ == "__main__":
    main()
