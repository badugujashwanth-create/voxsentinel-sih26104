#!/usr/bin/env python3
"""Fetch the ECAPA-TDNN speaker-verification checkpoint.

Downloads from a pinned revision of the official SpeechBrain model repository
and verifies every file against a recorded SHA-256 before use. Nothing fetched
here is committed to this repository.

    python ml/scripts/setup_ecapa.py [--force] [--verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.speaker.ecapa import (  # noqa: E402
    DEFAULT_VENDOR_DIR,
    ECAPA_REVISION,
    EXPECTED_SHA256,
)

REPOSITORY = "speechbrain/spkrec-ecapa-voxceleb"
RAW_BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{ECAPA_REVISION}"

#: Not loaded by the model, but the licence and model card belong next to the
#: weights so an auditor finds them where the weights are.
EXTRA_SHA256 = {
    "README.md": "00f58c3cbd7a7510de9374080da0e82a4c4e8f4df567f7338fe6efe108be705a",
}


def sha256_of(path: Path) -> str:
    """Returns the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artefacts() -> dict[str, str]:
    """Every file this script installs, mapped to its expected digest."""
    return {**EXPECTED_SHA256, **EXTRA_SHA256}


def is_installed(vendor_dir: Path = DEFAULT_VENDOR_DIR) -> bool:
    """Reports whether every artefact is present with the expected hash."""
    return all((vendor_dir / name).is_file() and sha256_of(vendor_dir / name) == expected for name, expected in artefacts().items())


def download(vendor_dir: Path = DEFAULT_VENDOR_DIR, force: bool = False) -> Path:
    """Downloads and verifies every artefact, returning the vendor directory."""
    vendor_dir.mkdir(parents=True, exist_ok=True)

    for name, expected in artefacts().items():
        target = vendor_dir / name
        if target.is_file() and not force:
            actual = sha256_of(target)
            if actual == expected:
                print(f"  ok       {name} (already present)")
                continue
            print(f"  stale    {name} (sha256 {actual[:12]}...), re-downloading")

        url = f"{RAW_BASE}/{name}"
        print(f"  fetching {name} <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310 - fixed https URL
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"download failed for {url}: {exc}") from exc

        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {name}\n  expected {expected}\n  got      {actual}\nRefusing to install unverified model files.")
        target.write_bytes(payload)
        print(f"  verified {name} ({len(payload):,} bytes)")

    (vendor_dir / "SOURCE.txt").write_text(
        "ECAPA-TDNN speaker-verification checkpoint\n"
        f"Source:   https://huggingface.co/{REPOSITORY}\n"
        f"Revision: {ECAPA_REVISION}\n"
        "Licence:  Apache-2.0 - see README.md in this directory for the model card\n"
        "Trained:  VoxCeleb 1 + VoxCeleb 2 development sets\n"
        "Output:   192-dimensional speaker embedding, compared with cosine similarity\n"
        "\nFetched by ml/scripts/setup_ecapa.py. Not committed to VoxSentinel.\n",
        encoding="utf-8",
    )
    return vendor_dir


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor-dir", type=Path, default=DEFAULT_VENDOR_DIR, help="where to install the checkpoint")
    parser.add_argument("--force", action="store_true", help="re-download even if files are present and valid")
    parser.add_argument("--verify-only", action="store_true", help="check existing files without downloading")
    args = parser.parse_args()

    if args.verify_only:
        if is_installed(args.vendor_dir):
            print(f"ECAPA artefacts verified in {args.vendor_dir}")
            return 0
        print(f"ECAPA artefacts missing or corrupt in {args.vendor_dir}", file=sys.stderr)
        return 1

    print(f"Installing ECAPA-TDNN from {REPOSITORY}@{ECAPA_REVISION[:12]} into {args.vendor_dir} (~89 MB)")
    download(args.vendor_dir, force=args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
