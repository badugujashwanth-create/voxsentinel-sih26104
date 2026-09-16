#!/usr/bin/env python3
"""Fetch the AASIST model definition and pretrained checkpoint.

Downloads from a pinned commit of the official NAVER/Clova AI repository and
verifies every file against a recorded SHA-256 before use. Nothing fetched here
is committed to this repository.

    python ml/scripts/setup_aasist.py [--force] [--verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Pinned commit of https://github.com/clovaai/aasist (MIT licence).
AASIST_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
AASIST_RAW_BASE = f"https://raw.githubusercontent.com/clovaai/aasist/{AASIST_COMMIT}"

VENDOR_DIR = Path(__file__).resolve().parents[1] / "spoof" / "vendor" / "aasist"

#: (remote path, local name, sha256) for every file we pull.
ARTEFACTS: tuple[tuple[str, str, str], ...] = (
    ("models/AASIST.py", "AASIST.py", "9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50"),
    ("models/weights/AASIST.pth", "AASIST.pth", "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"),
    ("LICENSE", "LICENSE", "da2e79b8592d166ef505224300968b80ebe1e4c217c43b94a5ec627d81cd4142"),
)


def sha256_of(path: Path) -> str:
    """Returns the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_installed() -> bool:
    """Reports whether every artefact is present with the expected hash."""
    return all((VENDOR_DIR / name).is_file() and sha256_of(VENDOR_DIR / name) == expected for _, name, expected in ARTEFACTS)


def download(force: bool = False) -> Path:
    """Downloads and verifies every artefact, returning the vendor directory."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    for remote, name, expected in ARTEFACTS:
        target = VENDOR_DIR / name
        if target.is_file() and not force:
            actual = sha256_of(target)
            if actual == expected:
                print(f"  ok       {name} (already present)")
                continue
            print(f"  stale    {name} (sha256 {actual[:12]}...), re-downloading")

        url = f"{AASIST_RAW_BASE}/{remote}"
        print(f"  fetching {name} <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"download failed for {url}: {exc}") from exc

        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {name}\n  expected {expected}\n  got      {actual}\nRefusing to install unverified model files.")
        target.write_bytes(payload)
        print(f"  verified {name} ({len(payload):,} bytes)")

    (VENDOR_DIR / "SOURCE.txt").write_text(
        "AASIST model definition and pretrained checkpoint\n"
        "Source:  https://github.com/clovaai/aasist\n"
        f"Commit:  {AASIST_COMMIT}\n"
        "Licence: MIT (c) 2021-present NAVER Corp. - see LICENSE in this directory\n"
        "Trained: ASVspoof 2019 Logical Access (LA) train partition\n"
        "\nFetched by ml/scripts/setup_aasist.py. Not committed to VoxSentinel.\n",
        encoding="utf-8",
    )
    return VENDOR_DIR


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if files are present and valid")
    parser.add_argument("--verify-only", action="store_true", help="check existing files without downloading")
    args = parser.parse_args()

    if args.verify_only:
        if is_installed():
            print(f"AASIST artefacts verified in {VENDOR_DIR}")
            return 0
        print(f"AASIST artefacts missing or corrupt in {VENDOR_DIR}", file=sys.stderr)
        return 1

    print(f"Installing AASIST from clovaai/aasist@{AASIST_COMMIT[:12]} into {VENDOR_DIR}")
    download(force=args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
