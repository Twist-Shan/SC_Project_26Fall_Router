"""Download the Arena training battles at the revision used in the paper."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


REVISION = "e5a2158d14d30003ef7ec8d4756c86359715dbc2"
URL = f"https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/{REVISION}/train.csv"
SHA256 = "0692154aaf20fc6649090c3f49b6b5dd1e693a765ecb5f0856ae0da5946f5be2"
DST = Path(__file__).resolve().with_name("train.csv")


def verify(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != SHA256:
        raise ValueError(f"SHA-256 mismatch for {path}. Expected {SHA256}, got {checksum.hexdigest()}.")


def main():
    if DST.exists():
        verify(DST)
        print(f"Verified existing {DST}")
        return
    print(f"Downloading {URL}")

    def progress(blocks, block_size, total):
        if total > 0 and sys.stdout.isatty():
            sys.stdout.write(f"\r{min(100, 100 * blocks * block_size / total):5.1f}%")
            sys.stdout.flush()

    with tempfile.NamedTemporaryFile(dir=DST.parent, suffix=".part", delete=False) as out:
        partial = Path(out.name)
    try:
        urllib.request.urlretrieve(URL, partial, progress)
        verify(partial)
        os.replace(partial, DST)
    finally:
        partial.unlink(missing_ok=True)
    print(f"\nVerified and saved {DST}")


if __name__ == "__main__":
    main()
