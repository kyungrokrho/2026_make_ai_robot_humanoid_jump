"""Build a portable student ZIP using an explicit allowlist.

Run from any working directory: python tools/package_release.py
"""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
  "README.md",
  "pyproject.toml",
  "uv.lock",
  ".gitignore",
  "THIRD_PARTY_NOTICES.md",
)
DIRECTORIES = ("src", "docs", "tests", "tools", "policies", "licenses")


def main():
  paths = [ROOT / name for name in FILES]
  for directory in DIRECTORIES:
    paths.extend(
      path
      for path in (ROOT / directory).rglob("*")
      if path.is_file()
      and "__pycache__" not in path.parts
      and path.suffix not in {".pyc", ".pyo"}
    )
  paths = sorted(set(paths))
  for path in paths:
    if not path.is_file() or path.is_symlink():
      raise ValueError(f"Missing file or unexpected symlink: {path}")
  destination = ROOT / "dist/jump_lab.zip"
  destination.parent.mkdir(exist_ok=True)
  with zipfile.ZipFile(
    destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6
  ) as archive:
    for path in paths:
      archive.write(path, Path("jump_lab") / path.relative_to(ROOT))
  with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
  digest = hashlib.sha256(destination.read_bytes()).hexdigest()
  destination.with_suffix(".zip.sha256").write_text(f"{digest}  {destination.name}\n")
  print(
    f"{destination} ({destination.stat().st_size / 1024**2:.1f} MiB, {len(paths)} files)"
  )


if __name__ == "__main__":
  main()
