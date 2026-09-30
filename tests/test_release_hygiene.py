"""Repository hygiene checks: nothing private, large, or cluster-specific may be tracked.

Runs on the git-tracked files (falls back to a directory walk outside a git checkout).
Run with:  pytest -q tests
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve().relative_to(ROOT).as_posix()

MAX_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".zip", ".npz")
FORBIDDEN_PATH_PART = re.compile(r"^(submitit.*|wandb|results_.*)$")
# The forbidden patterns are built from pieces so that this file does not match itself.
PRIVATE_TEXT = re.compile(
    "|".join(
        [
            "/" + r"mnt/",
            "/" + r"home/[\w.-]+",
            "/" + r"Users/[\w.-]+",
            r"[A-Za-z]:\\Users\\",
            "jkapoor" + r"\w*",
        ]
    )
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
# Absolute filesystem paths in notebook text outputs (image payloads are skipped).
ABS_PATH = re.compile(
    r"(?<![\w./:~-])/(?:tmp|home|mnt|Users|var|private|scratch|lustre|opt|usr|etc|root|srv)/[\w.\-/]*"
)


def tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode()
        rels = [p for p in out.split("\0") if p]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        skip = {".git", ".venv", "__pycache__", ".pytest_cache"}
        rels = [
            p.relative_to(ROOT).as_posix()
            for p in ROOT.rglob("*")
            if p.is_file() and not skip & set(p.relative_to(ROOT).parts)
        ]
    return [ROOT / p for p in rels if (ROOT / p).is_file()]


FILES = tracked_files()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def read_text(p: Path) -> str | None:
    try:
        return p.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_files_found():
    assert FILES, "no tracked files found"


def test_no_large_files():
    big = [f"{rel(p)} ({p.stat().st_size / 2**20:.1f} MB)" for p in FILES if p.stat().st_size > MAX_BYTES]
    assert not big, f"tracked files larger than 5 MB: {big}"


def test_no_binary_artifacts():
    bad = [rel(p) for p in FILES if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not bad, f"checkpoints / archives must not be tracked: {bad}"


def test_no_cluster_or_run_directories():
    bad = [rel(p) for p in FILES if any(FORBIDDEN_PATH_PART.match(part) for part in p.relative_to(ROOT).parts)]
    assert not bad, f"submitit / wandb / results_* paths must not be tracked: {bad}"


def test_no_private_paths_or_usernames():
    bad = []
    for p in FILES:
        if rel(p) == THIS_FILE:
            continue
        text = read_text(p)
        if text is None:
            continue
        for m in PRIVATE_TEXT.finditer(text):
            bad.append(f"{rel(p)}: {m.group(0)}")
            break
    assert not bad, f"private paths / usernames found: {bad}"


def test_no_email_addresses_outside_readme_contact():
    bad = []
    for p in FILES:
        if rel(p) == THIS_FILE:
            continue
        text = read_text(p)
        if text is None:
            continue
        if rel(p) == "README.md":
            head, sep, contact = text.partition("\n## Contact")
            text = head + "\n".join(contact.split("\n## ")[1:])  # everything but the contact section
        for m in EMAIL.finditer(text):
            bad.append(f"{rel(p)}: {m.group(0)}")
            break
    assert not bad, f"e-mail addresses outside the README contact section: {bad}"


def notebook_text_outputs(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []):
            if "text" in out:
                yield i, "".join(out["text"])
            if "traceback" in out:
                yield i, "\n".join(out["traceback"])
            for mime, val in out.get("data", {}).items():
                if not mime.startswith("image/"):
                    yield i, "".join(val) if isinstance(val, list) else str(val)


@pytest.mark.parametrize("nb_path", [p for p in FILES if p.suffix == ".ipynb"], ids=rel)
def test_notebook_outputs_have_no_absolute_paths(nb_path):
    bad = [f"cell {i}: {m.group(0)}" for i, txt in notebook_text_outputs(nb_path) for m in ABS_PATH.finditer(txt)]
    assert not bad, f"absolute paths in outputs of {rel(nb_path)}: {bad[:5]}"
