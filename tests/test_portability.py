"""The repository is public: no machine paths, LAN addresses or secrets in tracked files."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".build", "bin", "models", "run", "data", ".git", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".sh", ".js", ".html", ".css", ".md", ".txt", ""}

PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
HOME_PATH = re.compile(r"(?:/home/|/Users/|/root/)[A-Za-z0-9_.-]+")
SECRET = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})\b")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def tracked_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}:
            yield path


def test_no_machine_specific_values():
    problems = []
    for path in tracked_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in (PRIVATE_IP, HOME_PATH, SECRET, EMAIL):
            for m in pattern.finditer(text):
                problems.append(f"{path.relative_to(ROOT)}: {m.group(0)}")
    assert not problems, "\n".join(problems)


def test_root_has_only_the_expected_files():
    allowed = {"setup.sh", "serve.sh", ".gitignore", "README.md", "LICENSE"}
    files = {p.name for p in ROOT.iterdir() if p.is_file()}
    assert files <= allowed, files - allowed


def test_scripts_are_executable():
    for name in ("setup.sh", "serve.sh"):
        assert (ROOT / name).stat().st_mode & 0o111, f"{name} is not executable"
