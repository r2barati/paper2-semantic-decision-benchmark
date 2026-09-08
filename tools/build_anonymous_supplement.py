"""Build and inspect the anonymous reviewer artifact.

Double-blind review requires the supplement not to identify the authors. It
does NOT permit stripping third-party copyright and license notices, and this
tool never does: `third_party/` licenses are copied verbatim.

What is removed or rewritten:

* absolute author-machine paths in tracked manifests and reports, rewritten to
  repository-relative form;
* author name, institutional email and personal repository URLs in the vendored
  dependency's *packaging metadata* (the LICENSE text is kept intact, and the
  NOTICE records that a copyright holder exists and is named in it);
* git history, credentials, editor state and private planning documents;
* the superseded TMLR staging package.

What is checked afterwards: the built tree is re-scanned for the identifying
strings, and the build fails if any survive.

    python3 -m tools.build_anonymous_supplement --out /tmp/supplement
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Directories and files that never reach a reviewer.
EXCLUDE_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".venv", "venv",
    ".llm_cache", ".idea", ".vscode", "htmlcov", "dist", "build",
    "paper2_submission/manuscript",              # superseded TMLR staging
    "results/pre_correction_archive_2026",       # internal audit history
    "results/correction_audit/pre_correction_snapshot",
}
EXCLUDE_NAMES = {".env", ".env.local", ".DS_Store", ".coverage"}
EXCLUDE_SUFFIXES = {".pyc", ".aux", ".blg", ".out", ".synctex.gz"}

# Private planning material: internal decision records, not reviewer artifacts.
EXCLUDE_GLOBS = [
    "paper2_submission/KDD_WAIT_VS_TMLR_NOW.md",
    "paper2_submission/TMLR_*.md",
    "paper2_submission/VENUE_AUDIT.md",
    "paper2_submission/TITLE_TOURNAMENT.md",
    "paper2_submission/SECRET_IDENTITY_AUDIT.md",
    "paper2_submission/AI_USE_AUTHOR_CHECKLIST.md",
    # This tool necessarily contains the identity strings it removes, so it is
    # not part of the reviewer artifact. Shipping it also shipped them: the
    # word-boundary regexes below do not match their own source text, because
    # the character before "Reza" in the literal r"\bReza Barati\b" is the
    # word character "b" from the escape, so there is no word boundary there.
    "tools/build_anonymous_supplement.py",
]

# Identifying strings, and what replaces them.
REDACTIONS = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+"), "<repo>"),
    (re.compile(r"\br2barati@torontomu\.ca\b"), "anonymous@example.invalid"),
    (re.compile(r"\bReza Barati\b"), "Anonymous Author"),
    (re.compile(r"https?://github\.com/r2barati/[A-Za-z0-9._-]+"),
     "https://anonymous.example.invalid/anonymised-repository"),
    (re.compile(r"\br2barati\b"), "anonymous"),
    (re.compile(r"\btorontomu\.ca\b"), "example.invalid"),
]

# Files whose bytes are copied verbatim, redaction included: removing a
# copyright notice to achieve anonymity is not permissible.
VERBATIM = {"LICENSE", "LICENCE", "COPYING", "NOTICE"}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".tex", ".bib", ".cfg", ".toml", ".yml", ".yaml",
    ".json", ".jsonl", ".csv", ".cls", ".bst", ".sty", ".lock", ".in", ".gitignore",
}


def excluded(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    posix = rel.as_posix()
    if any(posix.startswith(d) for d in EXCLUDE_DIRS):
        return True
    if rel.name in EXCLUDE_NAMES or rel.suffix in EXCLUDE_SUFFIXES:
        return True
    for pattern in EXCLUDE_GLOBS:
        if rel.match(pattern) or posix == pattern:
            return True
    return False


def redact(text: str) -> tuple:
    hits = 0
    for pattern, replacement in REDACTIONS:
        text, n = pattern.subn(replacement, text)
        hits += n
    return text, hits


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copied = redacted = skipped = 0
    verbatim_kept = []
    for src in sorted(ROOT.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if excluded(rel):
            skipped += 1
            continue

        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if rel.name in VERBATIM:
            shutil.copy2(src, dst)
            verbatim_kept.append(rel.as_posix())
            copied += 1
            continue

        if src.suffix in TEXT_SUFFIXES:
            try:
                text = src.read_text()
            except UnicodeDecodeError:
                shutil.copy2(src, dst)
                copied += 1
                continue
            new_text, hits = redact(text)
            dst.write_text(new_text)
            copied += 1
            if hits:
                redacted += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    return {"copied": copied, "files_redacted": redacted, "skipped": skipped,
            "verbatim_license_files": verbatim_kept}


# Plain identity strings for the audit. Deliberately NOT the redaction regexes:
# a word-boundary pattern can fail to match its own escaped source text, which
# is how an earlier version of this tool reported a false pass while shipping
# the author's name and address inside its own REDACTIONS table.
IDENTITY_STRINGS = [
    "r2barati",
    "torontomu",
    "reza barati",
    "/users/",
]


def audit(out: Path) -> list:
    """Re-scan the BUILT tree. A build that still identifies the authors fails.

    Every file is scanned, not only text-suffixed ones, and matching is a
    case-insensitive substring test so that escaping, casing or an unexpected
    file type cannot hide a match.
    """
    findings = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out)
        # Preserved license text legitimately names a copyright holder.
        if rel.name in VERBATIM:
            continue
        try:
            text = path.read_text(errors="ignore").lower()
        except OSError:
            continue
        for needle in IDENTITY_STRINGS:
            idx = text.find(needle)
            if idx != -1:
                findings.append(f"{rel}: {needle!r} at offset {idx}")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/paper2-anonymous-supplement")
    args = ap.parse_args()

    out = Path(args.out)
    stats = build(out)
    print(json.dumps(stats, indent=2))

    findings = audit(out)
    if findings:
        print(f"\nANONYMITY AUDIT FAILED: {len(findings)} identifying string(s) remain")
        for f in findings[:20]:
            print(f"  {f}")
        return 1

    print("\nANONYMITY AUDIT PASSED: no identifying strings outside preserved "
          "license/notice files.")
    print("Third-party license files kept verbatim:")
    for f in stats["verbatim_license_files"]:
        print(f"  {f}")
    print(f"\nSupplement at {out}")
    print("Human checks still required: author declarations, conflicts, "
          "concurrent-submission status, and AI-use statement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
