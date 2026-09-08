"""Resolution of frozen offline artifacts (semantic caches, fitted models).

Two rules are enforced here, both of which the September 2026 audit found
violated in the shipped release:

1. **The frozen export is addressed from the repository root**, never relative
   to the runtime cache directory.  Previously the frozen path was derived as
   ``runtime_cache.parent.parent / "results" / "frozen_llm_outputs"``, so
   redirecting ``PAPER2_LLM_CACHE_DIR`` for a clean test run also redirected
   the supposedly immutable frozen inputs -- and they silently vanished.

2. **A cache lookup never depends on credentials, and a miss never degrades
   silently.**  The Phase-9 interpreters checked ``LLM_API_KEY`` *before*
   consulting the cache and returned a 50/50 prior when it was absent.  An
   offline run therefore produced episodes labelled ``gpt-4o`` that contained
   no model output at all.  Cache is now consulted first, and an unresolvable
   lookup raises :class:`MissingFrozenArtifact` with a repair instruction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


class MissingFrozenArtifact(RuntimeError):
    """A required offline artifact is absent and cannot be reconstructed."""


def repo_root() -> Path:
    return ROOT


def runtime_cache_dir() -> Path:
    """Writable cache location; redirectable for tests via the env var."""
    return Path(os.environ.get("PAPER2_LLM_CACHE_DIR", str(ROOT / ".llm_cache")))


def frozen_cache_dir() -> Path:
    """Immutable frozen semantic export. Always relative to the repo root."""
    return ROOT / "results" / "frozen_llm_outputs"


def find_cached_response(filename: str) -> Optional[Path]:
    """Return the first existing cache file, runtime before frozen."""
    for directory in (runtime_cache_dir(), frozen_cache_dir()):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def load_cached_response(filename: str) -> Optional[dict]:
    path = find_cached_response(filename)
    if path is None:
        return None
    return json.loads(path.read_text())


def require_cached_response(filename: str, *, model: str, text: str) -> dict:
    """Load a frozen response or fail loudly.

    Never returns a placeholder belief: a silent uninformative prior would be
    recorded under the model's own name and reported as that model's result.
    """
    payload = load_cached_response(filename)
    if payload is not None:
        return payload
    raise MissingFrozenArtifact(
        f"No frozen semantic output for model={model!r} (cache file {filename}).\n"
        f"Searched: {runtime_cache_dir()} and {frozen_cache_dir()}.\n"
        "Rebuild the offline inputs with:\n"
        "    python3 -m tools.rebuild_offline_artifacts\n"
        "or set LLM_API_KEY to regenerate this entry from the provider.\n"
        f"Text begins: {text[:80]!r}"
    )
