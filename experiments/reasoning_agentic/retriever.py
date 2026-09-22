"""Live frozen BM25 retriever over data/v3/corpus.jsonl (text-only).

Preregistered for R0 AND A3 (Correction 2). Never switch post-result.
Hard seal: aborts if qrels / evidence_labels / queries.jsonl metadata are reachable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import BM25  # frozen code: k1=1.5, b=0.75, RSJ idf floored at 0

K1 = 1.5
B = 0.75
TOP_K = 3

FORBIDDEN_NAMES = {"test.tsv", "dev.tsv", "evidence_labels.parquet", "queries.jsonl"}


class SealedCorpus:
    """Loads corpus text only; refuses sealed-label files."""

    def __init__(self, corpus_path: Path, seal_dir: Path | None = None):
        # Presence scan applies ONLY to sealed staging dirs (e.g. /kaggle/input),
        # which must never contain label files. The local repo legitimately stores
        # sealed files elsewhere; the local guarantee is that we never open them
        # (this loader reads corpus.jsonl text only — see below).
        if seal_dir is not None:
            present = {p.name for p in seal_dir.rglob("*") if p.is_file()}
            leaked = sorted(FORBIDDEN_NAMES & present)
            if leaked:
                raise RuntimeError(f"leakage seal violated, found in inputs: {leaked}")
        self.doc_ids: list[str] = []
        texts: list[str] = []
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # TEXT ONLY — metadata.kind is deliberately never read.
                self.doc_ids.append(d["_id"])
                texts.append(d["text"])
        self._bm25 = BM25(texts, k1=K1, b=B)
        self._texts = texts

    def __len__(self) -> int:
        return len(self.doc_ids)

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        import numpy as np

        scores = self._bm25.scores(query)
        idx = np.argsort(-scores, kind="stable")[:k]
        return [
            {"doc_id": self.doc_ids[i], "text": self._texts[i], "score": float(scores[i])}
            for i in idx
        ]
