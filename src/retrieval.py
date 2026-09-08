"""Retrieval and evidence-selection systems over the operational report feed.

Four ranking systems plus two references, all consuming the same query and
candidate pool and returning a ranked list under a matched evidence budget:

``bm25``      Okapi BM25 (Robertson/Sparck-Jones), implemented in-repo so the
              offline release needs no extra dependency and the scoring is
              auditable.
``tfidf``     Cosine similarity over TF-IDF vectors fitted on the pool.
``dense``     Cosine similarity over sentence-transformer embeddings.
              Embeddings are FROZEN in the repository, so reproducing the
              published numbers requires no model download; regenerating them
              needs the optional extras in requirements-retrieval.txt.
``random``    A seeded random permutation: the floor.
``oracle``    Selects the documents that are actually relevant. Not a
              retrieval system -- the ceiling for evidence selection.
``none``      Retrieves nothing. The no-retrieval control.

All systems are deterministic given the corpus and seed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_STORE = ROOT / "results" / "retrieval" / "frozen_embeddings.json"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_BM25 = "bm25"
SYSTEM_TFIDF = "tfidf"
SYSTEM_DENSE = "dense"
SYSTEM_RANDOM = "random"
SYSTEM_ORACLE = "oracle"
SYSTEM_NONE = "none"

RETRIEVAL_SYSTEMS = (
    SYSTEM_NONE, SYSTEM_RANDOM, SYSTEM_BM25, SYSTEM_TFIDF, SYSTEM_DENSE, SYSTEM_ORACLE,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 with the standard k1/b parameterisation.

    Implemented here rather than pulled from a package so that the exact
    scoring function used for the published numbers is in the release and can
    be checked against the formula in the paper.
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at zero
        # so that a term appearing in more than half the collection cannot
        # contribute a negative score.
        self.idf = {
            term: max(0.0, math.log(1.0 + (self.n - count + 0.5) / (count + 0.5)))
            for term, count in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        out = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            tf, dl = self.tf[i], self.doc_len[i]
            total = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out[i] = total
        return out


# ---------------------------------------------------------------------------
# Dense embeddings (frozen)
# ---------------------------------------------------------------------------

class FrozenEmbeddings:
    """Text -> vector lookup backed by a checked-in JSON store."""

    def __init__(self, path: Path = EMBEDDING_STORE):
        self.path = path
        self._store: dict = {}
        if path.exists():
            payload = json.loads(path.read_text())
            self.model_name = payload.get("model", DENSE_MODEL_NAME)
            self._store = {k: np.asarray(v, dtype=float) for k, v in payload["vectors"].items()}
        else:
            self.model_name = DENSE_MODEL_NAME

    @staticmethod
    def key(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    def __contains__(self, text: str) -> bool:
        return self.key(text) in self._store

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._store.get(self.key(text))

    def require(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self]
        if missing:
            raise RuntimeError(
                f"{len(missing)} text(s) have no frozen embedding. Regenerate with\n"
                "    python3 -m tools.build_retrieval_embeddings\n"
                "(needs the optional extras in requirements-retrieval.txt)."
            )
        return np.vstack([self.get(t) for t in texts])

    def add(self, texts: Sequence[str], vectors: np.ndarray) -> None:
        for text, vec in zip(texts, vectors):
            self._store[self.key(text)] = np.asarray(vec, dtype=float)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One line per vector rather than one line per float: the store is a
        # tracked artifact, and `indent=0` produced a 74k-line diff for a file
        # nobody reads by hand.
        lines = ['{']
        lines.append(f'  "model": {json.dumps(self.model_name)},')
        dim = int(next(iter(self._store.values())).shape[0]) if self._store else 0
        lines.append(f'  "dim": {dim},')
        lines.append(f'  "count": {len(self._store)},')
        lines.append('  "note": "Frozen sentence embeddings. Keys are sha256(text)[:24]. '
                     'Shipped so the retrieval experiment reproduces offline without '
                     'downloading model weights.",')
        lines.append('  "vectors": {')
        items = sorted(self._store.items())
        for i, (key, vec) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            payload = ",".join(f"{float(x):.7g}" for x in vec)
            lines.append(f'    "{key}": [{payload}]{comma}')
        lines.append("  }")
        lines.append("}")
        self.path.write_text("\n".join(lines) + "\n")


def _cosine_scores(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    dn = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)
    return dn @ qn


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank(
    system: str,
    corpus,
    seed: int = 0,
    embeddings: Optional[FrozenEmbeddings] = None,
) -> list:
    """Return document ids ordered best-first for the given system."""
    texts = [d.text for d in corpus.documents]
    ids = [d.doc_id for d in corpus.documents]

    if system == SYSTEM_NONE:
        return []

    if system == SYSTEM_RANDOM:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        return [ids[int(i)] for i in order]

    if system == SYSTEM_ORACLE:
        # Relevant documents first; ties and non-relevant order are made
        # deterministic by document id so the ceiling is reproducible.
        return sorted(ids, key=lambda d: (-corpus.relevance.get(d, 0), d))

    if system == SYSTEM_BM25:
        scores = BM25(texts).scores(corpus.query)
    elif system == SYSTEM_TFIDF:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode")
        matrix = vec.fit_transform(texts + [corpus.query])
        doc_m = matrix[:-1].toarray()
        q = matrix[-1].toarray()[0]
        scores = _cosine_scores(q, doc_m)
    elif system == SYSTEM_DENSE:
        emb = embeddings or FrozenEmbeddings()
        doc_m = emb.require(texts)
        q = emb.require([corpus.query])[0]
        scores = _cosine_scores(q, doc_m)
    else:
        raise ValueError(f"unknown retrieval system {system!r}")

    # Deterministic tie-breaking by document id.
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [ids[i] for i in order]


# ---------------------------------------------------------------------------
# Retrieval-quality metrics
# ---------------------------------------------------------------------------

def precision_at_k(ranked: Sequence[str], relevance: dict, k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return sum(1 for d in top if relevance.get(d, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], relevance: dict, k: int) -> float:
    total = sum(1 for v in relevance.values() if v > 0)
    if total == 0:
        return 0.0
    top = ranked[:k]
    return sum(1 for d in top if relevance.get(d, 0) > 0) / total


def reciprocal_rank(ranked: Sequence[str], relevance: dict) -> float:
    for i, doc in enumerate(ranked, start=1):
        if relevance.get(doc, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevance: dict, k: int) -> float:
    def dcg(docs):
        return sum(
            (2 ** relevance.get(d, 0) - 1) / math.log2(i + 2)
            for i, d in enumerate(docs)
        )
    ideal = sorted(relevance, key=lambda d: -relevance.get(d, 0))[:k]
    denom = dcg(ideal)
    return (dcg(ranked[:k]) / denom) if denom > 0 else 0.0


def retrieval_metrics(ranked: Sequence[str], relevance: dict, k: int) -> dict:
    return {
        f"precision_at_{k}": precision_at_k(ranked, relevance, k),
        f"recall_at_{k}": recall_at_k(ranked, relevance, k),
        f"ndcg_at_{k}": ndcg_at_k(ranked, relevance, k),
        "mrr": reciprocal_rank(ranked, relevance),
    }
