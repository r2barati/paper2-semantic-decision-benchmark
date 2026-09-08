"""Generate the frozen sentence embeddings used by the dense retriever.

The published retrieval numbers must reproduce offline, so the embeddings are
checked into the repository and the encoder is only needed to regenerate them.
Run this after changing the corpus, the query, or the model:

    python3 -m pip install -r requirements-retrieval.txt
    python3 -m tools.build_retrieval_embeddings
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import Regime
from src.retrieval import DENSE_MODEL_NAME, FrozenEmbeddings
from src.warning_corpus import OPERATIONAL_QUERY, build_episode_corpus


def collect_texts(seeds, regimes) -> list:
    """Every text the retrieval experiment can present to the encoder."""
    texts = {OPERATIONAL_QUERY}
    for seed in seeds:
        for regime in regimes:
            for doc in build_episode_corpus(seed=seed, true_regime=regime).documents:
                texts.add(doc.text)
    return sorted(texts)


def main() -> int:
    from src.experiment_retrieval import RETRIEVAL_SEEDS, RETRIEVAL_REGIMES

    texts = collect_texts(RETRIEVAL_SEEDS, RETRIEVAL_REGIMES)
    print(f"encoding {len(texts)} unique texts with {DENSE_MODEL_NAME}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed; see requirements-retrieval.txt",
              file=sys.stderr)
        return 1

    model = SentenceTransformer(DENSE_MODEL_NAME)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=False)

    store = FrozenEmbeddings()
    store.model_name = DENSE_MODEL_NAME
    store.add(texts, vectors)
    store.save()
    print(f"wrote {store.path} ({len(texts)} vectors, dim {vectors.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
