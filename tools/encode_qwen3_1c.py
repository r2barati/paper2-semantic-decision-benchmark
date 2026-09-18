"""Phase 1C: encode corpus (bare docs) + instruction-prefixed queries with real Qwen3-Embedding."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer
from src.corpus_v3 import build_proto_corpus

m = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',
                        revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
instr = (ROOT / 'configs/v3/qwen_instruction.txt').read_text().strip()
b = build_proto_corpus()
docs = list(b['docs'].values())
print('encoding docs...', flush=True)
E = m.encode([d.text for d in docs], batch_size=8, show_progress_bar=False,
             normalize_embeddings=True)
print('encoding queries...', flush=True)
Q = m.encode([f'Instruct: {instr}\nQuery: {q.text}' for q in b['queries']],
             batch_size=8, show_progress_bar=False, normalize_embeddings=True)
np.savez_compressed(str(ROOT / 'runs/v3proto/qwen3emb_instruct_vectors.npz'),
                    doc_ids=np.array([d.doc_id for d in docs]),
                    qids=np.array([q.query_id for q in b['queries']]),
                    doc_emb=E.astype('float32'), q_emb=Q.astype('float32'))
print('SAVED', E.shape, Q.shape, flush=True)
