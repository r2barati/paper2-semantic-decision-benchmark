# V5 runtime data

Input bundles, model caches, job outputs, and private lockbox artifacts are not stored in this Git branch. The qrel-free Kaggle reranker input bundle is externally staged as `rezabarati2/sem2act-v5-rerank-inputs`; its remote verification and SHA-256 manifest is `../manifests/upload/v5-rerank-inputs.json`. Operators must fetch into node-local or ephemeral storage and verify every hash before use. Never copy qrels into a preflight or model bundle.
