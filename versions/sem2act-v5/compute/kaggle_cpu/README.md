# Kaggle CPU preflight

This package is a non-lockbox feasibility preflight only. A generated private
kernel receives one `canary_config.json` and the hashed runtime smoke fixture.
It downloads one pinned official checkpoint, using the private Kaggle
`HF_TOKEN` secret for gated checkpoints and anonymous access only for public
pins, then builds llama.cpp at the locked commit, runs the fixed C1/C3 smoke
requests, and writes `preflight_manifest.json` plus raw smoke responses.

It does not receive `evidence_inputs.jsonl`, lockbox queries, qrels, or any
result-bearing shard. A successful preflight still must be assembled into the
local artifact and runtime manifests before any CPU result runner is allowed.
