# Sem2Act-v5 CPU backend

`CPU_BACKEND_V1` is a prospective execution amendment. It preserves the
scientific protocol hash and changes only inference representation and
hardware allocation after the GPU backends were excluded.

The safe lifecycle is:

1. Record a host profile with `scripts/cpu_host_preflight.py`.
2. Build official Qwen, Llama, and Mistral GGUF files from the pinned HF
   revisions using the pinned llama.cpp commit and `Q4_K_M`.
3. Record every binary, conversion-script, GGUF, and source hash.
4. Run the fixed non-lockbox four-model canary and require exact repeat
   equality and valid C1/C3 schemas.
5. Freeze `cpu_runtime_freeze.json`.
6. Plan and run deterministic query-range shards with one consumer model per
   host.
7. Verify exact disjoint/exhaustive coverage before assembly.

`cpu_runtime_freeze.json` is intentionally absent until steps 1–4 pass. The
consumer and reranker result runners refuse to start without it. GitHub Actions
is limited to contract checks, manifests, simulation, analysis, paper builds,
and release audits; it does not run 7B–8B inference on hosted runners.
