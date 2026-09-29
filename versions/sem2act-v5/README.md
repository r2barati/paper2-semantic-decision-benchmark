# Sem2Act v5

This is an isolated confirmation overlay derived from sem2act-v4. The v4
working version is not modified by v5 execution.

The frozen package has four evidence blocks:

1. a 240-query untouched lockbox with 20 paired seeds;
2. official Llama 3.1 and Mistral 7B v0.3 consumer families on the headline subset;
3. CausalOptimizer and BeliefBaseStock controller replay plus the explicitly
   evidence-insensitive WorstCaseBaseStock safety control;
4. entropy-selective consumption and a persistence/reset horizon diagnostic.

The primary persistence arm is Qwen3-8B, Rerank versus NoInfo, C1,
CausalOptimizer, with evidence released at t=12. Its signed estimands are
defined in protocol/confirmation.yaml and tested by the local unit tests.

Execution is fail-closed. A provenance validation failure invalidates the
lockbox. A model preflight failure excludes that model without substitution.
After protocol_hash is frozen, null, mixed, contradictory, and failed results
are evidence and do not trigger redesign.

The local execution chain is:

1. fetch and verify the private qrel-free reranker output;
2. assemble and stage the three qrel-free model bundles;
3. run the Qwen primary job and the Llama/Mistral cross-family jobs only after
   their private dataset versions and model revisions are recorded;
4. install only verifier-accepted outputs;
5. assemble beliefs, run the deterministic CPU replay, and execute the
   one-pass analysis;
6. freeze manifests before editing the isolated v5 paper snapshot.

## Current pilot routing (2026-09-29)

The historic Kaggle v2 and Lightning attempts remain excluded infrastructure
history. The active Kaggle V5 binding is the user-approved v3 amendment and
runtime lock, owned by `rezabarati2`; the old `siavashsimin` binding remains
only in superseded records. The Kaggle lock installs Torch 2.7.1+cu126 before
model loading instead of inheriting the provider image. Colab/Muse has a
separate 2.11.0+cu128 lock. Moon/OpenCode has a separate CPU reranker-only
16-thread policy and requires a fresh Moon host preflight. See the project
`AGENTS.md`, `STATUS.md`, `COMPUTE.md`, and `RUNBOOK.md` for the handoff and
exact pilot gates.

The first authorized result-bearing scope is one backend-specific 20-query
operations pilot (20 fixed queries × 50 candidates). Pilot outputs are
excluded from analysis. Stop and report after each pilot; Codex review is
required before another backend or a full 240-query run. The fixed
non-lockbox fixture is `fixtures/runtime_smoke_v1.json`.

`CPU_BACKEND_V1` and `cpu_runtime_lock.json` retain the earlier CPU consumer
and artifact-conversion route. They do not define the Moon reranker route and
do not authorize Moon consumer inference. Qwen AWQ-to-GGUF conversion and
unresolved Llama source access remain separate CPU consumer blockers. The
frozen GPU Qwen AWQ/vLLM path is unchanged.

The Kaggle CPU preflight requires a user secret named 'HF_TOKEN'; its value is
never stored in the repository or canary package. The canary package embeds
only the hash-checked qrel-free config and smoke fixture because Kaggle script
runtimes may omit auxiliary package files. A missing secret, model conversion
failure, schema failure, or nondeterministic repeat is recorded as an excluded
preflight failure and blocks the CPU runtime freeze.
