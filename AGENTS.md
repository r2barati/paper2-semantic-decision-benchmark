# Sem2Act / Paper 2 agent instructions

Read `${RESEARCH_OS_HOME:-$HOME/Research/research-os}/agents/AGENT_START_HERE.md`,
the shared `${RESEARCH_OS_HOME:-$HOME/Research/research-os}/AGENTS.md`, and this
project's [`STATUS.md`](STATUS.md), [`COMPUTE.md`](COMPUTE.md), and
[`RUNBOOK.md`](RUNBOOK.md). Treat `STATUS.md` as a concise human snapshot; the
protocol, approved amendments, and hashed machine manifests control.
Preserve stricter project-specific instructions in nested directories.

## Ownership and handoff

- Codex owns source code, repository changes, tests, experiment implementation,
  manifests, dependency/configuration changes, amendments, commits, and
  reproducibility.
- OpenCode owns compute resources and execution: Moon, Kaggle, GitHub Actions,
  cloud resources, credentials/configuration required to access compute, job
  launch/monitor/fetch, and runtime reporting.
- Muse is the Colab GPU execution operator. OpenCode is the Moon and Kaggle
  operator.
- Normal handoff: **Codex commit SHA + exact run instructions → operator
  preflight/execution/results/provenance → Codex review**. Use a clean checkout
  at the named SHA.
- Execute in stages: preflight → smallest valid pilot → review → later scale.
  Stop and report after each pilot; do not proceed into a full run, a new
  backend, or a consumer stage without the next Codex handoff.
- Operators must not silently change scientific code, frozen manifests, seeds,
  model choices, dependencies, experiment parameters, or Git history. Any
  protocol or scientific deviation requires an explicit Codex amendment and
  approval. Report repository defects to Codex for repair.
- Stop for a mismatch in account, input hashes, model/runtime, host, quota,
  authorization, output coverage, or provenance. Preserve the failed evidence.

## Frozen Sem2Act constraints

- Sem2Act V5 is a confirmation overlay with protocol hash
  `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.
  Do not edit `versions/sem2act-v5/protocol/confirmation.yaml` or
  `manifests/protocol_freeze.json` under this execution handoff.
- Preserve its 240 lockbox queries, 50 candidates/query, frozen prompts,
  schemas, estimands, private qrels/provenance gate, model revisions, decoding,
  seeds, and result policy. The 20-query shard is an operations pilot only:
  exclude it from analysis and rerun all 240 queries in a clean output folder
  for any later complete stage.
- The old Kaggle backend v2 record remains historical with owner
  `siavashsimin`. The user-approved active Kaggle v3 execution binding is
  `rezabarati2`, matching the staged reranker input dataset. Do not conflate
  either with the launcher's general `OWNER` binding.
- Kaggle requests one T4 under its explicit Torch 2.7.1+cu126 lock, even though
  the operator reported entitlement to two T4s. Colab has a separate explicit
  Torch 2.11.0+cu128 lock. Never inherit a changed provider image silently.
- Moon's 16-thread authorization is only for the CPU reranker pilot. It requires
  a fresh Moon-specific host/runtime preflight and a non-lockbox canary that
  repeats exactly at 4 and 16 threads with identical fixture rankings. Keep
  caches/builds off the small-quota home NFS. Moon consumer generation is not
  authorized.
- Qwen AWQ→GGUF conversion remains a separate CPU-consumer blocker. Do not
  substitute a model, converter, quantizer, or artifact. GPU Qwen uses its
  frozen AWQ/vLLM path; this CPU blocker does not authorize or alter it.
- Qwen, Llama, and Mistral consumer bundles are not staged. Build their
  qrel-free bundles only from a complete, fetched, verified 240-query reranker
  run and under a later stage-specific Codex handoff.
- Do not expose qrels to a preflight, smoke test, or model input bundle. Do not
  commit credentials, model caches, or run outputs unless Codex has explicitly
  reviewed and designated the provenance artifact for the repository.
