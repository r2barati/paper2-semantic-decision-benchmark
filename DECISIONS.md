# Sem2Act / Paper 2 decision log

Append new scientific or protocol decisions only. Keep this log append-only;
correct an entry with a dated follow-up instead of editing history. The
existing V2.0 freeze and V5 backend amendments remain canonical in their
protocol and manifest files and are linked below rather than copied here.

No scientific or protocol amendment was made for this operating-documentation
setup. In particular, V5 ownership, model choices, runtime locks, seeds, and
experiment parameters were not amended.

For every new entry, record:

- Date and issue
- Previous frozen state
- Proposed or approved amendment
- Rationale
- Affected files and manifests
- Whether scientific interpretation changes, and why
- Approval source and status
- Canonical record path and hash

Canonical history: `REPRODUCIBILITY.md`,
`versions/sem2act-v5/manifests/protocol_freeze.json`, and the individual files
under `versions/sem2act-v5/amendments/`.

## 2026-09-29 — Moon snapshot staging and CPU-time execution correction

- **Issue:** OpenCode reported the first Moon pilot stopped before scoring
  because generic model staging added `.gitattributes` and `README.md` to the
  exact accepted 12-file snapshot. The same report measured a 600 soft / 1200
  hard CPU-second process limit; one 1,000-pair scorer process would exceed it.
- **Previous frozen state:** The accepted 12 model files and per-file hashes
  in `cpu_reranker_canary.json` were authoritative; the Moon launcher invoked
  one process for the full 20-query shard. The 4-vs-16 parity gate and 16-thread
  float32 scoring settings were already authorized.
- **Approved amendment:** Derive the download allowlist from that frozen
  snapshot manifest without changing its hashes; run the same scorer
  sequentially in 20 subprocesses, one complete 50-candidate query per child,
  with child CPU limits of 540 soft / 600 hard seconds. Emit a blocker manifest
  for pre-scoring failures and verify complete query/pair coverage and
  deterministic concatenation.
- **Rationale:** This fixes transfer/staging shape and operating-system
  process accounting while preserving model bytes, query and candidate sets,
  order, scorer, thread count, numerical settings, parity gate, and scientific
  interpretation.
- **Affected files:** Moon runtime amendment/lock/freeze, model staging helper,
  Moon pilot launcher, shared CPU shard runner, operator runbook, compute map,
  status snapshot, and static fixture tests.
- **Scientific interpretation changes:** No. The protocol hash remains
  `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`; no
  result-bearing job was launched by Codex.
- **Approval/status:** User-authorized on 2026-09-29; implemented and locally
  fixture-tested by Codex. The previous execution SHA is superseded for Moon;
  OpenCode must use the replacement SHA supplied with this handoff.
- **Canonical record path and hash:**
  `versions/sem2act-v5/amendments/moon_reranker_backend_v1.yaml`, SHA-256
  `8ed3e398e472bcb8e46c24c1b32bdd0a27aec02879b4f45ef89f6a36933680ee`.
