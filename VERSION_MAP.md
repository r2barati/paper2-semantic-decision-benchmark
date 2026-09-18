# Version map — where v1, v2, and legacy live

Root working tree = **v2 (ECIR 2027, live)**. Frozen labeled snapshots sit
alongside it. Nothing was deleted; untracked precursors were moved, tracked
history was snapshotted.

| Path | Contents | Status |
|---|---|---|
| `v1/` | Full runnable snapshot of `main`: TMLR paper (*When Does Semantic Information Improve Sequential Decisions?*), `paper_sections/`, `src/` without retrieval, `results/phase5*`–`phase9*`, no `tools/`. See `v1/VERSION.md`. | Frozen. Do not edit; do not mix numbers with v2. |
| `v2/` | Full runnable snapshot of `ecir2027-repairs` HEAD: LNCS paper (*When Relevance Does Not Transfer*), Experiment R (`src/experiment_retrieval.py`, `src/retrieval.py`, `src/warning_corpus.py`, `src/notext_controls.py`), `results/retrieval/`, `results/correction_audit/`, `tools/verify_release.py`. See `v2/VERSION.md`. | Frozen snapshot of the live tree. Develop at root; re-snapshot before submission. |
| `legacy/` | `paper2/` (heterogeneous-orchestration pilot), `paper2-2/` (H/F/P toy router), `paper2_semantic_decision_mvp/` (Phase 3.5 precursor), `paper2_ecir2027_audit/` (7-Sept-2026 NOT READY audit). See `legacy/README.md`. | Irrelevant to both papers. Nothing in `v1/`/`v2/` imports it. |
| Root `src/`, `results/`, `benchmark/`, `tools/`, `tests/`, `docs/`, `paper2_submission/manuscript_lncs/` | Live v2 workspace (what CI, `tools/verify_release --all`, and the LNCS PDF build). | Editable. |
| Root `paper2_submission/manuscript/` + `paper_sections/` | Superseded TMLR staging. | Retained for history; canonical frozen view is `v1/`. |
| Root `third_party/gym-invmgmt-paper/` | Vendored simulator fork. | Legacy in function; v2 runs pinned `gym-invmgmt==0.2.1`. |
| Root `results/pre_correction_archive_2026/` | v1.0 outputs. | v1 evidence; do not compare directly with corrected v2 tables. |

## Quick discrimination
- Title *Semantic Information Improve* + no `experiment_retrieval.py` → **v1**.
- Title *Relevance Does Not Transfer* + `results/retrieval/` + `test_repairs_2026.py` → **v2**.
- `run_pilot.py` / `router.py` / `invrouter/` / Phase 3.5 11-template hypothesis → **legacy**.

Snapshots were made with `git archive main | tar -x -C v1` and
`git archive HEAD | tar -x -C v2` (tracked files only), plus the three
`VERSION.md`/`README.md` label files above. Re-run the two archive commands
to refresh a snapshot; they never touch the live tree.
