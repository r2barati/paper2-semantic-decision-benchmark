# Red-team package: Paper 2 @ ECIR 2027 (frozen V3 manuscript)

Instance of `docs/REDTEAM_PROTOCOL.md`, round (b) — frozen manuscript.
Round (a) (inception) predates the protocol; its recoverable outputs are
noted. Files authored here contain only what does not already exist in
audited form; everything else is mapped by path (no copies, no drift).

| Package file | Source |
|---|---|
| AUDIT_SNAPSHOT.md | `results/v3main/FINAL_FREEZE.md` + manuscript `main.pdf` (12 pp, built this round) + `git rev-parse HEAD` at submission |
| CONTRIBUTION.md | authored here |
| PRIOR_ART_MATRIX.md | novelty audit §1 (14-row matrix, in review working papers) + manuscript Related work |
| SELF_OVERLAP.md | authored here |
| CLAIM_LEDGER.csv | authored here (supersedes scattered matrices; abstract-ceiling enforced) |
| DATA_AUDIT.md | `configs/v3/FINAL_FREEZE.md`, seal-guard kernels, `results/v3main/NOTES.md` chronology |
| REPRO_AUDIT.md | authored this round (fresh-env rebuild, §10) |
| STATISTICAL_AUDIT.md | independent stats audit (round b) + headline-Holm + cluster tables |
| ROBUSTNESS.md | dual priors + 5 weightings + cluster bootstrap + sensitivity bounds (`delivery/`, `NOTES.md`) |
| CONTROLS.md | random/shuffled-history/oracle/abstention/constant/tuned arms (`SIM_REPORT.md`, LEDGER §8) |
| BOUNDARIES.md | `reports/v3main_threats_to_validity.md` + manuscript §Limitations |
| BASELINE_MATRIX.md | NoInfo / tuned no-text / oracle ladder (`SIM_REPORT.md`, LEDGER §8) +
| | second controller class (`results/v3main/sim_controllerB/`, LEDGER §9) |
| ALTERNATIVES.md | E-falsification battery (pre-rewrite review §2E + §10) |
| RESULT_PROVENANCE.md | `tools/gen_*` generators + drift rerun 57/57 + mtime proofs (`NOTES.md`) |
| PRESENTATION_QC.md | authored here (citation-supports-sentence audit) |
| VENUE_FIT.md | authored here |
| INDEPENDENT_REVIEWS/ | pre-rewrite panel (6 reviews) + second-pass panel (6 reviews) + B/C re-checks (review working papers) |
| STRONGEST_REJECTION.md | authored here |
| FAIRNESS_CHECK.md | invalidation audit (pre-rewrite review §21-equivalent reasoning) |
| ACCEPTANCE_CASE.md | authored here |
| FINAL_ASSESSMENT.md | `results/v3main/delivery/FINAL_ASSESSMENT.md` + second-pass verdict |
| gates.md | authored here (human-owned steps) |

Round (a) note: no formal inception gate was run for Paper 2 (platform
predates protocol). Recoverable inception outputs: benchmark chronology
(`benchmark_chronology.md`, rule changes with validity reasons), tuning
scope pre-declaration (`configs/v3/tuning_scope.yaml`), frozen metric/model
pins. The missing-at-inception item (formal prior-art gate) was executed
late (14-row matrix) — recorded as process debt, not evidence debt.
