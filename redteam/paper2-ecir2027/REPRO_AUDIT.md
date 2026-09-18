# Reproducibility audit: fresh-environment rebuild (protocol phase 10)

Date: 2026-09-17. Machine: macOS, Python 3.9.19 (matches verified env).
Fresh venv: /tmp/fresh_env (outside repo, deleted after audit except this
record), built ONLY from `requirements.txt` (+ locked `requirements.lock`
chain): numpy 1.26.4, scipy 1.13.1, pandas 2.3.3, matplotlib 3.9.0rc2,
pyarrow 21.0.0, ir-measures 0.4.3, pydantic 2.13.4. No network, no API keys,
no GPU used. Repo code+data read in place (PYTHONPATH=repo); outputs
compared by sha256 against frozen files (never edited by hand).

## Result: PASS with one hygiene fix

- 54/54 regenerated analysis files (semantic + delivery JSON/parquet/MD):
  byte-identical under the fresh interpreter.
- 99/99 tests pass under fresh deps (v3full, v3proto, semantic, repairs).
- 4/4 figure PDFs: identical GIVEN identical inputs (fresh build ==
  system build byte-for-byte on current inputs).
- Sim/gym episode files were not re-executed here (10+ GPU/CPU-hours);
  their determinism was proven by the in-repo drift rerun (57/57 identical)
  and every downstream number re-derives from them via the verified drivers.

## False alarm, investigated and closed

First comparison showed fig2/fig3 PDFs differing from an older baseline
while fig1/fig4 matched. Suspects interrogated in order: fonttools
(4.60.2 vs 4.51.0) — exonerated by reinstall + rebuild (still differed);
kiwisolver (1.4.7 vs 1.4.5) — exonerated the same way. Root cause: the
baseline predated two legitimate data changes (test-only sim recompute;
four-gaps bug fix), so the "baseline" was stale, not the build. Proof:
system-python rebuild reproduced the fresh bytes exactly, and all four
figures are byte-stable across consecutive builds (epoch-pinned). Lesson
encoded: baselines must be timestamped against the inputs that produced
them; `REPRO_AUDIT` records input hashes alongside output hashes.

## Hygiene fix from this audit

`fonttools` (matplotlib transitive dep affecting PDF subsetting) pinned to
4.51.0 in `requirements.txt`. Pillow identical both sides (11.3.0); PNGs
matched throughout. Remaining unpinned transitive deps (packaging,
pyparsing, contourpy, pytz...) have no output path in these drivers;
documented here rather than pinned speculatively.

## Head-only parameters: none found

Every number in the manuscript traces to: frozen episodes/beliefs (SHA
manifest) → listed driver scripts with pinned seeds → generated tables
(`tools/gen_lncs_tables_v3.py`). No parameter lives only in shell history
or notebooks; the two historical shell-only analyses (Q4-retrieval variant,
delay-heavy check) were recomputed from parquets and their values recorded
in LEDGER/NOTES, not hand-transcribed into the manuscript (manuscript cites
rounded values verified against tables in PRESENTATION_QC round).

Environment record: commit at submission time + dataset SHAs
(`configs/v3/FINAL_FREEZE.md`) + model pins (`configs/v3/models.yaml`) +
seeds (60000–60004 confirmation; 61000–61004 tuning) + raw outputs
(`results/v3main/`, `runs/v3main*`) + generators (`tools/run_*`,
`tools/build_delivery.py`, `tools/gen_lncs_tables_v3.py`).
