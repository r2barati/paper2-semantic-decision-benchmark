# Sem2Act provenance index

Keep run-specific manifests, execution attempts, source receipts, and result
hashes beside their canonical V5 or versioned run records, primarily under
`versions/sem2act-v5/manifests/`. Do not copy or rewrite frozen records here.
Use the shared run-manifest template at
`${RESEARCH_OS_HOME:-$HOME/Research/research-os}/templates/provenance/run_manifest.json`
for new execution envelopes when the project runbook calls for one.

`kaggle_gpu_smoke_operator_report_20260928.md` records the operator-reported
GPU observation and its scope. It is an attributed operational observation,
not a V5 owner/runtime amendment or a substitute for fresh preflight.
