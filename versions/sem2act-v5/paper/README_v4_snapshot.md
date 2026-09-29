# Sem2Act v4 manuscript

This is the independent manuscript source for `sem2act-v4`. It starts from
the current ECIR source but is allowed to evolve only through v4 protocols and
result manifests. The current ECIR manuscript remains in
`paper2_submission/manuscript_lncs/`.

## Build

From the repository root:

```bash
make v4-paper
```

The robustness table, QREL audit table, and theory appendix are generated inputs from
`versions/sem2act-v4/results/`. Rebuild those inputs before rebuilding the PDF.

## Status

This is a working v4 draft. Provenance-only QREL validation is complete
and the calibrated external judge was excluded by its frozen gate. The
Phi-4 cross-family wave is terminally blocked after four b3 strict rejections;
the fourth retry forced actual cache misses but reproduced the malformed C1
response. No Phi-4 claims or result-dependent analysis were added. The
manuscript is at the 12-page content limit for the current evidence set, but
the model-family wave remains an explicit release limitation.

The page-budget check reads the `endofcontent` label from `main.aux`:

```bash
python3 -c "import re;print(re.search(r'newlabel\\{endofcontent\\}\\{\\{[^}]*\\}\\{(\\d+)\\}',open('main.aux').read()).group(1))"
```

Generated tables are evidence artifacts. Edit the source protocol or generator
under `versions/sem2act-v4/`, never the rendered table by hand.
