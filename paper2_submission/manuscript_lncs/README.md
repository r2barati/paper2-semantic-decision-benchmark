# LNCS submission (current)

The ECIR 2027 full-paper submission. Supersedes `../manuscript/` (TMLR).

## Build

```bash
cd paper2_submission/manuscript_lncs
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`llncs.cls` and `splncs04.bst` are the official Springer LNCS files, vendored so
the build does not depend on a particular TeX distribution. `lmodern` is loaded
because a minimal TeX Live install ships only bitmap Computer Modern, which
makes `microtype`'s font expansion fail.

## Venue constraints and how they are checked

| Requirement | Check |
|---|---|
| Springer LNCS format | `\documentclass[runningheads]{llncs}` |
| <= 12 content pages **including appendices** | the `endofcontent` label immediately before `\bibliographystyle`; read its page from `main.aux`. Counting pages shipped before LaTeX reads `main.bbl` is wrong, because pages are shipped lazily |
| Appendices **before** references | `\appendix` precedes `\bibliography` in `main.tex` |
| Unlimited reference pages | references currently occupy pages 13-14 |
| Double-blind | anonymous author block; verify the artifact with `python3 -m tools.build_anonymous_supplement` |

Current status: **12 content pages**, 14 pages total, no overfull boxes, no
undefined references or citations.

```bash
python3 -c "import re;print(re.search(r'newlabel\{endofcontent\}\{\{[^}]*\}\{(\d+)\}',open('main.aux').read()).group(1))"
```

## Tables

Every table in `tables/` is **generated**, never hand-edited:

```bash
python3 -m results.publication.generate_lncs_tables
```

It reads the frozen episode files and recomputes each value with the declared
balanced weighted estimand (`src.metrics.weighted_benchmark_return`) and the
crossed bootstrap with shared seed draws (`src.metrics.crossed_bootstrap_ci`).
Editing a `.tex` file under `tables/` by hand will be silently overwritten, and
would break the guarantee that the paper and the data agree.

Every number quoted in the prose of `main.tex` also appears in one of those
generated tables. If you add a number to the prose, add it to a table too.
