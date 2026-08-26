# Clean offline reproduction audit

## Contract

The canonical offline verification is run from the repository root with the
documented dependency environment:

```bash
PAPER2_LLM_CACHE_DIR=/tmp/paper2-offline-cache \
PAPER2_PHASE5_RESULTS_DIR=/tmp/paper2-offline-results \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest tests/ -q
```

The exact temporary paths may be replaced with any writable empty directories.
The suite uses frozen cached outputs where required; it makes no network/API
calls. The two environment variables prevent tests from writing into the
source checkout.

## Clean-export protocol

A tracked-file archive of the committed repository is exported to an isolated
temporary directory. The pre-existing untracked Phase-9B YAML and repository
cache are not part of the export. Frozen result files remain available as
read-only test inputs; generated Phase-5 output is redirected to the temporary
output directory.

## Result

The repaired package is ready for a fresh run. The final recorded result,
including Python version, dependency source, test count, warnings, and runtime,
is appended after the clean run; no scientific experiment is run by this audit.

