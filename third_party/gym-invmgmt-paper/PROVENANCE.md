# Paper-1 Gym dependency provenance

This package directory is the importable `gym_invmgmt` package copied from the
Paper-1 repository at commit `a745fd5`.  It is included so the benchmark can
load the exact environment without an absolute path on the original author's
machine.  The upstream project and license are recorded in `LICENSE` and
`pyproject.toml`.

The benchmark adapters also accept `GYM_INVMGMT_PATH` when an external checkout
is preferred.
