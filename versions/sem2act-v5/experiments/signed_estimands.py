"""Signed persistence estimands for Sem2Act v5.

The subtraction signs in this module are part of the frozen protocol.
"""

from __future__ import annotations

from collections.abc import Mapping


def delta_j(j_rerank: float, j_noinfo: float) -> float:
    """Signed Rerank-versus-NoInfo realized-return effect."""
    return float(j_rerank) - float(j_noinfo)


def persistence_interaction(
    persistent: Mapping[int, float],
    reset: Mapping[int, float],
) -> float:
    """I_40,1 = [DeltaJ40 - DeltaJ1]_persistent - [DeltaJ40 - DeltaJ1]_reset."""
    required = {1, 40}
    if set(persistent) != required or set(reset) != required:
        raise ValueError("persistent and reset must contain exactly T=1 and T=40")
    return (float(persistent[40]) - float(persistent[1])) - (
        float(reset[40]) - float(reset[1])
    )


def primary_null_description() -> str:
    return (
        "H0: I_40_1 = 0. Positive I_40_1 means persistence amplifies the "
        "Rerank-versus-NoInfo effect relative to matched reset; the sign of "
        "Delta J determines whether that amplification is benefit or harm."
    )
