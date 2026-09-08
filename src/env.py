"""Sequential single-item inventory simulator.

Economic model (lost-sales variant):
    - Each period: observe demand, fulfill from on-hand, place an order.
    - Orders arrive after `lead_time` periods and are added to on-hand.
    - Unsatisfied demand is lost (lost-sales).
    - Holding cost on ending on-hand inventory.
    - Stockout cost per unit of unmet demand.
    - Revenue per unit sold.
    - Fixed + variable ordering cost.

Parameters are chosen so that the problem is non-trivial but small enough
for a one-day MVP.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.events import (
    SupplierDisruption, Regime, REGIME_PARAMS,
    P5_NORMAL_LEAD_TIME, P5_DEMAND_MEAN, P5_EVENT_START, P5_EVENT_DURATION,
    P4_WARNING_TIME,
)


# --- Economic parameters (documented, transparent) ---------------------------

REVENUE_PER_UNIT = 10.0          # selling price
ORDERING_FIXED_COST = 5.0        # fixed cost per non-zero order
ORDERING_VARIABLE_COST = 2.0     # per-unit ordering cost
HOLDING_COST_PER_UNIT = 1.0      # per-unit per-period ending inventory cost
STOCKOUT_COST_PER_UNIT = 8.0     # penalty per unit of unmet demand (lost sale)
INITIAL_INVENTORY = 30           # starting on-hand
HORIZON = 40                     # number of periods
DEMAND_MEAN = 8.0                # Poisson lambda (also used for base-stock calc)

# Forward-planning fallbacks for a controller acting on a warning whose text
# does not state when the event begins or how long it lasts.  Both are derived
# from the evaluation protocol (warning release time, planning horizon) and
# never from the environment's hidden disruption parameters.
DEFAULT_ASSUMED_START = P4_WARNING_TIME
DEFAULT_ASSUMED_DURATION = HORIZON

# Below this quantity a solver value is numerical residue rather than an order.
ORDER_ZERO_TOL = 1e-9


class PipelineSchedule:
    """Absolute-time order pipeline shared by the simulator and the planners.

    This is the single source of truth for arrival timing.  The declared
    semantics are:

    * An order placed in period ``t`` under lead time ``L`` arrives at the
      start of period ``t + L`` and can serve demand in that same period.
      A declared lead time of two therefore means an order placed at ``t=0``
      is on hand for period ``t=2``.
    * Orders already in transit keep the arrival period fixed at the moment
      they were placed.  A lead-time change -- disruption onset or recovery --
      applies only to orders placed from that period onward.

    The previous FIFO-list implementation padded the queue to ``L`` slots
    *before* appending, which delivered orders after ``L + 1`` periods and
    also pushed in-transit orders back whenever the lead time grew.  Both
    behaviours contradicted the declared experiment, so the queue is now keyed
    by absolute arrival period instead.
    """

    __slots__ = ("_schedule",)

    def __init__(self, schedule: Optional[dict] = None):
        self._schedule: dict[int, float] = dict(schedule) if schedule else {}

    def copy(self) -> "PipelineSchedule":
        return PipelineSchedule(self._schedule)

    def receive(self, t: int) -> float:
        """Remove and return the quantity arriving in period ``t``."""
        return self._schedule.pop(t, 0.0)

    def place(self, t: int, lead_time: float, qty: float) -> int:
        """Schedule ``qty`` placed in period ``t``; return its arrival period."""
        arrival = t + max(1, int(round(lead_time)))
        if qty:
            self._schedule[arrival] = self._schedule.get(arrival, 0.0) + float(qty)
        return arrival

    def outstanding(self) -> float:
        """Total units still in transit."""
        return float(sum(self._schedule.values()))

    def prune_before(self, t: int) -> None:
        """Drop stale entries scheduled before ``t`` (already received)."""
        for key in [k for k in self._schedule if k < t]:
            del self._schedule[key]

    def as_dict(self) -> dict:
        return dict(self._schedule)


@dataclass
class InventoryState:
    """Complete observable state of the inventory system at a point in time."""

    time: int = 0
    on_hand: float = 0.0
    pipeline: float = 0.0  # units in transit
    demand: float = 0.0
    sales: float = 0.0
    lost_sales: float = 0.0
    order_quantity: float = 0.0
    lead_time: int = 2
    revenue: float = 0.0
    ordering_cost: float = 0.0
    holding_cost: float = 0.0
    stockout_cost: float = 0.0
    period_profit: float = 0.0
    cumulative_profit: float = 0.0


class HindsightOracle:
    """Perfect-knowledge optimizer: pre-computes optimal order sequence.

    Given full knowledge of future demand AND disruption timing/regime,
    solves for the cost-minimizing order quantity at each period using
    a MILP with exact fixed ordering cost.

    Scope of the guarantee.  With continuous order quantities and binary setup
    indicators the MILP is an exact optimum *of the modelled problem*, i.e. the
    best achievable clairvoyant return under this simulator's dynamics, cost
    structure and order-quantity domain.  It is a non-causal, non-deployable
    reference, not a bound on any richer policy class, and it is only an upper
    bound once :meth:`verify_against_simulator` confirms the LP objective and
    the simulated return agree.

    Corrections applied September 2026 (previously invalidating the bound):
      * the empty initial pipeline was encoded with real decision-variable
        indices ``range(normal_lt)``, which let ``order[0..L-1]`` arrive before
        it was placed;
      * ``integrality=2`` is SciPy's *semi-continuous* marker, so the supposed
        binary setup indicators could take fractional values;
      * the first-period sales bound omitted the initial on-hand stock;
      * solver quantities were rounded to one decimal, discarding optimality;
      * the failure branch called an undefined ``_fallback_greedy``.
    """

    def __init__(
        self,
        seed: int,
        disruption: Optional[SupplierDisruption] = None,
        horizon: int = HORIZON,
        initial_inventory: int = INITIAL_INVENTORY,
        demand_mean: float = DEMAND_MEAN,
        regime: Optional[Regime] = None,
    ):
        self.seed = seed
        self.disruption = disruption
        self.horizon = horizon
        self.initial_inventory = initial_inventory
        self.demand_mean = demand_mean
        self.regime = regime

    def _get_regime_lead_time(self, t: int) -> int:
        """Get the lead time at period t under regime dynamics."""
        if self.regime is None:
            if self.disruption:
                return self.disruption.current_lead_time(t)
            return P5_NORMAL_LEAD_TIME if self.regime is not None else 2
        params = REGIME_PARAMS[self.regime]
        if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
            return P5_NORMAL_LEAD_TIME + params["lead_time_increase"]
        return P5_NORMAL_LEAD_TIME

    def _get_regime_demand_mean(self, t: int) -> float:
        """Get the demand mean at period t under regime dynamics."""
        if self.regime is None:
            return self.demand_mean
        params = REGIME_PARAMS[self.regime]
        if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
            return P5_DEMAND_MEAN * params["demand_multiplier"]
        return P5_DEMAND_MEAN

    def _get_known_demands(self) -> list[int]:
        """Get the demand sequence for this seed (deterministic from seed).

        For regime mode, demand mean varies per period.
        """
        rng = np.random.default_rng(self.seed)
        demands = []
        for t in range(self.horizon):
            if self.regime is not None:
                d_mean = self._get_regime_demand_mean(t)
            else:
                d_mean = self.demand_mean
            demands.append(int(rng.poisson(d_mean)))
        return demands

    def _compute_optimal_orders(self, demands: list[int]) -> list[float]:
        """Compute optimal order quantities using MILP with exact fixed ordering cost.

        Formulates a Mixed-Integer Linear Program that minimizes total cost
        (negative profit) subject to:
        - FIFO queue arrival matrix matching the env's exact dynamics
        - Binary variables modeling the fixed ordering cost exactly
        - Inventory balance, sales bounds, and non-negativity

        This is solved to global optimality by scipy's MILP solver.
        """
        from scipy.optimize import milp, LinearConstraint, Bounds

        H = self.horizon

        # Build the arrival matrix from the *shared* PipelineSchedule so the
        # oracle plans against exactly the dynamics InventoryEnv simulates.
        # The pipeline starts empty: no order can arrive before it is placed.
        A = np.zeros((H, H))
        normal_lt = (
            P5_NORMAL_LEAD_TIME if self.regime is not None
            else (self.disruption.normal_lead_time if self.disruption else 2)
        )
        schedule: dict[int, list[int]] = {}
        for t in range(H):
            if self.regime is not None:
                lt = self._get_regime_lead_time(t)
            elif self.disruption is not None:
                lt = self.disruption.current_lead_time(t)
            else:
                lt = normal_lt
            arrival = t + max(1, int(lt))
            schedule.setdefault(arrival, []).append(t)
        for arrival, placed in schedule.items():
            if arrival < H:
                for j in placed:
                    A[arrival, j] = 1.0

        # Variables: order[H], y[H] (binary), on_hand[H], sales[H]
        n = 4 * H

        # Objective: minimize total cost
        c = np.zeros(n)
        for t in range(H):
            c[t] = ORDERING_VARIABLE_COST            # variable ordering cost
            c[H + t] = ORDERING_FIXED_COST            # fixed ordering cost (binary)
            c[2 * H + t] = HOLDING_COST_PER_UNIT      # holding cost
            c[3 * H + t] = -(REVENUE_PER_UNIT + STOCKOUT_COST_PER_UNIT)  # revenue + stockout

        # Integrality: SciPy uses 0=continuous, 1=integer, 2=semi-continuous,
        # 3=semi-integer.  Binary is integrality 1 combined with [0, 1] bounds;
        # the previous value 2 declared the setup indicators semi-continuous
        # and therefore allowed fractional "binaries".
        integrality = np.zeros(n)
        integrality[H:2 * H] = 1

        # Bounds
        lb = np.zeros(n)
        ub = np.full(n, np.inf)
        ub[H:2 * H] = 1.0  # y in [0, 1]
        bounds = Bounds(lb, ub)

        # Equality: on_hand[t] = on_hand[t-1] + arrivals[t] - sales[t]
        A_eq = np.zeros((H, n))
        b_eq = np.zeros(H)
        for t in range(H):
            A_eq[t, 2 * H + t] = 1.0  # on_hand[t]
            if t > 0:
                A_eq[t, 2 * H + (t - 1)] = -1.0  # -on_hand[t-1]
            A_eq[t, 3 * H + t] = 1.0  # +sales[t]
            for j in range(H):
                if A[t, j] > 0.5:
                    A_eq[t, j] = -1.0  # -arrivals
            b_eq[t] = self.initial_inventory if t == 0 else 0.0

        # Inequality:
        # sales[t] <= on_hand[t-1] + arrivals[t]
        # sales[t] <= demands[t]
        # order[t] <= M * y[t].  Justified bound: no optimal solution ever
        # orders more than the total remaining demand, because unsold units
        # only accrue holding and variable ordering cost.
        M = float(sum(demands)) + 1.0
        A_ub = np.zeros((3 * H, n))
        b_ub = np.zeros(3 * H)
        for t in range(H):
            # sales[t] - on_hand[t-1] - arrivals[t] <= 0
            A_ub[t, 3 * H + t] = 1.0
            if t > 0:
                A_ub[t, 2 * H + (t - 1)] = -1.0
            for j in range(H):
                if A[t, j] > 0.5:
                    A_ub[t, j] = -1.0
            # The predecessor stock in period 0 is the (constant) initial
            # inventory, so it belongs on the right-hand side.
            b_ub[t] = float(self.initial_inventory) if t == 0 else 0.0

            # sales[t] <= demands[t]
            A_ub[H + t, 3 * H + t] = 1.0
            b_ub[H + t] = demands[t]

            # order[t] <= M * y[t]
            A_ub[2 * H + t, t] = 1.0
            A_ub[2 * H + t, H + t] = -M
            b_ub[2 * H + t] = 0.0

        constraints = [
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub),
        ]

        result = milp(
            c, constraints=constraints, bounds=bounds, integrality=integrality
        )

        if not result.success:
            raise RuntimeError(
                "HindsightOracle MILP did not solve to optimality "
                f"(status={result.status}: {result.message}). The oracle is a "
                "reference bound; silently substituting a heuristic would "
                "invalidate it."
            )
        # Do NOT round: the MILP optimum is the reference value, and rounding
        # order quantities would discard the optimality it certifies.
        #
        # The objective prices sales at (revenue + stockout) so that a lost
        # sale is charged implicitly.  Realised profit therefore equals
        # -fun - STOCKOUT_COST_PER_UNIT * total demand; the second term is a
        # constant that does not affect the argmin but must be restored before
        # the objective is compared with a simulated return.
        self.last_objective = float(result.fun)
        self.last_demand_total = float(sum(demands))
        self.last_planned_return = (
            -self.last_objective - STOCKOUT_COST_PER_UNIT * self.last_demand_total
        )

        # Clean solver residue only.  The simulator charges the fixed ordering
        # cost whenever ``order_qty > 0``, so an order of ~1e-16 would add a
        # spurious setup cost the MILP never paid for.  The setup indicator
        # y[t] is the model's own statement about whether an order is placed,
        # so a quantity is zeroed exactly when y[t] says "no order".  Genuine
        # quantities are never rounded.
        cleaned = []
        for t in range(H):
            qty = max(0.0, float(result.x[t]))
            placed = float(result.x[H + t]) >= 0.5
            cleaned.append(qty if (placed and qty > ORDER_ZERO_TOL) else 0.0)
        return cleaned

    def compute_optimal_trajectory(self) -> tuple[list[InventoryState], list[float]]:
        """Run the environment with pre-computed optimal orders.

        Returns:
            (history, orders) where history is the list of per-period states
            and orders is the list of optimal order quantities.
        """
        demands = self._get_known_demands()
        optimal_orders = self._compute_optimal_orders(demands)

        env = InventoryEnv(seed=self.seed, disruption=self.disruption, regime=self.regime)
        env.reset()
        history = []
        for t in range(self.horizon):
            state = env.step(optimal_orders[t])
            history.append(state)

        return history, optimal_orders

    def verify_against_simulator(self, tol: float = 1e-6) -> dict:
        """Check that the MILP objective equals the simulated realised return.

        The oracle may only be described as a reference upper bound when the
        plan it optimises and the trajectory the simulator actually produces
        agree.  Returns a dict with both values and the signed gap; callers
        (and :mod:`tests.test_env`) assert ``abs(gap) <= tol``.
        """
        history, orders = self.compute_optimal_trajectory()
        simulated = float(history[-1].cumulative_profit) if history else 0.0
        planned = float(getattr(self, "last_planned_return", float("nan")))
        return {
            "planned_objective": planned,
            "simulated_return": simulated,
            "gap": planned - simulated,
            "within_tolerance": abs(planned - simulated) <= tol,
            "orders": orders,
        }


class CausalOptimizer:
    """Receding-horizon optimizer — belief-aware multi-regime or single-disruption.

    Phase 4 mode (single disruption):
        Uses assumed_lt_increase/assumed_duration from semantic interpretation.

    Phase 5 mode (multi-regime):
        Accepts regime_probabilities dict and computes expected lead time
        and demand across regimes for each future period.

    At each period:
    1. Observe current state (on_hand, pipeline)
    2. Compute expected demand and lead time from regime beliefs
    3. Optimize order decisions over remaining horizon using LP
    4. Execute ONLY the first order
    5. Re-optimize next period

    CRITICAL: The optimizer uses BELIEF-BASED forward models, NOT true
    realized demand or true regime. This is how semantic interpretation
    quality propagates to operational decisions through a strong controller.
    """

    def __init__(
        self,
        seed: int,
        disruption: Optional[SupplierDisruption] = None,
        assumed_lt_increase: Optional[int] = None,
        assumed_duration: Optional[int] = None,
        assumed_start: Optional[int] = None,
        horizon: int = HORIZON,
        initial_inventory: int = INITIAL_INVENTORY,
        demand_mean: float = DEMAND_MEAN,
        regime_probabilities: Optional[dict[str, float]] = None,
        knows_true_disruption: bool = False,
    ):
        self.seed = seed
        self.disruption = disruption
        self.horizon = horizon
        self.initial_inventory = initial_inventory
        self.demand_mean = demand_mean
        self.regime_probabilities = regime_probabilities

        # Phase 5: base parameters for regime computation
        self._base_normal_lead_time = P5_NORMAL_LEAD_TIME
        self._base_demand_mean = P5_DEMAND_MEAN

        if disruption is not None:
            self.normal_lead_time = disruption.normal_lead_time
        else:
            self.normal_lead_time = (
                P5_NORMAL_LEAD_TIME if regime_probabilities else 2
            )
        self._pipeline = PipelineSchedule()

        # Build the ASSUMED disruption used for forward planning.
        #
        # INFORMATION BOUNDARY.  ``disruption`` describes the *environment's*
        # ground truth.  A controller may only plan against it when the caller
        # explicitly declares perfect information (``knows_true_disruption``),
        # which is true for the PerfectSemantic reference alone.  Any other
        # sensor plans either against its own interpreted disruption or, with
        # no usable interpretation, against the nominal no-disruption model.
        #
        # Before September 2026 every branch inherited ``disruption`` here, so
        # the NoInfo and low-confidence controllers silently planned with the
        # true lead-time trajectory.  That made NoInfo and PerfectSemantic
        # numerically identical in the historical sensor x controller matrix.
        self.knows_true_disruption = bool(knows_true_disruption)
        if assumed_lt_increase is not None and disruption is not None:
            self._assumed_disruption = SupplierDisruption(
                start_time=assumed_start if assumed_start is not None else DEFAULT_ASSUMED_START,
                duration=assumed_duration if assumed_duration is not None else DEFAULT_ASSUMED_DURATION,
                normal_lead_time=disruption.normal_lead_time,
                disrupted_lead_time=disruption.normal_lead_time + assumed_lt_increase,
            )
        elif self.knows_true_disruption:
            self._assumed_disruption = disruption
        else:
            # No information: plan under the nominal (undisrupted) lead time.
            self._assumed_disruption = None

    def set_regime_probabilities(self, regime_probabilities: dict) -> None:
        """Adopt a new regime belief without disturbing operational state.

        Used to release warning text mid-episode: the controller plans under
        the benchmark prior until the warning is published, then re-plans with
        the interpreted belief while keeping its on-hand stock and in-transit
        pipeline exactly as they are.
        """
        self.regime_probabilities = dict(regime_probabilities)

    def _get_expected_lt(self, t: int) -> float:
        """Get expected lead time at period t from regime beliefs."""
        if self.regime_probabilities is not None:
            expected_lt = 0.0
            for regime, prob in self.regime_probabilities.items():
                params = REGIME_PARAMS[regime]
                if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
                    lt = self._base_normal_lead_time + params["lead_time_increase"]
                else:
                    lt = self._base_normal_lead_time
                expected_lt += prob * lt
            return expected_lt
        elif self._assumed_disruption is not None:
            return float(self._assumed_disruption.current_lead_time(t))
        return float(self.normal_lead_time)

    def _get_expected_demand(self, t: int) -> float:
        """Get expected demand mean at period t from regime beliefs."""
        if self.regime_probabilities is not None:
            expected_demand = 0.0
            for regime, prob in self.regime_probabilities.items():
                params = REGIME_PARAMS[regime]
                if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
                    d = self._base_demand_mean * params["demand_multiplier"]
                else:
                    d = self._base_demand_mean
                expected_demand += prob * d
            return expected_demand
        return self.demand_mean

    def _get_lt(self, t: int) -> int:
        """Get the lead time at period t for forward planning (rounded for queue)."""
        if self.regime_probabilities is not None:
            return max(1, round(self._get_expected_lt(t)))
        if self._assumed_disruption is not None:
            return self._assumed_disruption.current_lead_time(t)
        return self.normal_lead_time

    def _simulate_arrivals(
        self, t_start: int, schedule: "PipelineSchedule", orders: list[float]
    ) -> list[float]:
        """Project arrivals under the shared :class:`PipelineSchedule` rules.

        This mirrors :meth:`InventoryEnv.step` period for period, so the
        planner's arrival model and the simulator cannot drift apart.
        """
        projected = schedule.copy()
        arrivals = []
        for i, order in enumerate(orders):
            t = t_start + i
            arrivals.append(projected.receive(t))
            projected.place(t, self._get_lt(t), order)
        return arrivals

    def _build_arrival_model(
        self, t_start: int, schedule: "PipelineSchedule", remaining: int
    ) -> tuple:
        """Build linear arrival model by simulation.

        Returns:
            a_fixed: arrivals from existing queue (length remaining)
            A: marginal arrival matrix (remaining x remaining)
        """
        a_fixed = np.array(
            self._simulate_arrivals(t_start, schedule, [0.0] * remaining)
        )
        A = np.zeros((remaining, remaining))
        for j in range(remaining):
            orders_j = [0.0] * remaining
            orders_j[j] = 1.0
            arr_j = np.array(
                self._simulate_arrivals(t_start, schedule, orders_j)
            )
            A[:, j] = arr_j - a_fixed
        return a_fixed, A

    def _solve_mpc(self, t: int, on_hand: float, remaining: int) -> float:
        """Solve receding-horizon LP for remaining periods. Returns first order."""
        from scipy.optimize import linprog

        R = remaining
        a_fixed, A_mat = self._build_arrival_model(t, self._pipeline, R)
        # Phase 5: use belief-aware expected demand per period
        demands = np.array([self._get_expected_demand(t + i) for i in range(R)])

        # Variables: order[0..R-1], on_hand[0..R-1], sales[0..R-1]
        n_vars = 3 * R
        c = np.zeros(n_vars)
        for i in range(R):
            c[i] = ORDERING_VARIABLE_COST
            c[R + i] = HOLDING_COST_PER_UNIT
            c[2 * R + i] = -(REVENUE_PER_UNIT + STOCKOUT_COST_PER_UNIT)

        # Equality: on_hand[i] = on_hand[i-1] + arrivals[i] - sales[i]
        A_eq = np.zeros((R, n_vars))
        b_eq = np.zeros(R)
        for i in range(R):
            A_eq[i, R + i] = 1.0  # on_hand[i]
            if i > 0:
                A_eq[i, R + (i - 1)] = -1.0  # -on_hand[i-1]
            A_eq[i, 2 * R + i] = 1.0  # +sales[i]
            for j in range(R):
                if A_mat[i, j] > 0.5:
                    A_eq[i, j] = -1.0  # -arrivals from order[j]
            b_eq[i] = a_fixed[i] + (on_hand if i == 0 else 0)

        # Inequality: sales[i] <= on_hand[i-1] + arrivals[i]
        #             sales[i] <= demands[i]
        # For i == 0 the predecessor stock is the *current* on-hand level, which
        # is a constant rather than a decision variable, so it belongs on the
        # right-hand side.  Omitting it (the pre-2026 form) made the LP believe
        # no stock was sellable in the first period.
        A_ub = np.zeros((2 * R, n_vars))
        b_ub = np.zeros(2 * R)
        for i in range(R):
            A_ub[i, 2 * R + i] = 1.0  # sales[i]
            if i > 0:
                A_ub[i, R + (i - 1)] = -1.0  # -on_hand[i-1]
            for j in range(R):
                if A_mat[i, j] > 0.5:
                    A_ub[i, j] = -1.0  # -arrivals
            b_ub[i] = a_fixed[i] + (on_hand if i == 0 else 0.0)

            A_ub[R + i, 2 * R + i] = 1.0
            b_ub[R + i] = demands[i]

        bounds = [(0, None)] * n_vars
        result = linprog(
            c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds, method="highs"
        )

        if result.success:
            return max(0.0, result.x[0])
        else:
            return 0.0

    def decide(self, state: "InventoryState") -> float:
        """Decide order quantity for current period using receding-horizon LP."""
        t = state.time
        on_hand = state.on_hand

        # Anything scheduled before the current period has already been
        # received by the simulator; drop it so projections stay aligned.
        self._pipeline.prune_before(t)

        remaining = self.horizon - t
        if remaining <= 0:
            return 0.0
        order = self._solve_mpc(t, on_hand, remaining)

        # Record the order under the same timing rule the simulator applies.
        self._pipeline.place(t, self._get_lt(t), order)

        return order


class InventoryEnv:
    """Single-item lost-sales inventory environment.

    Supports both Phase-4 (SupplierDisruption) and Phase-5 (Regime) modes.

    Phase 5: When regime is set, the environment applies regime-dependent
    lead time and demand modifications during the event window
    (P5_EVENT_START to P5_EVENT_START + P5_EVENT_DURATION).

    Usage:
        env = InventoryEnv(seed=1000)
        obs = env.reset()
        for _ in range(HORIZON):
            obs = env.step(order_qty)
    """

    def __init__(
        self,
        seed: int = 1000,
        disruption: Optional[SupplierDisruption] = None,
        horizon: int = HORIZON,
        initial_inventory: int = INITIAL_INVENTORY,
        demand_mean: float = DEMAND_MEAN,
        regime: Optional[Regime] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.disruption = disruption
        self.horizon = horizon
        self.initial_inventory = initial_inventory
        self.demand_mean = demand_mean
        self.regime = regime

        # Phase 5: base parameters for regime-dependent dynamics
        self._base_normal_lead_time = P5_NORMAL_LEAD_TIME
        self._base_demand_mean = P5_DEMAND_MEAN

        if disruption is not None:
            self.normal_lead_time = disruption.normal_lead_time
        else:
            self.normal_lead_time = (
                P5_NORMAL_LEAD_TIME if regime else 2
            )
        self._pipeline = PipelineSchedule()  # absolute-time order pipeline
        self._state: Optional[InventoryState] = None

    def _get_regime_lead_time(self, t: int) -> int:
        """Get the lead time at period t under regime dynamics."""
        if self.regime is None:
            return self.normal_lead_time
        params = REGIME_PARAMS[self.regime]
        if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
            return self._base_normal_lead_time + params["lead_time_increase"]
        return self._base_normal_lead_time

    def _get_regime_demand_mean(self, t: int) -> float:
        """Get the demand mean at period t under regime dynamics."""
        if self.regime is None:
            return self.demand_mean
        params = REGIME_PARAMS[self.regime]
        if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
            return self._base_demand_mean * params["demand_multiplier"]
        return self._base_demand_mean

    def reset(self) -> InventoryState:
        """Reset environment and return initial state."""
        self.rng = np.random.default_rng(self.seed)
        init_lt = (
            self._get_regime_lead_time(0) if self.regime
            else self.normal_lead_time
        )
        self._pipeline = PipelineSchedule()  # empty pipeline
        self._state = InventoryState(
            time=0,
            on_hand=float(self.initial_inventory),
            pipeline=0.0,
            demand=0.0,
            sales=0.0,
            lost_sales=0.0,
            order_quantity=0.0,
            lead_time=init_lt,
            revenue=0.0,
            ordering_cost=0.0,
            holding_cost=0.0,
            stockout_cost=0.0,
            period_profit=0.0,
            cumulative_profit=0.0,
        )
        return InventoryState(
            time=self._state.time,
            on_hand=self._state.on_hand,
            pipeline=self._state.pipeline,
            demand=self._state.demand,
            sales=self._state.sales,
            lost_sales=self._state.lost_sales,
            order_quantity=self._state.order_quantity,
            lead_time=self._state.lead_time,
            revenue=self._state.revenue,
            ordering_cost=self._state.ordering_cost,
            holding_cost=self._state.holding_cost,
            stockout_cost=self._state.stockout_cost,
            period_profit=self._state.period_profit,
            cumulative_profit=self._state.cumulative_profit,
        )

    def step(self, order_qty: float) -> InventoryState:
        """Execute one period: receive demand, fulfill, place order.

        Args:
            order_qty: Number of units to order this period (>= 0).

        Returns:
            Updated state after the period completes.
        """
        s = self._state
        t = s.time

        # 1. Determine current lead time (ground truth from disruption or regime)
        if self.disruption is not None:
            current_lt = self.disruption.current_lead_time(t)
        elif self.regime is not None:
            current_lt = self._get_regime_lead_time(t)
        else:
            current_lt = self.normal_lead_time
        s.lead_time = current_lt

        # 2. Receive arrivals (orders placed lead_time periods ago)
        arrived = self._pipeline.receive(t)
        s.on_hand += arrived

        # 3. Generate demand (regime-dependent in Phase 5)
        if self.regime is not None:
            current_demand_mean = self._get_regime_demand_mean(t)
        else:
            current_demand_mean = self.demand_mean
        demand = self.rng.poisson(current_demand_mean)
        s.demand = float(demand)

        # 4. Fulfill demand (lost-sales)
        sales = min(s.on_hand, demand)
        s.sales = sales
        s.lost_sales = max(0.0, demand - sales)
        s.on_hand -= sales

        # 5. Revenue
        s.revenue = sales * REVENUE_PER_UNIT

        # 6. Place order
        order_qty = max(0.0, order_qty)
        s.order_quantity = order_qty
        if order_qty > 0:
            s.ordering_cost = ORDERING_FIXED_COST + ORDERING_VARIABLE_COST * order_qty
        else:
            s.ordering_cost = 0.0

        # 7. Pipeline update — the order arrives in period (t + current_lt),
        #    per the declared PipelineSchedule semantics.  Orders already in
        #    transit are unaffected by a lead-time change.
        self._pipeline.place(t, current_lt, order_qty)

        # 8. Pipeline inventory = sum of in-transit orders
        s.pipeline = self._pipeline.outstanding()

        # 9. Costs
        s.holding_cost = HOLDING_COST_PER_UNIT * s.on_hand
        s.stockout_cost = STOCKOUT_COST_PER_UNIT * s.lost_sales

        # 10. Period profit
        s.period_profit = s.revenue - s.ordering_cost - s.holding_cost - s.stockout_cost
        s.cumulative_profit += s.period_profit

        # 11. Advance time and update lead_time
        s.time = t + 1
        if self.disruption is not None:
            s.lead_time = self.disruption.current_lead_time(s.time)
        elif self.regime is not None:
            s.lead_time = self._get_regime_lead_time(s.time)
        else:
            s.lead_time = self.normal_lead_time

        # Return a snapshot
        return InventoryState(
            time=s.time,
            on_hand=s.on_hand,
            pipeline=s.pipeline,
            demand=s.demand,
            sales=s.sales,
            lost_sales=s.lost_sales,
            order_quantity=s.order_quantity,
            lead_time=s.lead_time,
            revenue=s.revenue,
            ordering_cost=s.ordering_cost,
            holding_cost=s.holding_cost,
            stockout_cost=s.stockout_cost,
            period_profit=s.period_profit,
            cumulative_profit=s.cumulative_profit,
        )
