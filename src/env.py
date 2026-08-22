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

    This is the true decision upper bound (non-causal, non-deployable reference).
    It is strictly better than PerfectSemantic (which uses perfect interpretation
    but still applies a causal controller).
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

        # Build correct arrival matrix matching env's FIFO queue
        A = np.zeros((H, H))
        normal_lt = (
            P5_NORMAL_LEAD_TIME if self.regime is not None
            else (self.disruption.normal_lead_time if self.disruption else 2)
        )
        queue = list(range(normal_lt))
        for t in range(H):
            arrived = queue.pop(0) if queue else -1
            if self.regime is not None:
                lt = self._get_regime_lead_time(t)
            elif self.disruption is not None:
                lt = self.disruption.current_lead_time(t)
            else:
                lt = normal_lt
            while len(queue) < lt:
                queue.insert(0, -1)
            queue.append(t)
            if 0 <= arrived < H:
                A[t, arrived] = 1.0

        # Variables: order[H], y[H] (binary), on_hand[H], sales[H]
        n = 4 * H

        # Objective: minimize total cost
        c = np.zeros(n)
        for t in range(H):
            c[t] = ORDERING_VARIABLE_COST            # variable ordering cost
            c[H + t] = ORDERING_FIXED_COST            # fixed ordering cost (binary)
            c[2 * H + t] = HOLDING_COST_PER_UNIT      # holding cost
            c[3 * H + t] = -(REVENUE_PER_UNIT + STOCKOUT_COST_PER_UNIT)  # revenue + stockout

        # Integrality: 0=continuous, 2=binary
        integrality = np.zeros(n)
        integrality[H:2 * H] = 2  # y variables are binary

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
        # order[t] <= M * y[t]
        M = 200.0
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
            b_ub[t] = 0.0

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

        if result.success:
            return [max(0.0, round(result.x[t], 1)) for t in range(H)]
        else:
            return self._fallback_greedy(demands)

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
        self._queue: list[float] = [0.0] * self.normal_lead_time

        # Phase 4: Build the ASSUMED disruption for forward planning
        if assumed_lt_increase is not None and disruption is not None:
            self._assumed_disruption = SupplierDisruption(
                start_time=assumed_start if assumed_start is not None else disruption.start_time,
                duration=assumed_duration if assumed_duration is not None else disruption.duration,
                normal_lead_time=disruption.normal_lead_time,
                disrupted_lead_time=disruption.normal_lead_time + assumed_lt_increase,
            )
        else:
            self._assumed_disruption = disruption

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
        self, t_start: int, queue_state: list[float], orders: list[float]
    ) -> list[float]:
        """Simulate FIFO queue and return arrivals at each period."""
        queue = list(queue_state)
        arrivals = []
        for i, order in enumerate(orders):
            t = t_start + i
            arrived = queue.pop(0) if queue else 0.0
            arrivals.append(arrived)
            lt = self._get_lt(t)
            while len(queue) < lt:
                queue.insert(0, 0.0)
            queue.append(order)
        return arrivals

    def _build_arrival_model(
        self, t_start: int, queue_state: list[float], remaining: int
    ) -> tuple:
        """Build linear arrival model by simulation.

        Returns:
            a_fixed: arrivals from existing queue (length remaining)
            A: marginal arrival matrix (remaining x remaining)
        """
        a_fixed = np.array(
            self._simulate_arrivals(t_start, queue_state, [0.0] * remaining)
        )
        A = np.zeros((remaining, remaining))
        for j in range(remaining):
            orders_j = [0.0] * remaining
            orders_j[j] = 1.0
            arr_j = np.array(
                self._simulate_arrivals(t_start, queue_state, orders_j)
            )
            A[:, j] = arr_j - a_fixed
        return a_fixed, A

    def _solve_mpc(self, t: int, on_hand: float, remaining: int) -> float:
        """Solve receding-horizon LP for remaining periods. Returns first order."""
        from scipy.optimize import linprog

        R = remaining
        a_fixed, A_mat = self._build_arrival_model(t, self._queue, R)
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
        A_ub = np.zeros((2 * R, n_vars))
        b_ub = np.zeros(2 * R)
        for i in range(R):
            A_ub[i, 2 * R + i] = 1.0  # sales[i]
            if i > 0:
                A_ub[i, R + (i - 1)] = -1.0  # -on_hand[i-1]
            for j in range(R):
                if A_mat[i, j] > 0.5:
                    A_ub[i, j] = -1.0  # -arrivals
            b_ub[i] = a_fixed[i]

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

        # Pop front (match env's FIFO logic)
        if self._queue:
            self._queue.pop(0)

        # Get current lead time
        lt = self._get_lt(t)

        # Solve receding-horizon LP
        remaining = self.horizon - t
        if remaining <= 0:
            return 0.0
        order = self._solve_mpc(t, on_hand, remaining)

        # Update internal queue (match env's FIFO logic)
        while len(self._queue) < lt:
            self._queue.insert(0, 0.0)
        self._queue.append(order)

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
        self._arrivals: list[float] = []  # queue of arriving orders
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
        self._arrivals = [0.0] * init_lt  # empty pipeline
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
        arrived = self._arrivals.pop(0) if self._arrivals else 0.0
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

        # 7. Pipeline update — pad queue to (current_lt - 1) waiting slots,
        #    then append the new order (arrives after current_lt periods).
        while len(self._arrivals) < current_lt:
            self._arrivals.insert(0, 0.0)
        self._arrivals.append(order_qty)

        # 8. Pipeline inventory = sum of in-transit orders
        s.pipeline = sum(self._arrivals)

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
