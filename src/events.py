"""Ground-truth event definitions.

The simulator controls reality. Events are stored ground-truth parameters
that the simulator applies directly. The LLM never determines whether an
event occurs — it only interprets textual observations of it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SupplierDisruption:
    """Ground-truth supplier disruption event.

    This is the *true* state of the world. The simulator applies these
    parameters directly to inventory dynamics. The LLM receives only a
    textual observation and must infer these values imperfectly.
    """

    start_time: int
    duration: int
    normal_lead_time: int
    disrupted_lead_time: int

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration

    def current_lead_time(self, t: int) -> int:
        """Return the actual lead time at period t."""
        if self.start_time <= t < self.end_time:
            return self.disrupted_lead_time
        return self.normal_lead_time

    def is_active(self, t: int) -> bool:
        return self.start_time <= t < self.end_time


# Default ground-truth event used across all experiments
DEFAULT_DISRUPTION = SupplierDisruption(
    start_time=18,
    duration=8,
    normal_lead_time=2,
    disrupted_lead_time=5,
)


# True semantic parameters of the default disruption (used for interpretation targets)
TRUE_LT_INCREASE = DEFAULT_DISRUPTION.disrupted_lead_time - DEFAULT_DISRUPTION.normal_lead_time  # 3
TRUE_DURATION = DEFAULT_DISRUPTION.duration  # 8
TRUE_EVENT_TYPE = "supply_disruption"


# --- Textual warnings --------------------------------------------------------
# Templates now carry template_id and ambiguity_level metadata.
# ambiguity_level: "clear", "moderate", "vague"

WARNING_TEMPLATES: list[dict] = [
    {
        "template_id": "clear_1",
        "ambiguity_level": "clear",
        "text": (
            "Our distribution team has confirmed a significant congestion at the "
            "supplier facility. Shipments expected over the next 8 periods will "
            "experience delays of approximately 3 additional business days."
        ),
    },
    {
        "template_id": "clear_2",
        "ambiguity_level": "clear",
        "text": (
            "URGENT: Supplier has informed us of a facility disruption beginning "
            "immediately. All orders placed in the next 8 periods will have an "
            "extended lead time of 5 days instead of the normal 2 days."
        ),
    },
    {
        "template_id": "clear_3",
        "ambiguity_level": "clear",
        "text": (
            "We have received direct notification from our supplier that a "
            "production issue will delay deliveries. Expect lead times to "
            "increase from 2 to 5 periods for the next 8 ordering cycles."
        ),
    },
    {
        "template_id": "moderate_1",
        "ambiguity_level": "moderate",
        "text": (
            "Our distribution team is seeing congestion at the supplier facility. "
            "Several upcoming shipments may arrive later than usual."
        ),
    },
    {
        "template_id": "moderate_2",
        "ambiguity_level": "moderate",
        "text": (
            "We are currently experiencing operational delays and cannot guarantee "
            "normal shipment timing over the next several deliveries."
        ),
    },
    {
        "template_id": "moderate_3",
        "ambiguity_level": "moderate",
        "text": (
            "The supplier has flagged potential capacity constraints that could "
            "affect our delivery schedule. We recommend building safety stock."
        ),
    },
    {
        "template_id": "moderate_4",
        "ambiguity_level": "moderate",
        "text": (
            "A key supplier is experiencing staffing challenges at their main "
            "warehouse. We may see 2-4 extra days on inbound shipments for a "
            "limited period."
        ),
    },
    {
        "template_id": "vague_1",
        "ambiguity_level": "vague",
        "text": (
            "Recent logistics issues may affect delivery schedules during the "
            "coming week, although the exact extent is not yet known."
        ),
    },
    {
        "template_id": "vague_2",
        "ambiguity_level": "vague",
        "text": (
            "Heads up — there have been some rumblings about supplier reliability. "
            "Nothing confirmed yet, but worth keeping an eye on."
        ),
    },
    {
        "template_id": "vague_3",
        "ambiguity_level": "vague",
        "text": (
            "Market intelligence suggests possible upstream disruptions in our "
            "supply chain. Details remain unclear at this time."
        ),
    },
    {
        "template_id": "vague_4",
        "ambiguity_level": "vague",
        "text": (
            "Some of our peer companies are reporting delays from shared suppliers. "
            "We have not been directly impacted yet but the risk is elevated."
        ),
    },
]

# Backward-compatible alias
for _t in WARNING_TEMPLATES:
    _t["id"] = _t["template_id"]
    _t["clarity"] = _t["ambiguity_level"]


# =============================================================================
# Phase 5: Multi-Regime System
# =============================================================================
# The hidden regime Z determines the operational environment.
# The controller does NOT know Z at warning time; a warning provides partial info.
# PerfectSemantic knows Z; NoInfo uses the prior over Z.
# =============================================================================

from enum import Enum


class Regime(str, Enum):
    """Hidden operational regime. Determined at episode start, not by the warning."""
    NORMAL = "normal"
    SUPPLIER_DELAY = "supplier_delay"
    DEMAND_SURGE = "demand_surge"
    SUPPLIER_CAPACITY_DROP = "supplier_capacity_drop"


# Regime prior — the NoInfo controller knows this distribution
REGIME_PRIOR: dict[str, float] = {
    Regime.NORMAL: 0.35,
    Regime.SUPPLIER_DELAY: 0.35,
    Regime.DEMAND_SURGE: 0.30,
    Regime.SUPPLIER_CAPACITY_DROP: 0.0,
}

# Operational parameters for each regime
# Normal LT=4, Demand~Poisson(8), Event starts at t=20, duration=10
REGIME_PARAMS: dict[str, dict] = {
    Regime.NORMAL: {
        "description": "No disruption. Normal operations.",
        "lead_time_increase": 0,
        "demand_multiplier": 1.0,
    },
    Regime.SUPPLIER_DELAY: {
        "description": "Supplier facility congestion increases lead time.",
        "lead_time_increase": 4,
        "demand_multiplier": 1.0,
    },
    Regime.DEMAND_SURGE: {
        "description": "Strong customer purchasing increases demand.",
        "lead_time_increase": 0,
        "demand_multiplier": 1.75,
    },
    Regime.SUPPLIER_CAPACITY_DROP: {
        "description": "Supplier capacity reduction. Not used in Phase 5 env; see Phase 9 divergent topology.",
        "lead_time_increase": 0,
        "demand_multiplier": 1.0,
    },
}

# Phase 5 environment constants
P5_HORIZON = 40
P5_NORMAL_LEAD_TIME = 4
P5_DEMAND_MEAN = 8.0
P5_EVENT_START = 20
P5_EVENT_DURATION = 10
P5_WARNING_TIME = 12

# Regime warning templates: regime -> ambiguity_level -> list of templates
# Each template is a dict with template_id, ambiguity_level, text, and true_regime.
# Templates do NOT leak the regime label directly — they describe operational signals.
REGIME_WARNING_TEMPLATES: list[dict] = [
    # --- Supplier delay warnings ---
    {
        "template_id": "delay_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "CONFIRMED: Our supplier has notified us of facility congestion "
            "that will increase delivery lead times from 4 to 8 periods, "
            "beginning around period 20 and lasting approximately 10 periods. "
            "All inbound shipments will be affected during this window."
        ),
    },
    {
        "template_id": "delay_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Urgent supply chain alert: A production issue at our primary supplier "
            "will double our normal lead time to approximately 8 periods for the "
            "next 10 ordering cycles starting around period 20."
        ),
    },
    {
        "template_id": "delay_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Our logistics team is experiencing congestion at the supplier facility. "
            "Several upcoming shipments may arrive later than usual, potentially "
            "affecting delivery schedules in the coming weeks."
        ),
    },
    {
        "template_id": "delay_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "The supplier has flagged staffing challenges that could delay "
            "shipments. We may see extended delivery times for a limited period. "
            "Recommend building safety stock as a precaution."
        ),
    },
    {
        "template_id": "delay_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "There have been some rumblings about supplier reliability. Nothing "
            "confirmed yet, but logistics may be slower than usual if the "
            "situation develops further."
        ),
    },
    {
        "template_id": "delay_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Market intelligence suggests possible upstream disruptions in our "
            "supply chain. Delivery timing could be affected, though details "
            "remain unclear."
        ),
    },
    # --- Demand surge warnings ---
    {
        "template_id": "surge_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "CONFIRMED: Our sales team has secured a large contract that will "
            "increase demand from the normal 8 units/period to approximately "
            "14 units/period, starting around period 20 for 10 periods. "
            "All procurement plans must be updated accordingly."
        ),
    },
    {
        "template_id": "surge_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Urgent demand alert: Two major customers have confirmed unusually "
            "strong purchasing plans. Expect average demand to rise to roughly "
            "14 units per period for the next 10 periods beginning around "
            "period 20."
        ),
    },
    {
        "template_id": "surge_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Several major customers have indicated unusually strong near-term "
            "purchasing plans. Final order quantities remain uncertain but "
            "demand could be significantly above normal levels in the coming weeks."
        ),
    },
    {
        "template_id": "surge_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "The sales pipeline shows a substantial uptick in customer inquiries "
            "and pre-orders. We should prepare for above-normal demand, though "
            "the exact scale is not yet confirmed."
        ),
    },
    {
        "template_id": "surge_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Our sales team is hearing about increased customer interest across "
            "several accounts. Nothing firm yet, but demand could pick up "
            "noticeably if these opportunities materialize."
        ),
    },
    {
        "template_id": "surge_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Market conditions suggest stronger-than-usual customer purchasing "
            "in the near term. The extent is uncertain, but elevated demand "
            "is a real possibility."
        ),
    },
    # --- Normal / weak-signal warnings ---
    {
        "template_id": "normal_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime.NORMAL,
        "text": (
            "Operations remain fully on schedule. No supplier issues, demand "
            "changes, or logistics disruptions have been identified. Routine "
            "variability in delivery timing continues as normal."
        ),
    },
    {
        "template_id": "normal_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime.NORMAL,
        "text": (
            "All supply chain metrics are within normal parameters. Supplier "
            "facilities are operating normally, and customer demand patterns "
            "show no material changes."
        ),
    },
    {
        "template_id": "normal_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime.NORMAL,
        "text": (
            "Operations remain broadly on schedule, although routine variability "
            "in delivery timing is still possible. No unusual risk signals "
            "have been reported this cycle."
        ),
    },
    {
        "template_id": "normal_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime.NORMAL,
        "text": (
            "Supplier communications indicate normal operational capacity. "
            "Some minor scheduling fluctuations are expected but nothing "
            "outside standard operating parameters."
        ),
    },
    {
        "template_id": "normal_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime.NORMAL,
        "text": (
            "General market outlook appears stable. No significant supply or "
            "demand disruptions have been flagged by our monitoring systems, "
            "though conditions can change quickly."
        ),
    },
    {
        "template_id": "normal_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime.NORMAL,
        "text": (
            "Routine operational update: no material changes to report. "
            "Supply and demand conditions remain within expected ranges."
        ),
    },
]

# Backward-compatible aliases for regime templates
for _t in REGIME_WARNING_TEMPLATES:
    _t["id"] = _t["template_id"]
    _t["clarity"] = _t["ambiguity_level"]
