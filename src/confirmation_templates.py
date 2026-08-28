"""Phase 6 confirmation warning templates.

Created BEFORE any LLM evaluation. These are genuinely new warnings
with different wording from the original 18 Phase-5 templates.

Regime semantics match the original benchmark:
- supplier_delay: lead time deterioration at t~20
- demand_surge: demand increase at t~20
- normal: no disruption expected

Ambiguity labels assigned before evaluation:
- clear: specific parameters stated (LT numbers, timeframes, demand levels)
- moderate: partial information, some parameters missing or hedged
- vague: general concern signals without specific operational parameters
"""

from src.events import Regime

CONFIRMATION_TEMPLATES = [
    # =========================================================================
    # SUPPLIER DELAY — CLEAR (4 variants)
    # =========================================================================
    {
        "template_id": "cd_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "URGENT: Our primary warehouse is shutting down for maintenance "
            "starting period 20. Lead times will increase from 4 to 8 periods "
            "for approximately 10 periods. Plan inventory accordingly."
        ),
    },
    {
        "template_id": "cd_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Confirmed: Logistics partner reports a capacity constraint. "
            "Effective period 20, delivery schedules extend from 4 to 8 periods. "
            "Disruption expected to last 10 periods before normalization."
        ),
    },
    {
        "template_id": "cd_clear_3",
        "ambiguity_level": "clear",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "NOTICE: Transportation route realignment beginning period 20. "
            "Current lead time of 4 periods will rise to 8 periods. "
            "Duration: approximately 10 periods. Orders placed before period 20 "
            "are unaffected."
        ),
    },
    {
        "template_id": "cd_clear_4",
        "ambiguity_level": "clear",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "ALERT: Production facility operating at reduced capacity. "
            "From period 20 onward, supplier lead time doubles from 4 to 8 periods. "
            "Expected recovery in approximately 10 periods. "
            "Advance orders recommended."
        ),
    },

    # =========================================================================
    # SUPPLIER DELAY — MODERATE (4 variants)
    # =========================================================================
    {
        "template_id": "cd_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Supplier update: Some disruption expected around period 20. "
            "Lead times may increase significantly — possibly doubling. "
            "Duration unclear but likely several weeks."
        ),
    },
    {
        "template_id": "cd_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "We have been notified of potential delays from our main distributor. "
            "Starting around period 20, deliveries could take longer than usual. "
            "The extent of the delay has not been fully communicated yet."
        ),
    },
    {
        "template_id": "cd_moderate_3",
        "ambiguity_level": "moderate",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Operations advisory: A staffing issue at our supplier may affect "
            "delivery schedules starting mid-quarter. Lead times might extend "
            "by several periods. We are monitoring the situation."
        ),
    },
    {
        "template_id": "cd_moderate_4",
        "ambiguity_level": "moderate",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Heads up from procurement: Our secondary supplier is experiencing "
            "equipment issues. This could push out deliveries by a few weeks "
            "beginning around period 20. Specifics to follow."
        ),
    },

    # =========================================================================
    # SUPPLIER DELAY — VAGUE (4 variants)
    # =========================================================================
    {
        "template_id": "cd_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "There are rumblings in the supply chain that things might slow down "
            "in the coming weeks. Nothing confirmed yet but worth keeping an eye on."
        ),
    },
    {
        "template_id": "cd_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Market intelligence suggests possible upstream complications. "
            "No concrete timeline or impact assessment available at this time."
        ),
    },
    {
        "template_id": "cd_vague_3",
        "ambiguity_level": "vague",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "Industry reports indicate some logistics providers are facing "
            "capacity constraints. May affect delivery schedules broadly. "
            "Details pending."
        ),
    },
    {
        "template_id": "cd_vague_4",
        "ambiguity_level": "vague",
        "regime": Regime.SUPPLIER_DELAY,
        "text": (
            "General supply chain sentiment has turned cautious. "
            "Several vendors mentioning potential slowdowns. "
            "Nothing actionable yet."
        ),
    },

    # =========================================================================
    # DEMAND SURGE — CLEAR (4 variants)
    # =========================================================================
    {
        "template_id": "cs_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "CONFIRMED: Major client contract signed. Starting period 20, "
            "demand will increase from 8 to 14 units per period. "
            "This elevated demand is expected to persist for 10 periods. "
            "Ensure adequate stock levels."
        ),
    },
    {
        "template_id": "cs_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Sales update: New distribution agreement effective period 20. "
            "Order volume will rise from 8 to approximately 14 units per period "
            "for the next 10 periods. Increase replenishment orders."
        ),
    },
    {
        "template_id": "cs_clear_3",
        "ambiguity_level": "clear",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Demand forecast revision: Period 20 onward, customer orders expected "
            "at 14 units per period versus current 8. Elevated demand for "
            "10 periods. Recommend immediate procurement adjustment."
        ),
    },
    {
        "template_id": "cs_clear_4",
        "ambiguity_level": "clear",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "NOTICE: Seasonal promotion campaign launches period 20. "
            "Projected demand surge from 8 to 14 units per period. "
            "Campaign runs for 10 periods. Pre-position inventory."
        ),
    },

    # =========================================================================
    # DEMAND SURGE — MODERATE (4 variants)
    # =========================================================================
    {
        "template_id": "cs_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Sales pipeline update: A large opportunity is closing. "
            "If it comes through around period 20, demand could jump "
            "significantly above current levels. Exact volume uncertain."
        ),
    },
    {
        "template_id": "cs_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Market signals suggest a demand uptick beginning mid-quarter. "
            "Customer inquiries are up. We may see order volumes increase "
            "by 50 to 80 percent for several weeks."
        ),
    },
    {
        "template_id": "cs_moderate_3",
        "ambiguity_level": "moderate",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Commercial team reports: Two major RFPs are in final stages. "
            "If awarded, expect substantially higher order volumes starting "
            "around period 20. Quantities not yet finalized."
        ),
    },
    {
        "template_id": "cs_moderate_4",
        "ambiguity_level": "moderate",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Customer advisory: Key account is accelerating their purchase "
            "schedule. Demand may increase materially in the coming weeks. "
            "Specific volumes to be confirmed."
        ),
    },

    # =========================================================================
    # DEMAND SURGE — VAGUE (4 variants)
    # =========================================================================
    {
        "template_id": "cs_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Market conditions look favorable. Consumer sentiment is rising "
            "and there are early signs of increased purchasing activity. "
            "No specific forecast changes yet."
        ),
    },
    {
        "template_id": "cs_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Sales team is seeing more inbound interest than usual. "
            "Hard to tell if this is a blip or a trend. "
            "Will update when we have more clarity."
        ),
    },
    {
        "template_id": "cs_vague_3",
        "ambiguity_level": "vague",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Industry outlook has improved. Several analysts are revising "
            "demand forecasts upward. Exact impact on our product line unclear."
        ),
    },
    {
        "template_id": "cs_vague_4",
        "ambiguity_level": "vague",
        "regime": Regime.DEMAND_SURGE,
        "text": (
            "Competitor exit from the market may redirect customer demand "
            "our way. Volume impact unknown. Monitoring weekly."
        ),
    },

    # =========================================================================
    # NORMAL — CLEAR (4 variants)
    # =========================================================================
    {
        "template_id": "cn_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime.NORMAL,
        "text": (
            "STATUS UPDATE: All supply chain operations running normally. "
            "No disruptions expected in the next 20 periods. "
            "Lead times stable at 4 periods. Demand forecast unchanged at "
            "8 units per period."
        ),
    },
    {
        "template_id": "cn_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime.NORMAL,
        "text": (
            "Clear operations report: Supplier capacity fully available. "
            "No known constraints through period 40. "
            "Standard lead times of 4 periods confirmed. "
            "Demand expected at baseline levels."
        ),
    },
    {
        "template_id": "cn_clear_3",
        "ambiguity_level": "clear",
        "regime": Regime.NORMAL,
        "text": (
            "Confirmed: All logistics routes operating at normal capacity. "
            "No delays anticipated. Lead time remains 4 periods. "
            "Order volume forecast stable at 8 units per period."
        ),
    },
    {
        "template_id": "cn_clear_4",
        "ambiguity_level": "clear",
        "regime": Regime.NORMAL,
        "text": (
            "Supply chain status green. Warehouse operations normal. "
            "Delivery performance on track. No changes to lead time "
            "or demand expectations through the planning horizon."
        ),
    },

    # =========================================================================
    # NORMAL — MODERATE (4 variants)
    # =========================================================================
    {
        "template_id": "cn_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime.NORMAL,
        "text": (
            "Routine update: Operations continuing as expected. "
            "Minor seasonal fluctuations possible but nothing material. "
            "Lead times and demand remain at normal levels."
        ),
    },
    {
        "template_id": "cn_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime.NORMAL,
        "text": (
            "Weekly status: All systems nominal. Supplier relationships stable. "
            "No changes to delivery schedules reported. "
            "Standard planning assumptions should hold."
        ),
    },
    {
        "template_id": "cn_moderate_3",
        "ambiguity_level": "moderate",
        "regime": Regime.NORMAL,
        "text": (
            "Operations note: Minor maintenance scheduled but will not affect "
            "delivery timelines. Lead times and demand forecast unchanged. "
            "Continue with standard replenishment."
        ),
    },
    {
        "template_id": "cn_moderate_4",
        "ambiguity_level": "moderate",
        "regime": Regime.NORMAL,
        "text": (
            "Planning update: No significant changes to report. "
            "Supply and demand conditions remain stable. "
            "Current inventory policy adequate."
        ),
    },

    # =========================================================================
    # NORMAL — VAGUE (4 variants)
    # =========================================================================
    {
        "template_id": "cn_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime.NORMAL,
        "text": (
            "Things are generally quiet on the supply front. "
            "Nothing unusual to report at this time. "
            "We will keep you posted if anything changes."
        ),
    },
    {
        "template_id": "cn_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime.NORMAL,
        "text": (
            "Market remains stable. No major developments affecting "
            "our operations or customer demand patterns."
        ),
    },
    {
        "template_id": "cn_vague_3",
        "ambiguity_level": "vague",
        "regime": Regime.NORMAL,
        "text": (
            "General business conditions appear steady. "
            "No red flags from any of our supplier or customer touchpoints."
        ),
    },
    {
        "template_id": "cn_vague_4",
        "ambiguity_level": "vague",
        "regime": Regime.NORMAL,
        "text": (
            "Periodic review shows everything running smoothly. "
            "No deviations from expected operational parameters."
        ),
    },
]
