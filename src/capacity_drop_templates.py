"""Phase 9A: Linguistic templates for binary SupplierCapacityDrop task.

Binary classification: Normal vs SupplierCapacityDrop.
Templates have template_id, ambiguity_level, regime, and text fields.
"""
from enum import Enum


class Regime9(str, Enum):
    NORMAL = "normal"
    SUPPLIER_CAPACITY_DROP = "supplier_capacity_drop"


CAPACITY_DROP_TEMPLATES = [
    # --- Clear templates ---
    {
        "template_id": "cd_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "CONFIRMED: Factory 4 has informed us of a major equipment failure "
            "that will reduce production capacity from 90 units/period to approximately "
            "30 units/period, starting around period 10 and lasting through period 25. "
            "All downstream allocation plans must be adjusted accordingly."
        ),
    },
    {
        "template_id": "cd_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "URGENT: Our primary factory is cutting production capacity by two-thirds "
            "due to a critical machinery breakdown. Effective immediately, throughput "
            "drops from 90 to 30 units/period for the next 15 periods."
        ),
    },
    {
        "template_id": "cd_clear_3",
        "ambiguity_level": "clear",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "We have received direct notification from Factory 4 that a production line "
            "shutdown will limit output to 30 units/period (down from normal 90) for "
            "periods 10 through 25. Plan for significantly reduced inbound shipments."
        ),
    },
    # --- Moderate templates ---
    {
        "template_id": "cd_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "Factory 4 is experiencing significant production constraints and may need "
            "to reduce output substantially. We are awaiting confirmation on the exact "
            "capacity reduction and expected duration."
        ),
    },
    {
        "template_id": "cd_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "Our manufacturing partner has flagged a potential capacity issue at their "
            "main facility. Production rates could drop well below normal levels for "
            "several weeks. Recommend adjusting procurement plans as a precaution."
        ),
    },
    {
        "template_id": "cd_moderate_3",
        "ambiguity_level": "moderate",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "There are reports of staffing shortages and equipment maintenance at the "
            "factory that supplies our distributors. Output may be significantly lower "
            "than normal for an extended period."
        ),
    },
    {
        "template_id": "cd_moderate_4",
        "ambiguity_level": "moderate",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "The supplier has flagged a production capacity constraint that will affect "
            "our delivery schedule. Factory output is expected to be well below normal "
            "for the foreseeable future."
        ),
    },
    # --- Vague templates ---
    {
        "template_id": "cd_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "There have been some rumblings about production issues at the factory. "
            "Nothing confirmed yet, but supply capacity could be affected if the "
            "situation develops further."
        ),
    },
    {
        "template_id": "cd_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "Market intelligence suggests possible production slowdowns at key suppliers. "
            "The extent is unclear, but delivery volumes could be impacted."
        ),
    },
    {
        "template_id": "cd_vague_3",
        "ambiguity_level": "vague",
        "regime": Regime9.SUPPLIER_CAPACITY_DROP,
        "text": (
            "Some of our peer companies are reporting reduced output from shared "
            "manufacturing facilities. We have not been directly impacted yet but "
            "the risk is elevated."
        ),
    },
]

# --- Normal templates (no capacity disruption) ---
NORMAL_TEMPLATES = [
    {
        "template_id": "cn_clear_1",
        "ambiguity_level": "clear",
        "regime": Regime9.NORMAL,
        "text": (
            "Operations remain fully on schedule. All factory facilities are operating "
            "at normal capacity with no disruptions identified. Production output "
            "continues at standard rates."
        ),
    },
    {
        "template_id": "cn_clear_2",
        "ambiguity_level": "clear",
        "regime": Regime9.NORMAL,
        "text": (
            "All supply chain metrics are within normal parameters. Factory production "
            "rates are at normal levels and no capacity constraints have been identified."
        ),
    },
    {
        "template_id": "cn_moderate_1",
        "ambiguity_level": "moderate",
        "regime": Regime9.NORMAL,
        "text": (
            "Operations remain broadly on schedule. Routine variability in production "
            "timing is still possible, but no unusual capacity risk signals have been "
            "reported this cycle."
        ),
    },
    {
        "template_id": "cn_moderate_2",
        "ambiguity_level": "moderate",
        "regime": Regime9.NORMAL,
        "text": (
            "Supplier communications indicate normal operational capacity. Some minor "
            "scheduling fluctuations are expected but nothing outside standard "
            "operating parameters."
        ),
    },
    {
        "template_id": "cn_vague_1",
        "ambiguity_level": "vague",
        "regime": Regime9.NORMAL,
        "text": (
            "General production outlook appears stable. No significant capacity "
            "disruptions have been flagged by our monitoring systems, though conditions "
            "can change quickly."
        ),
    },
    {
        "template_id": "cn_vague_2",
        "ambiguity_level": "vague",
        "regime": Regime9.NORMAL,
        "text": (
            "Routine operational update: no material changes to report. Factory output "
            "and supply conditions remain within expected ranges."
        ),
    },
]

ALL_CAPACITY_DROP_TEMPLATES = NORMAL_TEMPLATES + CAPACITY_DROP_TEMPLATES

TRAIN_TEMPLATE_IDS = [
    "cn_clear_1", "cn_clear_2", "cn_moderate_1",
    "cd_clear_1", "cd_clear_2", "cd_moderate_1",
    "cd_moderate_3", "cd_moderate_4",
    "cn_vague_1",
]

TRAIN_TEMPLATES = [t for t in ALL_CAPACITY_DROP_TEMPLATES if t["template_id"] in TRAIN_TEMPLATE_IDS]

TEST_TEMPLATE_IDS = [
    "cn_moderate_2", "cn_vague_2",
    "cd_clear_3", "cd_moderate_2",
    "cd_vague_1", "cd_vague_2", "cd_vague_3",
]

TEST_TEMPLATES = [t for t in ALL_CAPACITY_DROP_TEMPLATES if t["template_id"] in TEST_TEMPLATE_IDS]
