"""A retrievable operational-report feed for the evidence-selection study.

The rest of this benchmark hands a controller one preselected warning.  Real
operational text arrives as a feed covering many sites and suppliers, most of
it about somebody else, some of it out of date.  Choosing what to act on is an
information access problem, and it is the step this module makes explicit.

**Separation of concerns.**  The three stages are deliberately disjoint:

* *Retrieval* decides which reports are about **this operator's node** and are
  **current**.  Relevance is therefore determined by the query -- entity and
  recency -- and a competent ranker can solve it.
* *Interpretation* decides which **regime** the retrieved text implies.  The
  regime is never in the query; it is what the interpreter extracts.
* *Control* acts on the resulting belief.

That split is what makes the retrieval/utility dissociation measurable.  A
report about a different site that describes a *different* regime is an
actively harmful retrieval error; an off-topic HR memo is merely useless; a
report about a different site that happens to describe the *correct* regime is
a lucky error.  All three are equally non-relevant under standard graded
relevance, so two systems with the same nDCG can have very different
downstream utility.

Document kinds
--------------
``relevant``          current report about the operator's own node
``wrong_entity``      same operational register, a different site or supplier
``stale``             the operator's node, but an event already resolved
``offtopic``          routine corporate traffic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.events import Regime, REGIME_WARNING_TEMPLATES
from src.confirmation_templates import CONFIRMATION_TEMPLATES

KIND_RELEVANT = "relevant"
KIND_WRONG_ENTITY = "wrong_entity"
KIND_STALE = "stale"
KIND_OFFTOPIC = "offtopic"

# --- Entities ---------------------------------------------------------------
# The operator owns NODE_SELF. Every other node produces structurally identical
# traffic, so entity discrimination -- not topic detection -- is the retrieval
# problem.

@dataclass(frozen=True)
class Entity:
    node_id: str
    site: str
    supplier: str
    line: str

    def header(self) -> str:
        return (f"[node {self.node_id} | site {self.site} | supplier {self.supplier} "
                f"| line {self.line}]")

    def descriptor(self) -> str:
        return f"node {self.node_id} site {self.site} supplier {self.supplier} line {self.line}"


NODE_SELF = Entity("R3", "CEDAR-FALLS-DC", "NORTHBRIDGE-LOGISTICS", "7742")

OTHER_NODES = [
    Entity("R1", "ALDERGROVE-DC", "WESTMARK-FREIGHT", "3310"),
    Entity("R2", "BRIARWOOD-DC", "COASTLINE-CARRIERS", "5128"),
    Entity("R4", "DUNMORE-DC", "STONEGATE-HAULAGE", "9014"),
    Entity("R5", "ELMHURST-DC", "TRAILHEAD-TRANSPORT", "6607"),
    Entity("R6", "FAIRBANK-DC", "QUARRYSIDE-SHIPPING", "2285"),
]

CURRENT_WINDOW = "planning window W-42"
PAST_WINDOW = "planning window W-31"

# The operator's standing information need: their own node, current window.
# The regime is deliberately absent -- that is the interpreter's job.
OPERATIONAL_QUERY = (
    f"current operational status for {NODE_SELF.descriptor()} in {CURRENT_WINDOW}: "
    "inbound supply, lead time and demand conditions affecting replenishment now"
)

_OFFTOPIC_TEMPLATES = [
    "IT notice: the ERP reporting module will be unavailable for scheduled "
    "maintenance this weekend. Inventory dashboards may lag by one period.",
    "HR bulletin: open enrolment for benefits closes at the end of the month. "
    "Please complete your elections in the employee portal.",
    "Facilities: the parking structure will be resurfaced over the next two "
    "weeks. Overflow capacity is available in the adjacent lot.",
    "Finance: quarterly expense reports are due. Please code freight accruals "
    "to the updated cost centre schedule.",
    "Marketing update: the autumn campaign creative review has been moved. "
    "Regional teams should submit assets on the revised schedule.",
    "Training: the annual safety certification window is open. Warehouse staff "
    "must complete the refresher module before the end of the period.",
    "Legal: an updated supplier code of conduct has been published. No action "
    "is required for existing contracts at this time.",
    "IT security: multi-factor authentication will be enforced for all vendor "
    "portal accounts. Contact the service desk for enrolment.",
    "Sustainability: scope 3 emissions reporting boundaries have been revised. "
    "Logistics data will be re-baselined for the annual disclosure.",
    "Office services: the second-floor printer has been replaced. Please "
    "reinstall the driver from the software catalogue.",
    "Travel policy: advance approval is now required for international trips. "
    "Submit requests at least two weeks before departure.",
    "Payroll: the pay calendar for the coming quarter has been published. "
    "There is no change to the deposit schedule.",
]


@dataclass
class Document:
    doc_id: str
    text: str
    kind: str
    regime: Optional[str]      # regime the text describes, if any
    entity: Optional[str]      # node the text is about, if any
    window: Optional[str]
    source: str

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "kind": self.kind, "regime": self.regime,
            "entity": self.entity, "window": self.window, "source": self.source,
        }


@dataclass
class EpisodeCorpus:
    seed: int
    true_regime: str
    documents: list
    relevance: dict = field(default_factory=dict)
    query: str = OPERATIONAL_QUERY

    @property
    def doc_ids(self) -> list:
        return [d.doc_id for d in self.documents]

    @property
    def n_relevant(self) -> int:
        return sum(1 for v in self.relevance.values() if v > 0)

    def doc(self, doc_id: str) -> Document:
        return next(d for d in self.documents if d.doc_id == doc_id)

    def composition(self) -> dict:
        counts: dict = {}
        for d in self.documents:
            counts[d.kind] = counts.get(d.kind, 0) + 1
        return counts

    def misleading_ids(self) -> set:
        """Non-relevant documents that describe a regime other than the truth.

        These are the retrieval errors that actively move the controller the
        wrong way, as opposed to merely wasting an evidence slot.
        """
        return {
            d.doc_id for d in self.documents
            if self.relevance.get(d.doc_id, 0) == 0
            and d.regime is not None and d.regime != self.true_regime
        }


def _scoped(text: str, entity: Entity, window: str) -> str:
    """Attach entity and window metadata the way a real report feed would."""
    return f"{entity.header()} {window}. {text}"


def _regime_text_pool(held_out_only: bool = True, exclude_ids: Sequence[str] = ()) -> dict:
    pool: dict = {r: [] for r in Regime}
    excluded = set(exclude_ids)
    sources = [("confirmation", CONFIRMATION_TEMPLATES)]
    if not held_out_only:
        sources.append(("regime_train", REGIME_WARNING_TEMPLATES))
    for source, templates in sources:
        for tmpl in templates:
            tid = str(tmpl.get("template_id", ""))
            if tid in excluded:
                continue
            regime = tmpl["regime"]
            regime = regime if isinstance(regime, Regime) else Regime(regime)
            pool[regime].append((f"{source}:{tid}", tmpl["text"], source))
    return pool


def build_episode_corpus(
    seed: int,
    true_regime: Regime,
    n_relevant: int = 2,
    n_wrong_entity: int = 10,
    n_stale: int = 3,
    n_offtopic: int = 9,
    exclude_template_ids: Sequence[str] = (),
    held_out_only: bool = True,
) -> EpisodeCorpus:
    """Assemble one episode's candidate pool with its relevance judgments."""
    rng = np.random.default_rng(seed)
    pool = _regime_text_pool(held_out_only=held_out_only, exclude_ids=exclude_template_ids)
    all_regimes = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]

    documents: list = []
    relevance: dict = {}

    def take(entries, count):
        if not entries or count <= 0:
            return []
        idx = rng.choice(len(entries), size=min(count, len(entries)), replace=False)
        return [entries[int(i)] for i in np.atleast_1d(idx)]

    # Relevant: the operator's own node, current window, true regime.
    for tid, text, source in take(pool.get(true_regime, []), n_relevant):
        doc = Document(
            f"rel::{tid}", _scoped(text, NODE_SELF, CURRENT_WINDOW), KIND_RELEVANT,
            true_regime.value, NODE_SELF.node_id, CURRENT_WINDOW, source,
        )
        documents.append(doc)
        relevance[doc.doc_id] = 1

    # Wrong entity: identical register, a different node. Regimes are spread
    # across all three so that some of these errors are actively misleading and
    # some are accidentally correct.
    wrong_entity_specs = []
    for i in range(n_wrong_entity):
        node = OTHER_NODES[i % len(OTHER_NODES)]
        regime = all_regimes[i % len(all_regimes)]
        wrong_entity_specs.append((node, regime))
    for i, (node, regime) in enumerate(wrong_entity_specs):
        picked = take(pool.get(regime, []), 1)
        if not picked:
            continue
        tid, text, source = picked[0]
        doc = Document(
            f"ent::{node.node_id}::{i}::{tid}",
            _scoped(text, node, CURRENT_WINDOW), KIND_WRONG_ENTITY,
            regime.value, node.node_id, CURRENT_WINDOW, source,
        )
        documents.append(doc)
        relevance[doc.doc_id] = 0

    # Stale: the operator's own node, but a window that has closed.
    stale_regimes = [r for r in all_regimes]
    for i in range(n_stale):
        regime = stale_regimes[i % len(stale_regimes)]
        picked = take(pool.get(regime, []), 1)
        if not picked:
            continue
        tid, text, source = picked[0]
        doc = Document(
            f"sta::{i}::{tid}",
            _scoped(f"RESOLVED / CLOSED: {text}", NODE_SELF, PAST_WINDOW), KIND_STALE,
            regime.value, NODE_SELF.node_id, PAST_WINDOW, source,
        )
        documents.append(doc)
        relevance[doc.doc_id] = 0

    # Off-topic corporate traffic, attributed to the operator's own node so it
    # cannot be filtered on entity alone.
    for i, text in enumerate(
        [t for _, t in take(list(enumerate(_OFFTOPIC_TEMPLATES)), n_offtopic)]
    ):
        doc = Document(
            f"off::{i}", _scoped(text, NODE_SELF, CURRENT_WINDOW), KIND_OFFTOPIC,
            None, NODE_SELF.node_id, CURRENT_WINDOW, "offtopic_template",
        )
        documents.append(doc)
        relevance[doc.doc_id] = 0

    order = rng.permutation(len(documents))
    documents = [documents[int(i)] for i in order]

    return EpisodeCorpus(
        seed=seed, true_regime=true_regime.value,
        documents=documents, relevance=relevance,
    )
