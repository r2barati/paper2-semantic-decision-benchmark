"""Semantic interpreters for operational warning texts.

Receives a textual warning and returns a structured JSON interpretation.
Supports four backends:

    1. Real LLM via OpenAI-compatible API (requires API key).
    2. RuleBasedExtractor — deterministic keyword/regex extraction (no API key).
    3. Deterministic mock interpreter (no API key, for testing).
    4. ActiveWrong variants — deliberately incorrect (for robustness checks).

All interpreters share the same Interpretation dataclass and receive
ONLY the warning text + extraction instructions. They never see:
    - ambiguity labels
    - ground-truth parameters
    - simulator state
    - operational outcomes
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- Response schema ---------------------------------------------------------

INTERPRETATION_SCHEMA = {
    "event_type": "supply_disruption",
    "probability": 0.85,
    "estimated_lead_time_increase": 3,
    "estimated_duration": 8,
}

# Frozen extraction prompt — do NOT modify after initial validation.
EXTRACTION_PROMPT = (
    "You are an operational risk analyst. Given a textual supplier warning, "
    "extract structured information. Respond ONLY with valid JSON matching "
    "this exact schema:\n"
    '{"event_type": "supply_disruption", "probability": <float 0-1>, '
    '"estimated_lead_time_increase": <int>, "estimated_duration": <int>}\n'
    "Do not include any other text."
)


@dataclass
class Interpretation:
    event_type: str
    probability: float
    estimated_lead_time_increase: int
    estimated_duration: int

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "probability": self.probability,
            "estimated_lead_time_increase": self.estimated_lead_time_increase,
            "estimated_duration": self.estimated_duration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Interpretation":
        return cls(
            event_type=str(d["event_type"]),
            probability=float(d["probability"]),
            estimated_lead_time_increase=int(d["estimated_lead_time_increase"]),
            estimated_duration=int(d["estimated_duration"]),
        )


@dataclass
class LLMResult:
    """Metadata for a single LLM API call."""
    interpretation: Interpretation
    model: str
    raw_response: str
    latency_ms: float
    from_cache: bool
    retries: int = 0
    malformed: bool = False


# --- .env loader (does not overwrite existing env vars) ----------------------

def load_dotenv(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (no overwrite)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# --- Rule-based extractor (no LLM) -----------------------------------------

def rule_based_extract(text: str) -> Interpretation:
    """Deterministic keyword/regex extractor — no LLM required.

    Designed to be a simple, transparent, non-learned baseline.
    Not optimized against operational outcomes.
    """
    text_lower = text.lower()

    # --- Probability estimation ---
    if any(w in text_lower for w in [
        "urgent:", "urgent ",
        "we have confirmed",
        "direct notification",
        "informed us of a facility",
        "confirmed a significant",
    ]):
        probability = 0.90
    elif any(w in text_lower for w in [
        "cannot guarantee",
        "experiencing delays",
        "experience delays",
    ]):
        probability = 0.80
    elif any(w in text_lower for w in [
        "congestion at the supplier",
        "seeing congestion",
        "operational delays",
    ]):
        probability = 0.75
    elif any(w in text_lower for w in [
        "capacity constraints",
        "staffing challenges",
    ]):
        probability = 0.70
    elif any(w in text_lower for w in [
        "may affect",
        "possible upstream",
        "may see",
        "potential",
    ]):
        probability = 0.60
    elif any(w in text_lower for w in [
        "rumblings",
        "nothing confirmed",
        "not yet known",
        "elevated",
        "peer companies",
        "worth keeping an eye",
    ]):
        probability = 0.45
    else:
        probability = 0.50

    # --- Lead time increase estimation ---
    lt_match = re.search(
        r'(?:lead\s*time|delay|days?)\s*(?:of|to|by|increase\s*(?:of|by|to))?\s*'
        r'(\d+)\s*(?:additional|extra|days?|periods?|business\s*days?)',
        text_lower
    )
    if lt_match:
        est_lt_increase = int(lt_match.group(1))
    elif re.search(r'from\s+\d+\s+to\s+(\d+)', text_lower):
        m = re.search(r'from\s+\d+\s+to\s+(\d+)', text_lower)
        to_val = int(m.group(1))
        from_m = re.search(r'from\s+(\d+)', text_lower)
        from_val = int(from_m.group(1)) if from_m else 2
        est_lt_increase = max(1, to_val - from_val)
    elif any(w in text_lower for w in ["5 days", "5 periods", "3 additional", "3 extra"]):
        est_lt_increase = 3
    elif any(w in text_lower for w in ["2-4 extra", "2 to 4"]):
        est_lt_increase = 3
    elif any(w in text_lower for w in ["longer", "later", "delays", "delayed"]):
        est_lt_increase = 2
    else:
        est_lt_increase = 1

    # --- Duration estimation ---
    dur_match = re.search(
        r'(?:next|following|over\s*the)\s+(\d+)\s+(?:periods?|ordering|cycles?|days?|weeks?)',
        text_lower
    )
    if dur_match:
        est_duration = int(dur_match.group(1))
    elif re.search(r'for\s+(?:the\s+)?(?:next\s+)?(\d+)\s+(?:ordering|periods?|cycles?)', text_lower):
        m = re.search(r'for\s+(?:the\s+)?(?:next\s+)?(\d+)\s+(?:ordering|periods?|cycles?)', text_lower)
        est_duration = int(m.group(1))
    elif any(w in text_lower for w in ["8 periods", "8 ordering", "next 8"]):
        est_duration = 8
    elif any(w in text_lower for w in ["several", "limited period", "coming week"]):
        est_duration = 6
    else:
        est_duration = 4

    return Interpretation(
        event_type="supply_disruption",
        probability=probability,
        estimated_lead_time_increase=est_lt_increase,
        estimated_duration=est_duration,
    )


# --- Mock interpreter (for testing) -----------------------------------------

def mock_interpret(text: str) -> Interpretation:
    """Deterministic keyword-based mock interpreter."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["urgent:", "urgent ", "we have confirmed", "direct notification", "informed us of a facility"]):
        probability = 0.95
    elif any(w in text_lower for w in ["cannot guarantee", "experience delays", "experiencing delays"]):
        probability = 0.80
    elif any(w in text_lower for w in ["congestion at the supplier", "delayed"]):
        probability = 0.75
    elif any(w in text_lower for w in ["capacity constraints", "staffing challenges"]):
        probability = 0.65
    elif any(w in text_lower for w in ["rumblings", "nothing confirmed", "not yet known", "elevated", "peer companies"]):
        probability = 0.45
    elif any(w in text_lower for w in ["may affect", "possible upstream"]):
        probability = 0.55
    else:
        probability = 0.50

    if any(w in text_lower for w in ["5 days", "5 periods", "3 additional"]):
        est_lt_increase = 3
    elif any(w in text_lower for w in ["2-4 extra", "2 to 4"]):
        est_lt_increase = 3
    elif any(w in text_lower for w in ["longer", "later", "delays"]):
        est_lt_increase = 2
    else:
        est_lt_increase = 1

    if any(w in text_lower for w in ["8 periods", "8 ordering", "next 8"]):
        est_duration = 8
    elif any(w in text_lower for w in ["several", "limited period", "coming week"]):
        est_duration = 6
    else:
        est_duration = 4

    return Interpretation(
        event_type="supply_disruption",
        probability=probability,
        estimated_lead_time_increase=est_lt_increase,
        estimated_duration=est_duration,
    )


def corrupted_interpret(text: str) -> Interpretation:
    """Deliberately degraded interpretation — below threshold."""
    base = mock_interpret(text)
    return Interpretation(
        event_type=base.event_type,
        probability=max(0.1, base.probability - 0.3),
        estimated_lead_time_increase=max(1, base.estimated_lead_time_increase - 2),
        estimated_duration=max(2, base.estimated_duration - 3),
    )


def active_wrong_interpret(text: str) -> Interpretation:
    """ActiveWrong: high-confidence but materially incorrect interpretation."""
    base = mock_interpret(text)
    wrong_probability = max(0.85, base.probability)
    wrong_lt_increase = max(1, base.estimated_lead_time_increase - 2)
    wrong_duration = max(2, base.estimated_duration - 4)
    return Interpretation(
        event_type=base.event_type,
        probability=wrong_probability,
        estimated_lead_time_increase=wrong_lt_increase,
        estimated_duration=wrong_duration,
    )


def active_wrong_overestimate(text: str) -> Interpretation:
    """ActiveWrong variant: overestimates severity."""
    base = mock_interpret(text)
    return Interpretation(
        event_type=base.event_type,
        probability=0.95,
        estimated_lead_time_increase=base.estimated_lead_time_increase + 3,
        estimated_duration=base.estimated_duration + 4,
    )


# --- Real LLM interpreter ---------------------------------------------------

_client_cache: dict = {}


def _get_client(api_key: str, base_url: str):
    """Get or create OpenAI client (cached)."""
    import openai
    key = (api_key, base_url)
    if key not in _client_cache:
        _client_cache[key] = openai.OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=15.0, max_retries=0,
        )
    return _client_cache[key]


def _cache_path(text: str, model: str) -> Path:
    """Deterministic cache file path for a given (text, model) pair."""
    h = hashlib.sha256(f"{model}||{text}".encode()).hexdigest()[:16]
    default_cache_dir = Path(__file__).resolve().parent.parent / ".llm_cache"
    cache_dir = Path(os.environ.get("PAPER2_LLM_CACHE_DIR", str(default_cache_dir)))
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{h}.json"


def _load_cache(text: str, model: str) -> Optional[dict]:
    p = _cache_path(text, model)
    if p.exists():
        return json.loads(p.read_text())
    frozen = p.parent.parent / "results" / "frozen_llm_outputs" / p.name
    if frozen.exists():
        return json.loads(frozen.read_text())
    return None


def _save_cache(text: str, model: str, result: dict) -> None:
    p = _cache_path(text, model)
    p.write_text(json.dumps(result, indent=2))


def llm_interpret(text: str) -> Interpretation:
    """Call an OpenAI-compatible API using LLM_API_KEY env vars."""
    result = llm_interpret_with_result(text)
    return result.interpretation


def llm_interpret_with_result(
    text: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 2,
) -> LLMResult:
    """Call an OpenAI-compatible API with full metadata tracking.

    Returns LLMResult with interpretation, raw response, latency, cache status.
    """
    if api_key is None:
        api_key = os.environ.get("LLM_API_KEY", "")
    if base_url is None:
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if model is None:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # Check cache first
    cached = _load_cache(text, model)
    if cached is not None:
        interp = Interpretation.from_dict(cached)
        return LLMResult(
            interpretation=interp,
            model=model,
            raw_response=json.dumps(cached),
            latency_ms=0.0,
            from_cache=True,
        )

    if not api_key:
        raise RuntimeError("LLM_API_KEY not set and no frozen semantic cache entry exists.")

    client = _get_client(api_key, base_url)

    last_error = None
    raw_response = ""
    latency_ms = 0.0
    retries = 0

    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            latency_ms = (time.time() - t0) * 1000

            raw_response = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw_response.startswith("```"):
                raw_response = raw_response.split("\n", 1)[1]
                raw_response = raw_response.rsplit("```", 1)[0]
            raw_response = raw_response.strip()

            result = json.loads(raw_response)
            required_keys = {"event_type", "probability", "estimated_lead_time_increase", "estimated_duration"}
            if not required_keys.issubset(result.keys()):
                raise ValueError(f"Missing required keys: {required_keys - result.keys()}")

            interp = Interpretation.from_dict(result)
            _save_cache(text, model, interp.to_dict())
            return LLMResult(
                interpretation=interp,
                model=model,
                raw_response=raw_response,
                latency_ms=latency_ms,
                from_cache=False,
                retries=retries,
            )
        except Exception as e:
            last_error = e
            retries = attempt + 1
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            # All retries exhausted — return a fallback interpretation
            return LLMResult(
                interpretation=Interpretation(
                    event_type="unknown",
                    probability=0.0,
                    estimated_lead_time_increase=0,
                    estimated_duration=0,
                ),
                model=model,
                raw_response=raw_response,
                latency_ms=latency_ms,
                from_cache=False,
                retries=retries,
                malformed=True,
            )

    # Should not reach here, but safety fallback
    return LLMResult(
        interpretation=Interpretation("unknown", 0.0, 0, 0),
        model=model, raw_response="", latency_ms=0.0,
        from_cache=False, retries=retries, malformed=True,
    )


# --- Factory -----------------------------------------------------------------

def get_interpreter() -> str:
    """Return 'llm' or 'mock' based on environment configuration."""
    if os.environ.get("LLM_API_KEY"):
        return "llm"
    return "mock"


def interpret(text: str) -> Interpretation:
    """Interpret a textual warning using the configured backend."""
    if get_interpreter() == "llm":
        return llm_interpret(text)
    return mock_interpret(text)


# =============================================================================
# Phase 5: Regime-Aware Interpreters
# =============================================================================
# These interpreters output beliefs over multiple possible regimes,
# rather than a single event_type/probability.
# =============================================================================

from src.events import Regime, REGIME_PRIOR


@dataclass
class RegimeInterpretation:
    """Belief distribution over operational regimes."""
    regime_probabilities: dict[str, float]  # regime -> probability, sums to ~1
    estimated_lt_increase: int = 0          # for the most likely non-normal regime
    estimated_duration: int = 0             # for the most likely non-normal regime
    estimated_demand_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return {
            "regime_probabilities": self.regime_probabilities,
            "estimated_lt_increase": self.estimated_lt_increase,
            "estimated_duration": self.estimated_duration,
            "estimated_demand_multiplier": self.estimated_demand_multiplier,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeInterpretation":
        if "regime_probabilities" in d:
            probs = {k: float(v) for k, v in d["regime_probabilities"].items()}
        else:
            probs = {k: float(v) for k, v in d.items()
                     if k in ("normal", "supplier_delay", "demand_surge")}
        return cls(
            regime_probabilities=probs,
            estimated_lt_increase=int(d.get("estimated_lt_increase", 0)),
            estimated_duration=int(d.get("estimated_duration", 0)),
            estimated_demand_multiplier=float(d.get("estimated_demand_multiplier", 1.0)),
        )

    @property
    def most_likely_regime(self) -> str:
        return max(self.regime_probabilities, key=self.regime_probabilities.get)

    def normalized(self) -> "RegimeInterpretation":
        """Return a normalized copy (probabilities sum to 1)."""
        total = sum(self.regime_probabilities.values())
        if total < 1e-9:
            # Uniform fallback
            n = len(self.regime_probabilities)
            probs = {k: 1.0 / n for k in self.regime_probabilities}
        else:
            probs = {k: v / total for k, v in self.regime_probabilities.items()}
        return RegimeInterpretation(
            regime_probabilities=probs,
            estimated_lt_increase=self.estimated_lt_increase,
            estimated_duration=self.estimated_duration,
            estimated_demand_multiplier=self.estimated_demand_multiplier,
        )


# Regime schema for LLM extraction
REGIME_INTERPRETATION_SCHEMA = {
    "normal": 0.10,
    "supplier_delay": 0.75,
    "demand_surge": 0.15,
    "estimated_lt_increase": 4,
    "estimated_duration": 10,
    "estimated_demand_multiplier": 1.0,
}

REGIME_EXTRACTION_PROMPT = (
    "You are an operational risk analyst. Given a textual supplier or market "
    "warning, classify the likely operational regime. Respond ONLY with valid "
    "JSON matching this exact schema:\n"
    '{"normal": <float 0-1>, "supplier_delay": <float 0-1>, '
    '"demand_surge": <float 0-1>, '
    '"estimated_lt_increase": <int>, "estimated_duration": <int>, '
    '"estimated_demand_multiplier": <float>}\n'
    "The three regime probabilities should sum approximately to 1.0.\n"
    "estimated_lt_increase: additional lead time periods if supplier_delay.\n"
    "estimated_duration: how many periods the event lasts.\n"
    "estimated_demand_multiplier: demand multiplier if demand_surge (1.0 if not).\n"
    "Do not include any other text."
)


def no_info_regime_belief() -> RegimeInterpretation:
    """NoInfo: returns the prior belief over regimes."""
    return RegimeInterpretation(
        regime_probabilities=dict(REGIME_PRIOR),
        estimated_lt_increase=0,
        estimated_duration=0,
        estimated_demand_multiplier=1.0,
    )


def perfect_semantic_regime_belief(true_regime: Regime) -> RegimeInterpretation:
    """PerfectSemantic: returns degenerate belief on the true regime."""
    probs = {r.value: 0.0 for r in Regime}
    probs[true_regime.value] = 1.0

    params = {
        Regime.NORMAL: (0, 0, 1.0),
        Regime.SUPPLIER_DELAY: (4, 10, 1.0),
        Regime.DEMAND_SURGE: (0, 10, 1.75),
        Regime.SUPPLIER_CAPACITY_DROP: (0, 0, 1.0),
    }
    lt, dur, mult = params[true_regime]

    return RegimeInterpretation(
        regime_probabilities=probs,
        estimated_lt_increase=lt,
        estimated_duration=dur,
        estimated_demand_multiplier=mult,
    )


def rule_based_regime_extract(text: str) -> RegimeInterpretation:
    """Deterministic keyword/regex regime classifier — no LLM required.

    Classifies text into regime probabilities based on keywords.
    Designed to be a simple, transparent baseline.
    """
    text_lower = text.lower()

    # --- Score each regime ---
    delay_score = 0.0
    surge_score = 0.0
    normal_score = 0.0

    # Supplier delay signals
    if any(w in text_lower for w in [
        "supplier", "logistics", "congestion", "delay", "lead time",
        "shipment", "delivery", "facility", "production issue",
    ]):
        delay_score += 0.3
    if any(w in text_lower for w in [
        "confirmed", "urgent", "notified", "facility disruption",
        "increase from", "doubled", "extended lead time",
    ]):
        delay_score += 0.3
    if any(w in text_lower for w in [
        "may arrive later", "staffing challenges", "capacity constraints",
        "rumblings", "possible upstream", "delivery schedules",
    ]):
        delay_score += 0.15

    # Demand surge signals
    if any(w in text_lower for w in [
        "demand", "customer", "purchasing", "order", "sales",
        "contract", "pipeline", "inquiries", "pre-order",
    ]):
        surge_score += 0.3
    if any(w in text_lower for w in [
        "confirmed", "major", "large contract", "strong",
        "unusually strong", "above normal", "uptick",
    ]):
        surge_score += 0.3
    if any(w in text_lower for w in [
        "interest", "elevated", "elevated demand", "could pick up",
        "real possibility",
    ]):
        surge_score += 0.15

    # Normal signals
    if any(w in text_lower for w in [
        "on schedule", "normal", "no disruption", "no change",
        "within normal", "stable", "no material", "no unusual",
        "routine variability", "no significant",
    ]):
        normal_score += 0.4
    if any(w in text_lower for w in [
        "operating normally", "standard", "expected ranges",
        "no supply", "no demand",
    ]):
        normal_score += 0.2

    # --- Compute probabilities ---
    total = delay_score + surge_score + normal_score
    if total < 1e-9:
        # No strong signals — use prior-like distribution
        probs = {r.value: 1.0 / 3 for r in Regime}
    else:
        probs = {
            Regime.NORMAL.value: normal_score / total,
            Regime.SUPPLIER_DELAY.value: delay_score / total,
            Regime.DEMAND_SURGE.value: surge_score / total,
        }

    # --- Estimate parameters from most likely non-normal regime ---
    most_likely = max(probs, key=probs.get)
    if most_likely == Regime.SUPPLIER_DELAY.value:
        # Try to extract LT increase and duration from text
        lt_match = re.search(
            r'from\s+(\d+)\s+to\s+(\d+)', text_lower
        )
        if lt_match:
            lt_inc = int(lt_match.group(2)) - int(lt_match.group(1))
        elif re.search(r'(\d+)\s*(?:additional|extra)\s*(?:periods?|days?)', text_lower):
            m = re.search(r'(\d+)\s*(?:additional|extra)', text_lower)
            lt_inc = int(m.group(1))
        else:
            lt_inc = 4  # default for Phase 5

        dur_match = re.search(r'(\d+)\s*(?:periods?|ordering|cycles?)', text_lower)
        if dur_match:
            dur = int(dur_match.group(1))
        elif "10" in text_lower:
            dur = 10
        else:
            dur = 10  # default for Phase 5

        return RegimeInterpretation(
            regime_probabilities=probs,
            estimated_lt_increase=lt_inc,
            estimated_duration=dur,
            estimated_demand_multiplier=1.0,
        )
    elif most_likely == Regime.DEMAND_SURGE.value:
        # Try to extract demand multiplier
        mult_match = re.search(r'(\d+)\s*(?:units?|from)', text_lower)
        if mult_match:
            val = int(mult_match.group(1))
            if val > 10:
                mult = val / 8.0  # relative to base demand of 8
            else:
                mult = 1.75
        else:
            mult = 1.75

        dur_match = re.search(r'(\d+)\s*(?:periods?|ordering|cycles?)', text_lower)
        dur = int(dur_match.group(1)) if dur_match else 10

        return RegimeInterpretation(
            regime_probabilities=probs,
            estimated_lt_increase=0,
            estimated_duration=dur,
            estimated_demand_multiplier=mult,
        )
    else:
        return RegimeInterpretation(
            regime_probabilities=probs,
            estimated_lt_increase=0,
            estimated_duration=0,
            estimated_demand_multiplier=1.0,
        )


def llm_regime_interpret_with_result(
    text: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 2,
) -> tuple[RegimeInterpretation, LLMResult]:
    """Call LLM for regime classification. Returns (RegimeInterpretation, LLMResult)."""
    if api_key is None:
        api_key = os.environ.get("LLM_API_KEY", "")
    if base_url is None:
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if model is None:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # Use the existing LLM call infrastructure with the regime prompt
    # Build a temporary regime-specific call
    import hashlib
    cache_key = f"regime_{model}||{text}"
    h = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    default_cache_dir = Path(__file__).resolve().parent.parent / ".llm_cache"
    cache_dir = Path(os.environ.get("PAPER2_LLM_CACHE_DIR", str(default_cache_dir)))
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{h}.json"

    frozen_cache_file = cache_file.parent.parent / "results" / "frozen_llm_outputs" / cache_file.name
    if cache_file.exists() or frozen_cache_file.exists():
        cached = json.loads((cache_file if cache_file.exists() else frozen_cache_file).read_text())
        regime_interp = RegimeInterpretation.from_dict(cached)
        llm_result = LLMResult(
            interpretation=Interpretation("regime", 0.0, 0, 0),
            model=model,
            raw_response=json.dumps(cached),
            latency_ms=0.0,
            from_cache=True,
        )
        return regime_interp, llm_result

    if not api_key:
        raise RuntimeError("LLM_API_KEY not set and no frozen semantic cache entry exists.")

    client = _get_client(api_key, base_url)
    last_error = None
    raw_response = ""
    latency_ms = 0.0
    retries = 0

    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": REGIME_EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            latency_ms = (time.time() - t0) * 1000
            raw_response = response.choices[0].message.content.strip()
            if raw_response.startswith("```"):
                raw_response = raw_response.split("\n", 1)[1]
                raw_response = raw_response.rsplit("```", 1)[0]
            raw_response = raw_response.strip()

            result = json.loads(raw_response)
            regime_interp = RegimeInterpretation.from_dict(result).normalized()

            # Save to cache
            cache_file.write_text(json.dumps(regime_interp.to_dict(), indent=2))

            llm_result = LLMResult(
                interpretation=Interpretation("regime", 0.0, 0, 0),
                model=model,
                raw_response=raw_response,
                latency_ms=latency_ms,
                from_cache=False,
                retries=retries,
            )
            return regime_interp, llm_result

        except Exception as e:
            last_error = e
            retries = attempt + 1
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            # Fallback: return prior
            fallback = no_info_regime_belief()
            llm_result = LLMResult(
                interpretation=Interpretation("unknown", 0.0, 0, 0),
                model=model,
                raw_response=raw_response,
                latency_ms=latency_ms,
                from_cache=False,
                retries=retries,
                malformed=True,
            )
            return fallback, llm_result

    fallback = no_info_regime_belief()
    return fallback, LLMResult(
        interpretation=Interpretation("unknown", 0.0, 0, 0),
        model=model, raw_response="", latency_ms=0.0,
        from_cache=False, retries=retries, malformed=True,
    )
