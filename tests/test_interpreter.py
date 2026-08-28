"""Tests for the LLM interpreter."""

from src.interpreter import (
    mock_interpret,
    corrupted_interpret,
    Interpretation,
    INTERPRETATION_SCHEMA,
)
from src.events import WARNING_TEMPLATES


def test_interpreter_output_schema():
    """Interpreter output follows the required schema."""
    for t in WARNING_TEMPLATES:
        interp = mock_interpret(t["text"])
        assert isinstance(interp, Interpretation)
        d = interp.to_dict()
        assert set(d.keys()) == set(INTERPRETATION_SCHEMA.keys())


def test_interpreter_probability_range():
    """Probability is between 0 and 1."""
    for t in WARNING_TEMPLATES:
        interp = mock_interpret(t["text"])
        assert 0.0 <= interp.probability <= 1.0


def test_interpreter_positive_estimates():
    """Lead time increase and duration are positive integers."""
    for t in WARNING_TEMPLATES:
        interp = mock_interpret(t["text"])
        assert interp.estimated_lead_time_increase >= 1
        assert interp.estimated_duration >= 1


def test_interpreter_event_type():
    """Event type is always supply_disruption for supplier warnings."""
    for t in WARNING_TEMPLATES:
        interp = mock_interpret(t["text"])
        assert interp.event_type == "supply_disruption"


def test_interpreter_clear_vs_vague():
    """Clear templates should generally yield higher probability than vague."""
    clear_probs = [mock_interpret(t["text"]).probability for t in WARNING_TEMPLATES if t["clarity"] == "clear"]
    vague_probs = [mock_interpret(t["text"]).probability for t in WARNING_TEMPLATES if t["clarity"] == "vague"]
    assert max(clear_probs) >= max(vague_probs), "Clear warnings should have at least as high probability as vague"


def test_interpreter_deterministic():
    """Mock interpreter is deterministic for same input."""
    text = WARNING_TEMPLATES[0]["text"]
    a = mock_interpret(text)
    b = mock_interpret(text)
    assert a.to_dict() == b.to_dict()


def test_corrupted_interpreter_reduces_values():
    """Corrupted interpreter produces lower probability and estimates."""
    for t in WARNING_TEMPLATES:
        base = mock_interpret(t["text"])
        corr = corrupted_interpret(t["text"])
        assert corr.probability <= base.probability + 0.01, "Corrupted should not increase probability"
        assert corr.estimated_lead_time_increase <= base.estimated_lead_time_increase
        assert corr.estimated_duration <= base.estimated_duration
