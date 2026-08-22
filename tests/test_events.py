"""Tests for event definitions."""

from src.events import SupplierDisruption, DEFAULT_DISRUPTION, WARNING_TEMPLATES


def test_disruption_end_time():
    d = SupplierDisruption(start_time=18, duration=8, normal_lead_time=2, disrupted_lead_time=5)
    assert d.end_time == 26


def test_disruption_is_active():
    d = SupplierDisruption(start_time=18, duration=8, normal_lead_time=2, disrupted_lead_time=5)
    assert not d.is_active(17)
    assert d.is_active(18)
    assert d.is_active(25)
    assert not d.is_active(26)


def test_disruption_lead_time():
    d = SupplierDisruption(start_time=18, duration=8, normal_lead_time=2, disrupted_lead_time=5)
    assert d.current_lead_time(10) == 2
    assert d.current_lead_time(18) == 5
    assert d.current_lead_time(25) == 5
    assert d.current_lead_time(26) == 2


def test_default_disruption_parameters():
    assert DEFAULT_DISRUPTION.start_time == 18
    assert DEFAULT_DISRUPTION.duration == 8
    assert DEFAULT_DISRUPTION.normal_lead_time == 2
    assert DEFAULT_DISRUPTION.disrupted_lead_time == 5


def test_warning_templates_count():
    assert len(WARNING_TEMPLATES) >= 10


def test_warning_templates_have_required_keys():
    for t in WARNING_TEMPLATES:
        assert "id" in t
        assert "clarity" in t
        assert "text" in t
        assert t["clarity"] in ("clear", "moderate", "vague")


def test_warning_templates_clarity_spread():
    clarities = [t["clarity"] for t in WARNING_TEMPLATES]
    assert "clear" in clarities
    assert "moderate" in clarities
    assert "vague" in clarities
