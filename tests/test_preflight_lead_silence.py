"""preflight.py's --lead-silence-s: ngspice-deck keeps its 3 s default, livespice defaults to none (existing behaviour) and only
prepends a settling lead-in when asked (for an AC-front-end amp that needs >= 6 s), other backends never do."""
import pytest

from preflight import resolve_lead_silence


@pytest.mark.parametrize("backend,requested,expected", [
    ("ngspice-deck", None, 3.0),
    ("ngspice-deck", 0.0, 0.0),
    ("ngspice-deck", 5.0, 5.0),
    ("livespice", None, 0.0),
    ("livespice", 0.0, 0.0),
    ("livespice", 6.0, 6.0),
    ("ngspice", 6.0, 0.0),
    ("ltspice-deck", 6.0, 0.0),
])
def test_resolve_lead_silence(backend, requested, expected):
    assert resolve_lead_silence(backend, requested) == expected
