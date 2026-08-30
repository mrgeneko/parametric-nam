"""Properties knob_classify.classify() must have -- it decides which real fleet knobs
get a direction check, which get preflight.py's --eq-check-drive-level EQ-swamp protection,
and which metric gen_dataset_from_schx.py's knob-sensitivity check uses. A wrong or missed
classification is silent everywhere it matters: an unrecognized name gets the LEAST scrutiny,
not the most (see classify()'s own docstring).

See knob_classify.py, preflight.py, scaffold_config.py.
"""
from knob_classify import DIRECTION_RULES, TONE_LIKE_KINDS, classify


class TestRealFleetNames:
    """Names actually used on real, shipped devices this session touched or referenced
    (this fleet's own pedals and amps) -- a change to DIRECTION_RULES that breaks
    any of these silently degrades preflight.py's checks on a real device, not a hypothetical
    one."""

    def test_tone_shaping_knobs(self):
        for name in ("Tone", "Treble", "Bass", "Middle", "Presence", "Bright"):
            kind, _ = classify(name)
            assert kind in ("hi", "lo", "mid"), f"{name} should classify as tone-like, got {kind}"

    def test_treble_specifically_hi(self):
        assert classify("Treble")[0] == "hi"
        assert classify("Tone")[0] == "hi"
        assert classify("Presence")[0] == "hi"

    def test_bass_specifically_lo(self):
        assert classify("Bass")[0] == "lo"

    def test_middle_specifically_mid(self):
        assert classify("Middle")[0] == "mid"
        assert classify("Mid")[0] == "mid"

    def test_drive_knobs(self):
        for name in ("Gain", "Drive", "Distortion", "Sustain", "Overdrive", "Fuzz"):
            assert classify(name)[0] == "drive", f"{name} should classify as drive"

    def test_level_knobs(self):
        for name in ("Volume", "Level", "Master", "Output"):
            assert classify(name)[0] == "rms", f"{name} should classify as rms"

    def test_case_insensitive(self):
        assert classify("GAIN")[0] == classify("gain")[0] == classify("Gain")[0] == "drive"


class TestUnknown:
    """A name matching no keyword returns (None, 'unknown') -- callers must handle this,
    not assume every knob classifies to something."""

    def test_unrecognized_name_returns_none(self):
        assert classify("Zorp") == (None, "unknown")

    def test_empty_string(self):
        assert classify("")[0] is None


class TestKnownFalsePositives:
    """classify() is a plain substring match on keywords, not a word-boundary or NLP match --
    it inherited this from preflight.py's original heuristic, not something introduced this
    session. Documenting the actual failure mode (not just asserting it "works") so a future
    change to DIRECTION_RULES doesn't accidentally make this worse without anyone noticing,
    and so anyone relying on classify() knows its real limits. 'od' (a drive/overdrive
    keyword) matches inside unrelated words containing that substring."""

    def test_od_substring_false_positive(self):
        # "od" (meant to catch "OD"/"Overdrive") also matches "mOD"/"wOOD" — a real,
        # currently-unfixed false positive, not a hypothetical one.
        assert classify("Modulation") == ("drive", "gain/distortion")
        assert classify("Wood") == ("drive", "gain/distortion")


class TestToneLikeKinds:
    def test_exact_set(self):
        assert set(TONE_LIKE_KINDS) == {"hi", "lo", "mid"}

    def test_drive_and_rms_excluded(self):
        assert "drive" not in TONE_LIKE_KINDS
        assert "rms" not in TONE_LIKE_KINDS


class TestDirectionRulesShape:
    """Structural sanity on the rule table itself -- catches a malformed entry (wrong tuple
    arity, empty keyword list) that would otherwise only surface as a confusing classify()
    result far from the actual mistake."""

    def test_every_rule_is_keywords_kind_label(self):
        for rule in DIRECTION_RULES:
            assert len(rule) == 3
            keywords, kind, label = rule
            assert isinstance(keywords, tuple) and len(keywords) > 0
            assert isinstance(kind, str)
            assert isinstance(label, str)

    def test_kinds_cover_exactly_the_documented_five(self):
        kinds = {rule[1] for rule in DIRECTION_RULES}
        assert kinds == {"hi", "lo", "mid", "drive", "rms"}
