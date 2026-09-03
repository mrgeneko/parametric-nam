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


class TestChannelPrefixVsRoleSuffix:
    """A knob name ending in a level/gain word is a much stronger signal than an incidental
    substring match elsewhere in the name -- confirmed on a real device: BrightVol on a
    dual-channel (Bright/Normal) tweed amp was classifying "hi" (treble) purely because
    "bright" is a treble/presence keyword, when "Bright" here names the CHANNEL and BrightVol
    is that channel's volume control (same role as NormalVol). Any amp/pedal with a named
    channel or mode prefix (Bright, Normal, OD, Red, Or/Orange, ...) can hit this same
    collision; these are regression guards for the fix, not hypothetical cases."""

    def test_brightvol_is_rms_not_hi(self):
        # The bug this fix exists for: was ("hi", "treble/high-freq"), wrong role AND wrong
        # metric (checked "does treble content rise" instead of "does output level rise").
        assert classify("BrightVol") == ("rms", "output level")

    def test_bright_alone_is_still_hi(self):
        # The channel-name word alone (no role suffix) is still correctly a tone/presence
        # knob on amps where "Bright" really is a brightness control, not a channel name --
        # this fix must not blanket-exclude "bright" from the hi classification.
        assert classify("Bright") == ("hi", "treble/high-freq")

    def test_od_level_is_rms_not_drive(self):
        # Same root cause, found auditing the real fleet's other knob names while fixing
        # BrightVol: "OD" is a channel prefix (Overdrive channel) on several amps, and
        # "OD Level"/"ODLevel" is that channel's level control, not a gain/drive knob --
        # was misclassified "drive" via the "od" substring.
        assert classify("OD Level") == ("rms", "output level")
        assert classify("ODLevel") == ("rms", "output level")

    def test_od_gain_is_still_drive(self):
        # Same channel prefix, but the ACTUAL role word ("Gain") is a drive suffix -- must
        # still classify correctly; this fix is about role-suffix priority, not about
        # excluding "OD"-prefixed names from "drive" generally.
        assert classify("OD Gain") == ("drive", "gain/distortion")

    def test_normalvol_still_rms(self):
        # Was already correct before this fix (no tone-keyword collision in "normal") --
        # confirms the fix doesn't regress the sibling knob it's most directly compared against.
        assert classify("NormalVol") == ("rms", "output level")

    def test_suffix_priority_beats_earlier_substring_in_rule_order(self):
        # Direct test of the mechanism: "hi" is checked before "drive"/"rms" in
        # DIRECTION_RULES's plain substring order, so a naive first-match-wins substring scan
        # would classify this "hi" (contains "treble") despite ending in a level suffix.
        assert classify("TrebleLevel") == ("rms", "output level")


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
