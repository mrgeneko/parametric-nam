"""Properties schx_to_ngspice.py's bjt_model() must have.

The .schx BipolarJunctionTransistor element (and LiveSPICE's own C# class behind it) only
carries Type/IS/BF/BR -- there is nowhere in the file format to record a real datasheet/PSpice
fit's secondary Gummel-Poon parameters (VAF, IKF, RB, real junction caps, etc), even though
ngspice's own model supports all of them. Added 2026-09-12 for the Arbiter Fuzz Face's AC128
germanium transistors, whose ElectroSmash-sourced PSpice fit (VAF=102.207, IKF=9.981m,
RB=173.312, CJE=6p/CJC=3.75p not the generic 8p/4p defaults, etc) was previously unreachable.

See schx_to_ngspice.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ngspice"))
import schx_to_ngspice as X  # noqa: E402


BJT_P = {"Type": "PNP", "IS": "85.8 nA", "BF": "85", "BR": "20"}


class TestBjtModelBackwardCompatibility:
    """No --conv, no new schx attributes: output must be byte-identical to before the
    optional-parameter extension existed -- this is what every OTHER device's ngspice render
    (which never sets any of the new bjt_* conv keys) depends on."""

    def test_no_conv_produces_the_original_six_parameters_only(self):
        out = X.bjt_model(BJT_P, {})
        assert out == "PNP(IS=8.58e-08 BF=85.0 BR=20.0 CJE=8e-12 CJC=4e-12 TF=5e-10)"

    def test_unrelated_conv_keys_do_not_add_any_optional_parameter(self):
        out = X.bjt_model(BJT_P, {"diode_cjo": "100p", "tmax": "5u"})
        assert out == "PNP(IS=8.58e-08 BF=85.0 BR=20.0 CJE=8e-12 CJC=4e-12 TF=5e-10)"

    def test_npn_type_still_selects_npn(self):
        out = X.bjt_model({"Type": "NPN", "IS": "1e-12", "BF": "200", "BR": "1"}, {})
        assert out.startswith("NPN(")


class TestBjtModelOptionalParameters:
    """VAF/IKF/RB/etc are appended ONLY when a value is actually available (conv override,
    schx attribute takes priority if ever present), and omitted entirely otherwise -- an
    omitted parameter lets ngspice's own native default apply, rather than this module
    guessing one."""

    def test_conv_override_appends_the_parameter(self):
        out = X.bjt_model(BJT_P, {"bjt_vaf": "102.207"})
        assert "VAF=102.207" in out

    def test_unset_optional_parameters_are_never_emitted(self):
        out = X.bjt_model(BJT_P, {"bjt_vaf": "102.207"})
        for key in ("IKF", "RB", "ISE", "TR", "XCJC"):
            assert f"{key}=" not in out

    def test_conv_can_override_cje_cjc_with_real_fitted_values(self):
        """CJE/CJC always have a value (generic 8p/4p convergence default) -- conv must be
        able to replace that default with a real datasheet figure, not just add to it."""
        out = X.bjt_model(BJT_P, {"bjt_cje": "6e-12", "bjt_cjc": "3.75e-12"})
        assert "CJE=6e-12" in out
        assert "CJC=3.75e-12" in out

    def test_schx_attribute_takes_priority_over_conv_override(self):
        """Matches the existing CJE/CJC/TF precedence (_cv: schx attr > conv > default) --
        a real per-instance schx value, if the format ever grows one, must not be silently
        overridden by a blanket --conv applying to every BJT in the netlist."""
        p = dict(BJT_P, VAF="55")
        out = X.bjt_model(p, {"bjt_vaf": "102.207"})
        assert "VAF=55.0" in out
        assert "VAF=102.207" not in out

    def test_full_ac128_fit_all_present(self):
        """The actual Arbiter Fuzz Face AC128 conv override (Q1/Q2 share every secondary
        parameter -- only IS/BF differ, and those come from the schx per-instance)."""
        conv = {
            "bjt_vaf": "102.207", "bjt_ikf": "0.009981", "bjt_ise": "4.35e-10",
            "bjt_ne": "1.2", "bjt_var": "20", "bjt_ikr": "0.001248",
            "bjt_isc": "1.208e-7", "bjt_nc": "1.2", "bjt_rb": "173.312",
            "bjt_irb": "5e-6", "bjt_rbm": "43.328", "bjt_re": "20", "bjt_rc": "60",
            "bjt_cje": "6e-12", "bjt_vje": "0.4", "bjt_mje": "0.4", "bjt_tf": "1.5e-7",
            "bjt_xtf": "9.996", "bjt_vtf": "2", "bjt_itf": "0.009983", "bjt_ptf": "1",
            "bjt_cjc": "3.75e-12", "bjt_vjc": "0.6", "bjt_mjc": "0.33",
            "bjt_xcjc": "0.65", "bjt_tr": "2.865e-6", "bjt_fc": "0.75",
        }
        out = X.bjt_model(BJT_P, conv)
        for pspice_key, ckey in X._BJT_OPTIONAL:
            assert f"{pspice_key}=" in out, f"missing {pspice_key} from {out}"
        # Q1's own IS/BF/BR (from the schx, not conv) must still come through unmangled.
        assert out.startswith("PNP(IS=8.58e-08 BF=85.0 BR=20.0")

    def test_uppercase_m_suffix_is_mega_not_milli_pspice_convention(self):
        """The exact unit trap this module's own qty() docstring warns about: PSpice writes
        milli as 'M' (e.g. IKF=9.981M means 9.981 mA), but THIS codebase's qty() treats
        uppercase M as MEGA and lowercase m as milli. A --conv value copied verbatim from a
        PSpice .MODEL card using PSpice's own 'M' convention would be silently wrong by 1e9 --
        conv values must already be converted to this codebase's convention (or written as
        plain scientific notation) before being passed in."""
        wrong = X.bjt_model(BJT_P, {"bjt_ikf": "9.981M"})   # PSpice-style, NOT converted
        right = X.bjt_model(BJT_P, {"bjt_ikf": "9.981m"})   # this codebase's milli
        assert "IKF=9981000.0" in wrong    # 9.981 MEGA -- the trap, if triggered
        assert "IKF=0.009981" in right     # 9.981 milli -- the intended value
