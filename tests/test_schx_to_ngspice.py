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


# --------------------------------------------------------------------------- op-amp model
# The default macromodel hard-limits the output 0.5 V inside each rail; LiveSPICE's own OpAmp
# clamps its gain node with two very sharp diodes ~2 V inside, so the swing ends ~1.0-1.3 V
# inside. A cross-engine comparison of a circuit that drives an op-amp into its rails compares
# two different op-amps unless opamp_model=livespice is set (Bluesbreaker, 2026-09-25).

def _opamp_netlist():
    def comp(name, typ, terms, **params):
        return {"name": name, "type": typ, "value": None, "isPot": False, "params": params,
                "terminals": [{"name": k, "node": v} for k, v in terms.items()]}
    return {"components": [
        comp("J_IN", "Input", {"Anode": "IN", "Cathode": "GND"}, V0dBFS="1"),
        comp("V9", "Rail", {"V9": "V9"}, Voltage="9"),
        comp("IC1", "OpAmp", {"+": "IN", "-": "OUT", "Out": "OUT", "Vcc+": "V9", "Vcc-": "GND"},
             Aol="200000", GBP="3000000", Rout="100", Rin="1000000000"),
        comp("S_OUT", "Speaker", {"Anode": "OUT", "Cathode": "GND"}, V0dBFS="1")]}


class TestOpampModel:
    def test_default_is_the_hard_limit_and_unchanged(self):
        sub = "\n".join(X.opamp_subckt("OA0", 2e5, 3e6, 100.0, True))
        assert "min(max(V(g),V(vn)+0.5),V(vp)-0.5)" in sub
        assert "DCL" not in sub

    def test_livespice_model_is_the_two_sharp_clamp_diodes(self):
        sub = "\n".join(X.opamp_subckt("OA0", 2e5, 3e6, 100.0, True, model="livespice"))
        assert ".model DCL_OA0 D(IS=8e-16 N=1)" in sub
        assert "Vch vp nch DC 2" in sub and "Dch g nch DCL_OA0" in sub     # gain node <= Vcc-2 (+Vd)
        assert "Vcl ncl vn DC 2" in sub and "Dcl ncl g DCL_OA0" in sub     # gain node >= Vee+2 (-Vd)
        assert "min(" not in sub and "Bo " not in sub                      # no hard limiter
        assert "Ro g out" in sub                                           # buffered through Rout

    def test_livespice_model_uses_livespices_own_gain_stage_constants(self):
        sub = "\n".join(X.opamp_subckt("OA0", 2e5, 3e6, 100.0, True, model="livespice"))
        assert "Ra g 0 1000.0" in sub                                         # OpAmp.cs Rp1 = 1000
        assert "Ga 0 g inp inn 200.0" in sub                                # Aol / Rp1 = 200 S

    def test_livespice_model_without_supply_pins_falls_back_like_livespice_does(self):
        """LiveSPICE only clamps when Vcc+/Vcc- are connected; with no rails there is no clamp
        to copy, so the default (unclamped) macromodel is used either way."""
        a = X.opamp_subckt("OA0", 2e5, 3e6, 100.0, False)
        b = X.opamp_subckt("OA0", 2e5, 3e6, 100.0, False, model="livespice")
        assert a == b

    def test_conv_key_reaches_the_translated_deck(self):
        deck = X.translate(_opamp_netlist(), conv={"opamp_model": "livespice"})
        assert "DCL_OA0" in deck and "min(max(" not in deck

    def test_without_the_conv_key_the_deck_is_byte_identical_to_before(self):
        base = X.translate(_opamp_netlist())
        assert base == X.translate(_opamp_netlist(), conv={})
        assert "min(max(V(g),V(vn)+0.5),V(vp)-0.5)" in base and "DCL" not in base

    def test_unrelated_conv_keys_do_not_change_the_opamp(self):
        assert X.translate(_opamp_netlist(), conv={"diode_cjo": "100p"}) == X.translate(_opamp_netlist())
