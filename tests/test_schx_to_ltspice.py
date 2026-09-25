"""Properties schx_to_ltspice.py must have.

schx_to_ltspice.py is the LTspice counterpart of ngspice/schx_to_ngspice.py -- see its module
docstring for why it exists (the Ampeg SVT Full render-speed investigation) and its reuse
strategy (dialect-independent parsing/translation math imported directly from
schx_to_ngspice.py; only the genuinely LTspice-specific pieces -- ternary->if() conditionals,
wavefile=/.wave I/O, the .tran/.options/.save deck skeleton -- get new code).

These tests focus on the parts that are NEW here, not re-testing schx_to_ngspice.py's own
math (already covered by test_schx_to_ngspice.py) -- reused functions are exercised only
enough to confirm the import/reuse wiring itself works.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ltspice"))
import schx_to_ltspice as L  # noqa: E402

KOREN_TRIODE_P = {"Mu": "100", "Ex": "1.4", "Kg": "1060", "Kp": "600", "Kvb": "300"}
PENTODE_P = {"Mu": "7.9", "Ex": "1.35", "Kg1": "570", "Kg2": "4200", "Kp": "60", "Kvb": "24"}


class TestTernaryToIfConversion:
    """ngspice's `(cond)?a:b` ternary is not documented LTspice B-source syntax (LTspice's own
    B-device reference lists `if(x,y,z)` -- 'y if x>0.5, else z' -- with no ternary operator
    mentioned) -- see module docstring. Every conditional emitted by this module's own
    Koren-triode/pentode subckts must use if(), never a bare '?'."""

    def test_koren_triode_uses_if_not_ternary(self):
        lines = L.triode_subckt_koren_ltspice("TRI0", KOREN_TRIODE_P)
        text = "\n".join(lines)
        assert "?" not in text, f"ternary operator leaked into LTspice output: {text}"
        assert "if(" in text

    def test_pentode_uses_if_not_ternary(self):
        lines = L.pentode_subckt_ltspice("PEN0", PENTODE_P)
        text = "\n".join(lines)
        assert "?" not in text, f"ternary operator leaked into LTspice output: {text}"
        assert text.count("if(") >= 2  # plate current AND the shared ikoren both gate on E1>0

    def test_koren_triode_bsource_syntax_is_unspaced(self):
        """LTspice's own B-device reference always shows I=/V= unspaced -- never seen it
        reject the ngspice-style spaced form in practice, but there's no reason to rely on
        it (see _bsrc_normalize's docstring); the LOCAL (not reused) Koren/pentode builders
        must emit the unspaced form directly rather than needing normalization after the fact."""
        text = "\n".join(L.triode_subckt_koren_ltspice("TRI0", KOREN_TRIODE_P))
        assert "I=" in text
        assert " I = " not in text

    def test_pentode_ports_and_structure(self):
        lines = L.pentode_subckt_ltspice("PEN0", PENTODE_P)
        assert lines[0] == ".subckt PEN0 P G2 G K"
        assert lines[-1] == ".ends"
        assert any(ln.startswith("Bp P K I=") for ln in lines)
        assert any(ln.startswith("Bg2 G2 K I=") for ln in lines)

    def test_koren_triode_still_has_grid_diode_and_interelectrode_caps(self):
        """Matches schx_to_ngspice.py's own triode_subckt_koren -- the DGRID diode + fixed
        2.4p/2.3p Cgp/Cgk that make the Koren convergence-mode triode numerically soft."""
        lines = L.triode_subckt_koren_ltspice("TRI0", KOREN_TRIODE_P)
        text = "\n".join(lines)
        assert "Dgk G K DGRID" in text
        assert "CGP G P 2.4p" in text
        assert "CGK G K 2.3p" in text


class TestBsourceNormalize:
    """_bsrc_normalize is what lets this module reuse schx_to_ngspice.py's DempwolfZolzer
    triode/op-amp B-source lines VERBATIM (see translate()'s tube_sub/opamp_sub closures) --
    it must strip the ' = ' spacing ngspice tolerates down to LTspice's documented I=/V=."""

    def test_strips_space_around_i_equals(self):
        out = L._bsrc_normalize(["Bg G K I = 1+2"])
        assert out == ["Bg G K I=1+2"]

    def test_strips_space_around_v_equals(self):
        out = L._bsrc_normalize(["Bo ob 0 V = min(max(1,2),3)"])
        assert out == ["Bo ob 0 V=min(max(1,2),3)"]

    def test_leaves_unrelated_lines_untouched(self):
        lines = [".subckt X P G K", "Dgk G K DGRID", ".ends"]
        assert L._bsrc_normalize(lines) == lines

    def test_reused_dempwolfzolzer_triode_has_no_spaced_assignment_left(self):
        """End-to-end: schx_to_ngspice.triode_subckt's own literal output (' I = ') must come
        out normalized once routed through this module's tube_sub closure."""
        import schx_to_ngspice as NG
        p = {"Mu": "100", "G": "2e-3", "Gg": "0.6e-3", "C": "3.4", "Cg": "9.9",
             "Xi": "1.3", "Gamma": "1.26", "Ig0": "8e-8"}
        raw = NG.triode_subckt("TRI0", p)
        assert any(" I = " in ln for ln in raw), "fixture assumption changed upstream"
        normalized = L._bsrc_normalize(raw)
        assert not any(" I = " in ln for ln in normalized)
        assert any("I=" in ln for ln in normalized)


def _tiny_netlist(extra_components=None):
    """Minimal but complete netlist: Rail -> Resistor -> Input -> ground, Speaker probing the
    resistor's far end -- enough for translate() to produce a loadable deck skeleton without
    needing a real .schx dump."""
    comps = [
        {"name": "BP", "type": "Rail", "value": None, "isPot": False,
         "params": {"Voltage": "300 V"},
         "terminals": [{"name": "Anode", "node": "BP"}]},
        {"name": "J_IN", "type": "Input", "value": None, "isPot": False,
         "params": {"V0dBFS": "1"},
         "terminals": [{"name": "Anode", "node": "IN"}, {"name": "Cathode", "node": "GND"}]},
        {"name": "R1", "type": "Resistor", "value": "100 kΩ", "isPot": False, "params": {},
         "terminals": [{"name": "A", "node": "IN"}, {"name": "B", "node": "OUT"}]},
        {"name": "S_OUT", "type": "Speaker", "value": None, "isPot": False, "params": {},
         "terminals": [{"name": "Anode", "node": "OUT"}, {"name": "Cathode", "node": "GND"}]},
        {"name": "GND", "type": "Ground", "value": None, "isPot": False, "params": {},
         "terminals": [{"name": "Anode", "node": "GND"}]},
    ]
    if extra_components:
        comps += extra_components
    return {"components": comps}


class TestTranslateDeckSkeleton:
    """translate() must produce a self-contained LTspice deck: no separate .control block to
    append to afterward (LTspice has none -- see module docstring), so .tran/.options/.save/
    .wave must all be present in the ONE returned string."""

    def test_produces_required_dot_commands(self):
        out = L.translate(_tiny_netlist(), wav_path="in.wav", dur=0.5, out_wav="out.wav")
        assert ".tran " in out
        assert ".options " in out
        assert ".save V(spkout)" in out
        assert '.wave "' in out
        assert out.rstrip().endswith(".end")

    def test_input_restores_real_volts_via_e_source(self):
        """wav_path's own samples are real-input-volts/in_scale (PCM-safety convention) --
        translate() must multiply back by in_scale (times the schx's own V0dBFS) via an
        E-source, not feed the raw wavefile straight into the circuit."""
        out = L.translate(_tiny_netlist(), wav_path="in.wav", in_scale=2.5, dur=0.5, out_wav="out.wav")
        assert 'wavefile="' in out
        # Cathode="GND" resolves to SPICE ground "0" via node() -- same convention
        # schx_to_ngspice.py's own terms()/node() use.
        m = re.search(r"EJ_IN IN 0 n\S+_raw 0 (\S+)", out)
        assert m, out
        assert abs(float(m.group(1)) - 2.5) < 1e-9

    def test_output_scaled_before_wave_line(self):
        out = L.translate(_tiny_netlist(), wav_path="in.wav", dur=0.5, out_wav="out.wav", out_scale=0.02)
        assert "Eoutscale spkout 0 OUT 0 0.02" in out

    def test_tap_override_wins_over_schx_speaker(self):
        out = L.translate(_tiny_netlist(), wav_path="in.wav", dur=0.5, out_wav="out.wav", tap="IN")
        assert "Eoutscale spkout 0 IN 0" in out

    def test_no_speaker_and_no_tap_is_a_hard_error(self):
        nl = _tiny_netlist()
        nl["components"] = [c for c in nl["components"] if c["type"] != "Speaker"]
        try:
            L.translate(nl, wav_path="in.wav", dur=0.5, out_wav="out.wav")
            assert False, "expected SystemExit"
        except SystemExit:
            pass

    def test_uic_flag_appends_uic_to_tran_line(self):
        out = L.translate(_tiny_netlist(), wav_path="in.wav", dur=0.5, out_wav="out.wav", uic=True)
        tran_line = next(ln for ln in out.splitlines() if ln.startswith(".tran"))
        assert tran_line.endswith(" uic")

    def test_method_override_appears_in_options(self):
        out = L.translate(_tiny_netlist(), wav_path="in.wav", dur=0.5, out_wav="out.wav", method="gear")
        opts_line = next(ln for ln in out.splitlines() if ln.startswith(".options"))
        assert "method=gear" in opts_line


class TestCenterTapTransformerReuse:
    """The CenterTapTransformer branch calls NG.centertap_lines() directly (extracted from
    schx_to_ngspice.py's own translate(), 2026-09-24, verified byte-identical output there) --
    this is the biggest single piece of reuse in this module (standard E/F-controlled-source
    and K-coupled-inductor SPICE syntax, identical across both dialects), so it is worth its
    own smoke test rather than only relying on schx_to_ngspice.py's coverage."""

    def test_ideal_xfmr_emits_e_and_f_sources(self):
        xfmr = {"name": "TX1", "type": "CenterTapTransformer", "value": "7:320", "isPot": False,
                "params": {},
                "terminals": [{"name": "PA", "node": "GND"}, {"name": "PC", "node": "nSpk"},
                              {"name": "SA", "node": "nA"}, {"name": "ST", "node": "BP"},
                              {"name": "SC", "node": "nC"}]}
        nl = _tiny_netlist([xfmr])
        out = L.translate(nl, wav_path="in.wav", dur=0.5, out_wav="out.wav")
        assert "ETX1_a" in out and "FTX1_a" in out
        assert "KTX1" not in out  # ideal path (default) never emits mutual-inductance K-lines


class TestReusedModelBuilders:
    """Diode/BJT/JFET .model strings are standard SPICE parameter lists (no ternary, no
    dialect-specific syntax) -- reused directly from schx_to_ngspice.py with zero wrapping.
    Confirms the wiring, not the math (already covered by test_schx_to_ngspice.py)."""

    def test_diode_uses_ng_diode_model_verbatim(self):
        import schx_to_ngspice as NG
        p, conv = {"IS": "1e-14", "n": "1"}, {}
        assert L.NG.diode_model(p, conv) == NG.diode_model(p, conv)
