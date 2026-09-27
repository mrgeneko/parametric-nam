"""Regression test for the CenterTapTransformer power-conservation bug (fixed 2026-09-27,
livespice-cli commit b28c45c / LiveSPICE commit 161473e).

WHY THIS TEST EXISTS AS A REAL, ENFORCED ASSERTION.
LiveSPICE's own Tests/Test.cs has no pass/fail assertions at all -- `dotnet run -- test` just
renders circuits and dumps stats/plots for a human to eyeball. A wrong ampere-turns equation
silently violated energy conservation (power delivered to a resistive secondary load came out at
2x the power drawn from the primary, for ANY load and ANY turns ratio) and nothing in that repo's
own tooling could have caught it, or would catch a regression. This lives here instead, where it
actually gates: a real subprocess call to the built oracle, checked against real physics.

WHAT IT CHECKS, AND WHY THE CIRCUIT HAS Rsrc.
The circuit is LiveSPICE's own Tests/Circuits/CenterTapTransformer Power Conservation.schx (also
committed there, reused here rather than duplicated): an ideal transformer, driven on the primary
through a finite source resistance `Rsrc` (27.5 ohm) by a voltage source, with two DIFFERENT
resistive loads (1k, 2.2k) on the two secondary halves, returning to a grounded center tap.

An EARLIER version of this circuit drove the primary directly from an ideal (zero-impedance)
voltage source, with no Rsrc. That topology cannot detect this bug at all: the two voltage
equations (`Vp = Vs1 * turns * 2` and `Vp = Vs2 * turns * 2`, both correct and unaffected by the
bug) already force the secondary voltages to a load-independent value of `Vin / (2 * turns)`
regardless of what the (buggy or fixed) current equation says -- an ideal source just supplies
whatever current either version of the equation demands, "for free". Confirmed directly: the
pre-fix binary passed the same assertions this file used to make, with an ideal-source circuit --
i.e. that version of this test provided zero regression protection.

Adding `Rsrc` couples the primary current back into the primary voltage via
`Vp = Vin - Ip * Rsrc`, which is exactly the mechanism that makes the bug externally visible in
real amp circuits (a tube's finite plate resistance plays the same role as Rsrc here -- see the
module-level comment in CenterTapTransformer.cs). Solving the full system in closed form:

    Vp = Vin / (1 + (Rsrc / (k * turns**2)) * (1/Ra + 1/Rc))

where k=4 for the fixed ampere-turns equation (`Ip*turns*2 = Isa+Isc`) and k=2 for the buggy one
(`Ip*turns = Isa+Isc`). With Rsrc=27.5, Ra=1k, Rc=2.2k, turns=1:10, this works out to an exactly
clean `Vp_fixed = Vin/2` vs `Vp_buggy = Vin/3` -- a 1.5x (3.52 dB) ratio in the secondary voltage,
verified empirically against both the pre-fix and post-fix livespice_cli binaries (measured ratio:
1.50000, matching the closed form to 5 significant figures). This is the discriminating assertion
below: `test_secondary_voltage_matches_fixed_closed_form` fails on the pre-fix binary and passes
on the post-fix one, which is the whole point of a regression test for this bug.

The two secondary-half voltages remain equal to each other despite the unequal loads in BOTH the
buggy and fixed cases (Vs1 = Vs2 = Vp / (2*turns) always, independent of Ra/Rc) -- that symmetry
comes from the (always-correct) voltage equations alone, not the current equation, so it's a
separate, complementary assertion, not a substitute for the absolute-value one above.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gen_dataset_from_schx import LIVESPICE_CLI  # noqa: E402

CIRCUIT = (Path(__file__).resolve().parent.parent / "../livespice-cli/extern/LiveSPICE/Tests/"
           "Circuits/CenterTapTransformer Power Conservation.schx").resolve()
TURNS = 1 / 10          # Np : Ns_total, matches the circuit's Turns="1:10"
VIN = 0.1                # probe tone amplitude, volts
RSRC = 27.5              # ohms, matches the circuit's Rsrc
RA = 1000.0              # ohms, matches the circuit's Ra
RC = 2200.0              # ohms, matches the circuit's Rc
SR = 48000

pytestmark = pytest.mark.skipif(
    not LIVESPICE_CLI.exists(), reason=f"livespice_cli oracle not built at {LIVESPICE_CLI}")


def _predicted_secondary_rms(vin_rms: float, k: float) -> float:
    """Closed-form secondary half-voltage RMS for this circuit, given the ampere-turns
    equation's coefficient `k` (4 for the fixed `Ip*turns*2 = Isa+Isc`, 2 for the historically
    buggy `Ip*turns = Isa+Isc`). See the module docstring for the derivation."""
    vp_rms = vin_rms / (1 + (RSRC / (k * TURNS ** 2)) * (1 / RA + 1 / RC))
    return vp_rms / (2 * TURNS)


def _render(tmp_path, probe_node: str) -> np.ndarray:
    """Render the shared test circuit with S_OUT retargeted to `probe_node` (the fleet's own
    non-loading-probe convention: Speaker Impedance="Infinity" draws no current, so retargeting
    it never perturbs the circuit under test)."""
    assert CIRCUIT.exists(), f"expected LiveSPICE test circuit at {CIRCUIT}"
    schx = CIRCUIT.read_text()
    # The shared circuit has no built-in probe. We add one non-invasively by inserting a
    # zero-impedance-draw Speaker on the target node, alongside the existing parts.
    scratch = tmp_path / f"probe_{probe_node}.schx"
    insertion = (
        f'\n  <Element Type="Circuit.Symbol, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" Rotation="0" Flip="false" Position="900,{"0" if probe_node == "nSA" else "100"}">\n'
        f'    <Component _Type="Circuit.Speaker, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" Impedance="∞ Ω" Name="S_OUT" />\n  </Element>\n'
        f'  <Element Type="Circuit.Wire, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" A="900,{"-20" if probe_node == "nSA" else "80"}" '
        f'B="900,{"-30" if probe_node == "nSA" else "70"}" />\n'
        f'  <Element Type="Circuit.Symbol, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" Rotation="0" Flip="false" '
        f'Position="900,{"-30" if probe_node == "nSA" else "70"}">\n'
        f'    <Component _Type="Circuit.NamedWire, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" WireName="{probe_node}" Name="NWprobe_{probe_node}" />\n  </Element>\n'
        f'  <Element Type="Circuit.Wire, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" A="900,{"20" if probe_node == "nSA" else "120"}" '
        f'B="900,{"30" if probe_node == "nSA" else "130"}" />\n'
        f'  <Element Type="Circuit.Symbol, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" Rotation="0" Flip="false" '
        f'Position="900,{"30" if probe_node == "nSA" else "130"}">\n'
        f'    <Component _Type="Circuit.NamedWire, Circuit, Version=1.0.0.0, Culture=neutral, '
        f'PublicKeyToken=null" WireName="GND" Name="NWprobeg_{probe_node}" />\n  </Element>\n'
    )
    scratch.write_text(schx.replace("</Schematic>", insertion + "</Schematic>"))

    in_wav = tmp_path / "probe_in.wav"
    t = np.arange(int(SR * 0.5)) / SR
    sf.write(str(in_wav), (VIN * np.sin(2 * math.pi * 100 * t)).astype(np.float32), SR,
             subtype="FLOAT")
    out_wav = tmp_path / f"out_{probe_node}.wav"
    r = subprocess.run([str(LIVESPICE_CLI), "--input", str(in_wav), "--output", str(out_wav),
                        "--circuit", str(scratch)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"livespice_cli failed on {probe_node}: {r.stderr[-2000:]}"
    y, _ = sf.read(str(out_wav))
    return np.asarray(y, dtype=np.float64)


class TestCenterTapTransformerPowerConservation:
    """Rsrc gives the source finite impedance, which is what makes the ampere-turns bug
    externally visible here -- see the module docstring for the closed-form derivation and why
    an ideal-voltage-source topology (this test's earlier version) could not catch it."""

    def test_secondary_voltage_matches_fixed_closed_form(self, tmp_path):
        """The discriminating assertion: fails on the pre-fix binary (which produces the
        k=2 / Vin-divided-by-3 result, ~3.52 dB lower), passes on the fixed one (k=4)."""
        va = _render(tmp_path, "nSA")
        vc = _render(tmp_path, "nSC")
        skip = int(0.15 * SR)
        rms_a = float(np.sqrt(np.mean(va[skip:] ** 2)))
        rms_c = float(np.sqrt(np.mean(vc[skip:] ** 2)))
        vin_rms = VIN / math.sqrt(2)
        expected = _predicted_secondary_rms(vin_rms, k=4)

        assert rms_a == pytest.approx(expected, rel=0.02)
        assert rms_c == pytest.approx(expected, rel=0.02)

    def test_secondary_halves_agree_with_each_other_despite_unequal_loads(self, tmp_path):
        """Vs1 = Vs2 = Vp / (2*turns) regardless of Ra != Rc: this symmetry comes from the
        (always-correct) voltage equations alone, so it holds in both the buggy and fixed
        solver -- it's a complementary check (catches a hypothetical future regression that
        coupled the two halves' voltage equations to their own individual loads), not a
        substitute for the absolute-value assertion above."""
        va = _render(tmp_path, "nSA")
        vc = _render(tmp_path, "nSC")
        skip = int(0.15 * SR)
        rms_a = float(np.sqrt(np.mean(va[skip:] ** 2)))
        rms_c = float(np.sqrt(np.mean(vc[skip:] ** 2)))
        assert rms_a == pytest.approx(rms_c, rel=0.01)
