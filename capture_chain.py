#!/usr/bin/env python3
"""The virtual capture chain: model the audio interface a hardware NAM capture goes through.

WHY THIS EXISTS. A standard .nam is trained on a wet file recorded through an audio
interface, whose input stage rolls off below ~20 Hz. NAM's architecture was designed
against that distribution. Rendering a circuit in a simulator and probing a node directly
skips that stage entirely, so our targets can carry sub-audio content no hardware capture
would ever contain -- and NAM's ~132 ms receptive field cannot model content whose state
evolves over seconds.

Measured on Duke of Tone (Distortion), 2026-09-10: 64-74% of the rendered target's energy
sat BELOW 19 Hz (62% below 5 Hz), against 0.49% in the input. Cause is real and faithful to
the pedal -- an out-of-loop BAT46 shunt clipper into a soft bias rail (VB: 47k||47k with
100 uF, tau 2.35 s), passed by an output network with tau 2.0 s (R11 1M with C8||C9 2 uF,
0.08 Hz corner). Both trainers plateaued at the same place: our ParametricA2 at w8 0.525 and
the OFFICIAL upstream nam-full at a comparable ESR on a static single-setting capture. Two
different architectures, one shared receptive field -- the failure tracked the shared
constraint. The predicted floor from unmodellable content, >=0.64, bracketed the observed
0.525.

For scale: 13 other fleet datasets (Joyo, Mesa RED/Orange, JCM800, Duke OVERDRIVE, statics)
all sit at 0.20-9.47% sub-19 Hz. The Distortion is not the tail of that distribution.

FILTER SHAPE is governed by a MUSICAL rule, not an equipment spec: **preserve bass low E
(41.2 Hz) to within -0.5 dB**. That is the lowest note anyone plays through these models, so
it is the thing we must not damage; a converter's datasheet is a proxy for it at best.

3rd order at 18 Hz costs **-0.030 dB** at 41.2 Hz -- inaudible -- and reaches -33.4 dB at
5 Hz. Raising the ORDER buys stopband rejection; raising the CORNER buys it by trading away
bass. Measured on Duke of Tone (Distortion)'s worst corner, sub-19 Hz residual:

    2nd @ 11.8 Hz   10.69%   -0.029 dB @41.2   (the previous default)
    3rd @ 15.0 Hz    7.23%   -0.010 dB @41.2
    3rd @ 18.0 Hz    4.42%   -0.030 dB @41.2   <- default
    3rd @ 29.0 Hz    0.56%   -0.498 dB @41.2   (and -2.27 dB at 5-string low B)

3rd @ 18 Hz more than halves the residual for the same low-E cost as the old 2nd @ 11.8.
Pushing the corner to 29 Hz would halve it again but takes 2.3 dB off low B, permanently and
irreversibly, in a model meant to stay composable with the user's own amp and cab.

The filter's OWN memory does not distinguish these: measured tail energy beyond the 132 ms
receptive field is 0.0000% for both 2nd @ 11.8 Hz and 3rd @ 18 Hz, so neither is harder for
the model to represent. (An earlier version of this note argued the steeper filter was
EASIER on those grounds -- an artifact of a receptive field that was itself wrong by 2.5x.
The case for 3rd @ 18 Hz rests solely on the frequency-response trade above.)

CAUSAL, ALWAYS. sosfilt, never sosfiltfilt. A zero-phase filter makes the target depend on
FUTURE input, so its pre-ringing is unpredictable to a causal model by construction -- a
self-inflicted error floor in the name of removing one. The 3rd-order 18 Hz impulse response
dies well inside the 132 ms receptive field (measured above), so it is learnable.

THIS IS NOT A UNIVERSAL SAFETY NET. It removes sub-audio, which is the SYMPTOM. A circuit
whose multi-second state reaches into the AUDIO band is still unmodellable and this will not
save it -- the Distortion's own bias modulation shows up as a 0.119 ESR audio-band
difference. Use context_sensitivity() to detect that class; an LF-energy check cannot.
"""
import numpy as np
from scipy.signal import butter, sosfilt

# -0.030 dB at bass low E (41.2 Hz); see FILTER SHAPE above.
DEFAULT_CORNER_HZ = 18.0
DEFAULT_ORDER = 3

# The governing invariant: no capture-chain setting, default or per-device override, may
# take more than this off bass low E. Asserted in tests/test_capture_chain.py.
PRESERVE_HZ = 41.2
PRESERVE_MAX_LOSS_DB = 0.5


def _sos(sr, corner_hz=DEFAULT_CORNER_HZ, order=DEFAULT_ORDER):
    if not (0 < corner_hz < sr / 2):
        raise ValueError(f"corner_hz {corner_hz} must be in (0, {sr/2})")
    return butter(order, corner_hz, "hp", fs=sr, output="sos")


def capture_chain(y, sr, corner_hz=DEFAULT_CORNER_HZ, order=DEFAULT_ORDER):
    """Apply the virtual interface input stage. CAUSAL.

    `y` may be 1-D (one signal) or 2-D (combinations x samples); filtering is along the
    LAST axis either way, so a (n_combos, n_samples) dataset filters in one call without
    the caller transposing or looping.
    """
    a = np.asarray(y)
    out = sosfilt(_sos(sr, corner_hz, order), a.astype(np.float64), axis=-1)
    return out.astype(a.dtype, copy=False)


def lf_energy_fraction(y, sr, below_hz=19.0):
    """Fraction (0-1) of `y`'s energy below `below_hz`. The cheap screen.

    19 Hz is a SCREENING band, kept for continuity: every measurement recorded in this repo
    and in the device notes uses it (fleet 0.2-9.5%, Duke of Tone Distortion 64-74%), so
    changing the default would silently break comparison against all of them.

    It was originally justified as the reciprocal of the receptive field -- and that was
    wrong, because the receptive field was wrong. The real figure is 6332 samples /
    131.9 ms (param_train.RECEPTIVE_FIELD_SAMPLES), making the slowest periodicity
    resolvable within one window **7.58 Hz**, not 19 Hz. Pass below_hz=7.58 for the
    structurally-motivated number; 19 Hz remains the comparable one, and is conservative
    in the sense that it counts some content the model can actually resolve.
    """
    a = np.asarray(y, dtype=np.float64).ravel()
    if a.size < 2:
        return 0.0
    w = np.hanning(a.size)
    S = np.abs(np.fft.rfft(a * w)) ** 2
    f = np.fft.rfftfreq(a.size, 1.0 / sr)
    tot = S.sum()
    return float(S[f < below_hz].sum() / tot) if tot > 0 else 0.0


def context_sensitivity(y_ctx_a, y_ctx_b, sr, skip_s=0.2):
    """ESR between two renders of the SAME input preceded by DIFFERENT context.

    The direct test for state outliving the receptive field, and the one that catches what
    lf_energy_fraction cannot: a circuit whose long memory reaches the audio band. Both
    inputs must be the identical segment; pass the post-context portion of each render.
    `skip_s` drops the first 0.2 s, where a genuine short transient still differs.
    """
    a = np.asarray(y_ctx_a, dtype=np.float64).ravel()
    b = np.asarray(y_ctx_b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    s = min(int(skip_s * sr), max(0, n - 1))
    a, b = a[s:n], b[s:n]
    den = (b ** 2).sum()
    return float(((a - b) ** 2).sum() / den) if den > 0 else 0.0


def add_cli_args(ap):
    """Register --no-capture-chain / --capture-hp-hz / --capture-order on a parser.

    Shared so the measurement tools and the renderer cannot drift apart: an onset measured
    through a different chain than the dataset was rendered through sizes the excitation
    against a signal that never existed.
    """
    # store_const + default=None, NOT store_true: argparse cannot distinguish "flag absent"
    # from "flag defaulted to False", so with store_true a config.toml could never turn the
    # chain OFF -- resolve() would read False and be unable to tell whether the user meant it.
    ap.add_argument("--no-capture-chain", action="store_const", const=True, default=None,
                    help="Measure the RAW node instead of through the virtual capture chain "
                         "(see capture_chain.py). Off by default -- a measurement should see "
                         "what the MODEL will be trained on.")
    ap.add_argument("--capture-hp-hz", type=float, default=None,
                    help=f"Capture-chain corner (default {DEFAULT_CORNER_HZ} Hz).")
    ap.add_argument("--capture-order", type=int, default=None,
                    help=f"Capture-chain filter order (default {DEFAULT_ORDER}).")


def cfg_from_args(args):
    """The chain's kwargs from parsed args, or None when disabled. Plain dict: picklable
    across a worker pool and recordable verbatim in config.json."""
    if getattr(args, "no_capture_chain", False):
        return None
    hz = getattr(args, "capture_hp_hz", None)
    order = getattr(args, "capture_order", None)
    return {"corner_hz": DEFAULT_CORNER_HZ if hz is None else hz,
            "order": DEFAULT_ORDER if order is None else order}


def _cfg_lookup(cfg, key):
    """Read `key` from a loaded config, accepting hyphens or underscores.

    run_pipeline.load_config normalises keys to argparse dests (underscores), but
    grid_adequacy.py loads raw TOML where they stay hyphenated. Accept both rather than
    making every caller remember which loader it used.
    """
    if not cfg:
        return None
    for k in (key, key.replace("_", "-")):
        if k in cfg:
            return cfg[k]
    return None


def resolve(args, cfg=None):
    """The capture chain to use: CLI flag > config.toml > default. None when disabled.

    WHY A RESOLVER AND NOT cfg_from_args. Four tools now honour the chain, and a per-device
    override has to reach all of them identically or they measure signals that never
    coexist. Before this, a capture key in a config.toml was picked up by run_pipeline
    (which uses set_defaults) and silently IGNORED by prepare_excitation,
    check_transient_coverage and grid_adequacy, which load the config and cherry-pick
    individual keys. Half-honoured is worse than unsupported: it looks like it works.

    Relies on --no-capture-chain being store_const/None (see add_cli_args): with store_true
    its False default is indistinguishable from an explicit off, so config could never
    disable the chain.
    """
    off = getattr(args, "no_capture_chain", None)
    if off is None:
        off = bool(_cfg_lookup(cfg, "no_capture_chain"))
    if off:
        return None
    hz = getattr(args, "capture_hp_hz", None)
    if hz is None:
        hz = _cfg_lookup(cfg, "capture_hp_hz")
    order = getattr(args, "capture_order", None)
    if order is None:
        order = _cfg_lookup(cfg, "capture_order")
    return {"corner_hz": DEFAULT_CORNER_HZ if hz is None else float(hz),
            "order": DEFAULT_ORDER if order is None else int(order)}


def assert_dataset_match(capture, dataset_dir):
    """Hard-fail if `dataset_dir` was rendered through a different chain than `capture`.

    TWO SOURCES OF TRUTH, and which one governs depends on timing: before a render,
    config.toml is INTENT; after it, the dataset's config.json is FACT. A tool re-probing
    an already-rendered dataset must use what the dataset WAS rendered with. Silently
    honouring a config.toml edited after the render re-introduces exactly the raw-vs-chained
    mismatch this machinery exists to prevent -- so config governs what gets RENDERED, the
    dataset governs what gets MEASURED, and a disagreement is a hard error, not a warning.
    """
    import sys
    actual = read_dataset_chain(dataset_dir)
    reason = mismatch_reason(capture, actual)
    if reason:
        sys.exit(f"ERROR: {dataset_dir} was rendered through a different capture chain than "
                 f"the one resolved from the config/CLI -- {reason}. The DATASET is the "
                 f"authority for anything measured against it; re-render it, or drop the "
                 f"override so it matches. (capture_chain.py)")


def cache_tag(capture):
    """Cache-key fragment identifying the chain a measurement was taken through.

    MUST be in every findpeak cache_extra. The key already carries os/iterations/maxv; without
    the chain too, an onset measured on the RAW node is served to a caller asking for a chained
    one, and vice versa. That is not hypothetical -- on 2026-09-10 a stale findpeak entry
    silently defeated a verified fix to the sweep itself for two full runs, because a failure
    had been cached and the key could not tell the two apart.
    """
    if not capture:
        return "|cap=off"
    return f"|cap={capture['corner_hz']:g}/{capture['order']}"


def describe(capture):
    """One-line, human-readable statement of the chain. Printed at render start.

    Recording the chain in config.json is necessary but not sufficient: nobody reads a JSON
    file before wondering why an ESR moved. A dataset that silently gained (or lost) a
    capture stage is the same shape of trap as a silently cached failure -- invisible until
    it has already cost you a day.
    """
    if not capture:
        return ("capture chain: DISABLED (--no-capture-chain) -- targets keep sub-audio content "
                "no hardware capture would contain, which a ~132 ms receptive field cannot model")
    n = capture["order"]
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return (f"capture chain: {n}{suffix}-order high-pass at {capture['corner_hz']:g} Hz "
            f"(the audio-interface input stage a hardware NAM capture goes through)")


def read_dataset_chain(dataset_dir):
    """The capture chain a dataset was rendered through.

    Returns the dict, or None if the dataset declares it was rendered WITHOUT one, or the
    string "unknown" for a dataset predating the field entirely -- a distinction that matters,
    because "rendered raw on purpose" and "rendered before this existed" warrant different
    treatment and must not be collapsed into one falsy value.
    """
    import json
    from pathlib import Path
    p = Path(dataset_dir) / "config.json"
    if not p.exists():
        return "unknown"
    try:
        cfg = json.loads(p.read_text())
    except Exception:
        return "unknown"
    return cfg.get("capture_chain", "unknown") if "capture_chain" in cfg else "unknown"


def mismatch_reason(a, b):
    """Why chains `a` and `b` are incomparable, or None if they agree.

    "unknown" is treated as compatible with anything: a pre-2026-09-10 dataset cannot be
    proven either way, and hard-failing every historical dataset would be a worse outcome
    than the mismatch this guards against.
    """
    if a == "unknown" or b == "unknown":
        return None
    if bool(a) != bool(b):
        on, off = ("first", "second") if a else ("second", "first")
        return f"{on} has a capture chain, {off} does not"
    if not a and not b:
        return None
    for k in ("corner_hz", "order"):
        if a.get(k) != b.get(k):
            return f"{k} differs: {a.get(k)} vs {b.get(k)}"
    return None
