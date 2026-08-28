#!/usr/bin/env python3
"""
plot_tone_response.py — frequency-response documentation for an exported parametric .nam.

Renders a low-level log sine sweep through EVERY tier of a SlimmableContainer .param.nam,
at each tone knob's min and max (other knobs held mid, the drive knob held low), and draws
a grid of magnitude-response curves: rows = tiers, cols = swept knobs. This documents the
model's SMALL-SIGNAL EQ (the linear tone-stack shaping) as it will actually ship -- it is
rendered through the real C++ product path (render_parametric), tier-selected via --size.

Nothing about the model is hardcoded:
  * tiers + size ratios <- the .param.nam submodels (channels, max_value)
  * knob order + bounds <- the dataset config.json (knobs, bounds, input.samplerate)
The drive knob (default: the first knob, e.g. Gain/Drive) is held LOW so the sweep stays
quasi-linear and the curves reflect frequency shaping, not distortion. It is NOT swept,
because a linear magnitude response is the wrong tool for a distortion control; pass
--include-drive to chart it anyway.

SCHX OVERLAY (--schx): the model's own response only tells you the tiers agree with EACH
OTHER, not whether they match the real circuit. Pass --schx to also render the schematic
through the LiveSPICE oracle (livespice_cli) at the same settings; the chart then becomes a
per-knob overlay of each tier's KNOB EFFECT [dB(max)-dB(min)] against the circuit's (ground
truth), and the summary reports each tier's error vs the circuit. This is what actually
answers "does the model's Bass/Mid/Treble magnitude match the schx?".

Usage:
  python plot_tone_response.py \
      --model  bundle.optimal.param.nam \
      --config dataset/config.json \
      --out    tone_response.svg \
      [--render-bin PATH] [--summary tone_response.md] \
      [--drive-knob Gain] [--drive-value 0.29] [--include-drive] \
      [--schx circuit.schx] [--oracle PATH] [--oversample 4] [--iterations 20]

--render-bin defaults to $RENDER_PARAMETRIC, else the first render_parametric found on PATH
or in a couple of known build dirs. If none is found the tool exits 0 with a warning (so a
release pipeline can call it best-effort without the C++ fork built). --oracle likewise
defaults to $LIVESPICE_CLI / PATH / the livespice-cli oracle build; if --schx is given but no
oracle is found, the overlay is skipped and the plain model-only chart is produced instead.
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

F0, F1, DUR, AMP = 20.0, 20000.0, 10.0, 0.05    # low-level ESS -> quasi-linear
FIT_LO, FIT_HI = 40.0, 10000.0                  # band for schx-fidelity error (guitar-relevant)
COL = {"lo": "#59a5ff", "hi": "#ff8c42"}
SCHX_COL = "#e6e6e6"
TIER_PALETTE = ["#ff8c42", "#59a5ff", "#4ade80", "#e879c9", "#fbbf24"]


def find_render_bin(explicit):
    for c in (explicit, os.environ.get("RENDER_PARAMETRIC")):
        if c and Path(c).is_file():
            return c
    which = shutil.which("render_parametric")
    if which:
        return which
    return None


def tiers_from_model(model_path):
    """[(size_ratio, label, max_value), ...] ascending, from the container submodels."""
    d = json.load(open(model_path))
    subs = d.get("config", {}).get("submodels", [])
    if not subs:
        raise SystemExit(f"{model_path}: not a SlimmableContainer (no submodels) — nothing to chart")
    tiers, lower = [], 0.0
    for s in subs:
        mv = float(s["max_value"])
        ch = s["model"]["config"].get("layers")
        ch = ch[0] if isinstance(ch, list) else ch
        ratio = (lower + mv) / 2.0          # a value the container maps to THIS submodel
        tiers.append((ratio, f"w{ch}" if ch else f"tier{len(tiers)}", mv))
        lower = mv
    return tiers


def make_sweep(sr):
    """Exponential sine sweep + its Farina inverse filter (spectrally-flat deconvolution).

    A naive |FFT_out|/|FFT_in| ratio is unusable here: an ESS deposits almost no energy in
    the top octaves (~-60 dB at 16-20 kHz vs 1 kHz), so the ratio divides by near-zero and
    the HF blows up into tens of dB of spurious rise. Farina deconvolution is a matched
    filter -- immune to the sweep's energy taper -- and time-separates harmonic/alias
    products (they land ~1 s before the linear impulse), so the recovered linear response
    is clean full-band.
    """
    n = int(sr * DUR)
    t = np.arange(n) / sr
    R = math.log(F1 / F0)
    x = (AMP * np.sin(2 * np.pi * F0 * (DUR / R) * (np.exp(t * R / DUR) - 1.0))).astype(np.float32)
    fade = 2048
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    inv = (x[::-1] * np.exp(-t * R / DUR)).astype(np.float64)   # +6dB/oct time-reversed
    inv /= np.max(np.abs(fftconvolve(x, inv)))                  # unity at the linear-IR peak
    return x, inv


def freq_response(x_in, inv, y_out, sr, fpts):
    h = fftconvolve(y_out.astype(np.float64), inv)
    pk = int(np.argmax(np.abs(h)))
    seg = h[max(0, pk - 256): pk + 6000]
    seg = seg * np.hanning(len(seg))
    H = 20 * np.log10(np.abs(np.fft.rfft(seg)) + 1e-12)
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    band = (freqs >= F0) & (freqs <= F1)
    out = np.full(len(fpts), np.nan)
    for i, fc in enumerate(fpts):                       # 1/12-octave smoothing
        m = band & (freqs >= fc / 2 ** (1 / 24)) & (freqs <= fc * 2 ** (1 / 24))
        if m.any():
            out[i] = H[m].mean()
    good = ~np.isnan(out)
    return np.interp(np.arange(len(out)), np.where(good)[0], out[good])


def render(bin_path, model, sweep_wav, knobs, size_ratio, tmp):
    out = str(Path(tmp) / "r.wav")
    kv = ",".join(f"{k:.4f}" for k in knobs)
    subprocess.run([bin_path, model, sweep_wav, out, "--knobs", kv, "--size", str(size_ratio)],
                   check=True, capture_output=True)
    y, _ = sf.read(out, dtype="float32")
    return y


def find_oracle_bin(explicit):
    for c in (explicit, os.environ.get("LIVESPICE_CLI")):
        if c and Path(c).is_file():
            return c
    which = shutil.which("livespice_cli")
    if which:
        return which
    hit = Path.home() / "work/livespice-cli/publish/livespice_cli"
    return str(hit) if hit.is_file() else None


def render_oracle(oracle, schx, sweep_wav, settings, fixed_params, param_map, oversample, iters, tmp):
    """Render the real circuit at a knob setting through the LiveSPICE oracle (ground truth)."""
    out = str(Path(tmp) / "o.wav")
    swept = ",".join(f"{param_map.get(k, k)}={v}" for k, v in settings.items())
    params = f"{fixed_params},{swept}" if fixed_params else swept
    a = [oracle, "--input", sweep_wav, "--output", out, "--circuit", schx, "--params", params]
    if oversample:
        a += ["--oversample", str(oversample)]
    if iters:
        a += ["--iterations", str(iters)]
    r = subprocess.run(a, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"oracle render failed ({params}):\n{r.stderr[-400:]}")
    y, _ = sf.read(out, dtype="float32")
    return y


def build_svg(fpts, curves, tiers, swept, ymin, ymax):
    PW, PH, ML, MT, GX, GY = 340, 200, 60, 66, 42, 60
    ncol, nrow = len(swept), len(tiers)
    W = ML + ncol * PW + (ncol - 1) * GX + 30
    H = MT + nrow * PH + (nrow - 1) * GY + 64
    LFX0, LFX1 = math.log10(F0), math.log10(F1)
    fx = lambda x0, f: x0 + (math.log10(f) - LFX0) / (LFX1 - LFX0) * PW
    fy = lambda y0, db: y0 + (1 - (db - ymin) / (ymax - ymin)) * PH
    xt = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    xl = {20:"20",50:"50",100:"100",200:"200",500:"500",1000:"1k",2000:"2k",5000:"5k",10000:"10k",20000:"20k"}
    g = [f'<rect width="{W}" height="{H}" fill="#0b0e14"/>',
         f'<text x="{W/2}" y="30" fill="#e6e6e6" font-family="system-ui,Arial" font-size="17" '
         f'font-weight="600" text-anchor="middle">Tone-stack frequency response (small-signal)</text>',
         f'<text x="{W/2}" y="50" fill="#8b93a7" font-family="system-ui,Arial" font-size="12" '
         f'text-anchor="middle">magnitude vs input (dB) • solid = knob max, dashed = knob min • '
         f'rows: {" / ".join(t[1] for t in reversed(tiers))} (widest top)</text>']
    for ri, (_, tlabel, _) in enumerate(reversed(tiers)):
        for ci, knob in enumerate(swept):
            x0 = ML + ci * (PW + GX)
            y0 = MT + ri * (PH + GY)
            g.append(f'<rect x="{x0}" y="{y0}" width="{PW}" height="{PH}" fill="#11151f" stroke="#232936"/>')
            for db in range(int(ymin), int(ymax) + 1, 6):
                y = fy(y0, db)
                g.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+PW}" y2="{y:.1f}" stroke="#1b2130"/>')
                g.append(f'<text x="{x0-6}" y="{y+3:.1f}" fill="#6b7488" font-family="system-ui,Arial" '
                         f'font-size="9" text-anchor="end">{db}</text>')
            for f in xt:
                xx = fx(x0, f)
                g.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+PH}" stroke="#161b26"/>')
                if ri == nrow - 1:
                    g.append(f'<text x="{xx:.1f}" y="{y0+PH+15}" fill="#6b7488" '
                             f'font-family="system-ui,Arial" font-size="9" text-anchor="middle">{xl[f]}</text>')
            for lv in ("lo", "hi"):
                arr = curves[(tlabel, knob, lv)]
                pts = " ".join(f"{fx(x0,f):.1f},{fy(y0,d):.1f}" for f, d in zip(fpts, arr))
                dash = '' if lv == "hi" else ' stroke-dasharray="5,4"'
                g.append(f'<polyline points="{pts}" fill="none" stroke="{COL[lv]}" stroke-width="2"{dash}/>')
            g.append(f'<text x="{x0+8}" y="{y0+16}" fill="#cfd6e6" font-family="system-ui,Arial" '
                     f'font-size="13" font-weight="600">{tlabel} · {knob}</text>')
    g.append(f'<text x="{ML+(ncol*PW+(ncol-1)*GX)/2}" y="{H-16}" fill="#a9b2c7" '
             f'font-family="system-ui,Arial" font-size="12" text-anchor="middle">frequency (Hz, log)</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">' \
           + "".join(g) + '</svg>'


def build_overlay_svg(fpts, effect, tiers, swept, tier_col):
    """One panel per swept knob; each panel overlays the knob-EFFECT curve dB(max)-dB(min)
    for the schx (ground truth) and every tier. effect[(source, knob)] -> dB array."""
    allc = np.concatenate([v for v in effect.values()])
    ymin = math.floor(min(allc.min(), -2) / 2) * 2
    ymax = math.ceil(max(allc.max(), 2) / 2) * 2
    PW, PH, ML, MT, GX = 360, 300, 62, 74, 44
    ncol = len(swept)
    W = ML + ncol * PW + (ncol - 1) * GX + 30
    H = MT + PH + 78
    LFX0, LFX1 = math.log10(F0), math.log10(F1)
    fx = lambda x0, f: x0 + (math.log10(f) - LFX0) / (LFX1 - LFX0) * PW
    fy = lambda db: MT + (1 - (db - ymin) / (ymax - ymin)) * PH
    xt = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    xl = {20:"20",50:"50",100:"100",200:"200",500:"500",1000:"1k",2000:"2k",5000:"5k",10000:"10k",20000:"20k"}
    legend = "white = schx (circuit ground truth) • " + \
             " • ".join(f'<tspan fill="{tier_col[t[1]]}">{t[1]}</tspan>' for t in tiers)
    g = [f'<rect width="{W}" height="{H}" fill="#0b0e14"/>',
         f'<text x="{W/2}" y="30" fill="#e6e6e6" font-family="system-ui,Arial" font-size="17" '
         f'font-weight="600" text-anchor="middle">Knob effect vs the real circuit</text>',
         f'<text x="{W/2}" y="50" fill="#8b93a7" font-family="system-ui,Arial" font-size="12" '
         f'text-anchor="middle">dB(knob max) − dB(knob min), small-signal • {legend}</text>']
    for ci, knob in enumerate(swept):
        x0 = ML + ci * (PW + GX)
        g.append(f'<rect x="{x0}" y="{MT}" width="{PW}" height="{PH}" fill="#11151f" stroke="#232936"/>')
        for db in range(int(ymin), int(ymax) + 1, 2):
            y = fy(db)
            c = "#2a3346" if db == 0 else "#1b2130"
            g.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+PW}" y2="{y:.1f}" stroke="{c}"/>')
            g.append(f'<text x="{x0-6}" y="{y+3:.1f}" fill="#6b7488" font-family="system-ui,Arial" '
                     f'font-size="9" text-anchor="end">{db:+d}</text>')
        # shade the fidelity-fit band
        g.append(f'<rect x="{fx(x0,FIT_LO):.1f}" y="{MT}" width="{fx(x0,FIT_HI)-fx(x0,FIT_LO):.1f}" '
                 f'height="{PH}" fill="#ffffff" opacity="0.03"/>')
        for f in xt:
            xx = fx(x0, f)
            g.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{MT+PH}" stroke="#161b26"/>')
            g.append(f'<text x="{xx:.1f}" y="{MT+PH+15}" fill="#6b7488" font-family="system-ui,Arial" '
                     f'font-size="9" text-anchor="middle">{xl[f]}</text>')
        order = ["schx"] + [t[1] for t in tiers]
        for src in order:
            if (src, knob) not in effect:
                continue
            col = SCHX_COL if src == "schx" else tier_col[src]
            dash = '' if src == "schx" or src == tiers[-1][1] else ' stroke-dasharray="6,4"'
            wid = 2.6 if src == "schx" else 2.2
            pts = " ".join(f"{fx(x0,f):.1f},{fy(d):.1f}" for f, d in zip(fpts, effect[(src, knob)]))
            g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wid}"{dash}/>')
        g.append(f'<text x="{x0+8}" y="{MT+18}" fill="#cfd6e6" font-family="system-ui,Arial" '
                 f'font-size="14" font-weight="600">{knob}</text>')
    g.append(f'<text x="{ML+(ncol*PW+(ncol-1)*GX)/2}" y="{H-14}" fill="#a9b2c7" '
             f'font-family="system-ui,Arial" font-size="12" text-anchor="middle">'
             f'frequency (Hz, log) — shaded = {FIT_LO:.0f} Hz–{FIT_HI/1000:.0f} kHz error band</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">' \
           + "".join(g) + '</svg>'


def main():
    ap = argparse.ArgumentParser(description="Frequency-response chart for a parametric .nam")
    ap.add_argument("--model", required=True, help="exported .param.nam (SlimmableContainer)")
    ap.add_argument("--config", required=True, help="dataset config.json (knobs + bounds + sr)")
    ap.add_argument("--out", required=True, help="output .svg")
    ap.add_argument("--summary", help="optional markdown summary table")
    ap.add_argument("--render-bin", help="render_parametric path (else $RENDER_PARAMETRIC / PATH)")
    ap.add_argument("--drive-knob", help="knob held low during the sweep (default: first knob)")
    ap.add_argument("--drive-value", type=float, help="value for the drive knob (default: 25%% of its range)")
    ap.add_argument("--include-drive", action="store_true", help="also sweep the drive knob")
    ap.add_argument("--schx", help="overlay the real circuit (ground truth) via the LiveSPICE oracle")
    ap.add_argument("--oracle", help="livespice_cli path (else $LIVESPICE_CLI / PATH / livespice-cli build)")
    ap.add_argument("--oversample", type=int, help="oracle oversample (default: config's, else 4)")
    ap.add_argument("--iterations", type=int, default=20, help="oracle Newton iterations (default 20)")
    args = ap.parse_args()

    rbin = find_render_bin(args.render_bin)
    if not rbin:
        print("plot_tone_response: no render_parametric binary found "
              "(set --render-bin or $RENDER_PARAMETRIC) — skipping chart.", file=sys.stderr)
        return 0

    cfg = json.load(open(args.config))
    knobs = cfg["knobs"]
    bounds = cfg.get("bounds", {})
    sr = int(cfg.get("input", {}).get("samplerate", 48000))
    lo = {k: bounds.get(k, [0.0, 1.0])[0] for k in knobs}
    hi = {k: bounds.get(k, [0.0, 1.0])[1] for k in knobs}
    mid = {k: (lo[k] + hi[k]) / 2 for k in knobs}

    drive = args.drive_knob or knobs[0]
    if drive not in knobs:
        raise SystemExit(f"--drive-knob {drive!r} not in {knobs}")
    drive_val = args.drive_value if args.drive_value is not None else lo[drive] + 0.25 * (hi[drive] - lo[drive])
    swept = [k for k in knobs if args.include_drive or k != drive]
    if not swept:
        raise SystemExit("no knobs to sweep")

    tiers = tiers_from_model(args.model)
    tier_col = {t[1]: TIER_PALETTE[i % len(TIER_PALETTE)] for i, t in enumerate(reversed(tiers))}
    fpts = np.logspace(math.log10(F0), math.log10(F1), 240)

    # Ground-truth overlay is opt-in (--schx) and best-effort: without the oracle we still
    # produce the model-only chart rather than failing.
    obin = None
    if args.schx:
        obin = find_oracle_bin(args.oracle)
        if not obin:
            print("plot_tone_response: --schx given but no livespice_cli found "
                  "(set --oracle or $LIVESPICE_CLI) — producing model-only chart.", file=sys.stderr)
    overlay = bool(args.schx and obin)
    fixed_params = cfg.get("fixed_params", "") or ""
    param_map = cfg.get("param_map", {}) or {}
    oversample = args.oversample if args.oversample is not None else int(cfg.get("oversample", 4))

    def settings(knob, val):
        s = {k: mid[k] for k in knobs}
        s[drive] = drive_val
        s[knob] = val
        return s

    with tempfile.TemporaryDirectory() as tmp:
        sweep_wav = str(Path(tmp) / "sweep.wav")
        x, inv = make_sweep(sr)
        sf.write(sweep_wav, x, sr, subtype="FLOAT")

        curves = {}
        for ratio, tlabel, _ in tiers:
            for knob in swept:
                for lv, val in (("lo", lo[knob]), ("hi", hi[knob])):
                    s = settings(knob, val)
                    y = render(rbin, args.model, sweep_wav, [s[k] for k in knobs], ratio, tmp)
                    curves[(tlabel, knob, lv)] = freq_response(x, inv, y, sr, fpts)
                    print(f"  rendered {tlabel} {knob}={val:.3f}", file=sys.stderr)
        if overlay:
            try:
                for knob in swept:
                    for lv, val in (("lo", lo[knob]), ("hi", hi[knob])):
                        y = render_oracle(obin, args.schx, sweep_wav, settings(knob, val),
                                          fixed_params, param_map, oversample, args.iterations, tmp)
                        curves[("schx", knob, lv)] = freq_response(x, inv, y, sr, fpts)
                        print(f"  rendered schx {knob}={val:.3f}", file=sys.stderr)
            except Exception as e:                      # oracle present but a render failed:
                overlay = False                          # keep the model-only chart, don't lose it
                curves = {k: v for k, v in curves.items() if k[0] != "schx"}
                print(f"plot_tone_response: schx overlay failed ({e}); "
                      f"producing model-only chart.", file=sys.stderr)

    if overlay:
        effect = {(src, knob): curves[(src, knob, "hi")] - curves[(src, knob, "lo")]
                  for src in ["schx"] + [t[1] for t in tiers] for knob in swept}
        Path(args.out).write_text(build_overlay_svg(fpts, effect, tiers, swept, tier_col))
        print(f"wrote {args.out}")
        fit = (fpts >= FIT_LO) & (fpts <= FIT_HI)
        tlabels = [t[1] for t in tiers]
        header = "| knob | schx peak effect | " + " | ".join(f"{t} err (mean/p95)" for t in tlabels) + " |"
        lines = [header, "|---|---|" + "---|" * len(tlabels)]
        for knob in swept:
            se = effect[("schx", knob)]
            cells = [f"{np.abs(se[fit]).max():.1f} dB"]
            for t in tlabels:
                err = np.abs(effect[(t, knob)] - se)[fit]
                cells.append(f"{err.mean():.2f} / {np.percentile(err, 95):.2f} dB")
            lines.append(f"| {knob} | " + " | ".join(cells) + " |")
        # p95 not max: the dB-domain error spikes at tone-stack notches (a small notch-frequency
        # shift = a large local dB gap on a steep skirt), so a single bin would dominate max and
        # overstate the audible discrepancy. The full curve (notches included) is on the chart.
        table = (f"Fidelity vs schx (circuit ground truth), knob-effect error over "
                 f"{FIT_LO:.0f} Hz–{FIT_HI/1000:.0f} kHz, small-signal, drive {drive}={drive_val:.2f}:\n\n"
                 + "\n".join(lines) + "\n")
    else:
        allv = np.concatenate([v for v in curves.values()])
        ymin = math.floor(np.nanpercentile(allv, 0.5) / 3) * 3
        ymax = math.ceil(np.nanpercentile(allv, 99.5) / 3) * 3
        Path(args.out).write_text(build_svg(fpts, curves, tiers, swept, ymin, ymax))
        print(f"wrote {args.out}")
        lines = ["| knob | " + " | ".join(t[1] for t in tiers) + " |",
                 "|---|" + "---|" * len(tiers)]
        for knob in swept:
            cells = [f"{np.abs(curves[(t[1], knob, 'hi')] - curves[(t[1], knob, 'lo')]).max():.1f} dB"
                     for t in tiers]
            lines.append(f"| {knob} | " + " | ".join(cells) + " |")
        table = ("Peak magnitude swing (knob max vs min, small-signal, drive "
                 f"{drive}={drive_val:.2f}):\n\n" + "\n".join(lines) + "\n")

    print("\n" + table)
    if args.summary:
        Path(args.summary).write_text(table)
        print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
