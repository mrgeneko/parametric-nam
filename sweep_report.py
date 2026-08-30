#!/usr/bin/env python3
"""
Generate a self-contained HTML sweep analysis report from a set of WAV files.

Each WAV represents one parameter setting (e.g. different gain values).
The report shows overlaid waveforms, frequency spectra, and per-file stats.

Usage:
    python sweep_report.py limelight_gain2_*.wav -o report.html
    python sweep_report.py /tmp/sweep_out/limelight_*.wav --title "Amp Volume Sweep"
    python sweep_report.py *.wav --param-name "volume" --model "Boutique Dual-Channel Amp"
"""

import argparse, json, re, sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Cold → hot color ramp (blue → teal → green → amber → red)
_COLOR_ANCHORS = [
    (0.00, (91,  141, 217)),
    (0.25, (78,  181, 160)),
    (0.50, (139, 195,  74)),
    (0.75, (244, 166,  35)),
    (1.00, (232,  69,  69)),
]


def gain_color(t: float) -> str:
    t = max(0.0, min(1.0, t))
    for i in range(len(_COLOR_ANCHORS) - 1):
        t0, c0 = _COLOR_ANCHORS[i]
        t1, c1 = _COLOR_ANCHORS[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    t_, c_ = _COLOR_ANCHORS[-1]
    return f"#{c_[0]:02x}{c_[1]:02x}{c_[2]:02x}"


def extract_gain_label(path: Path, param_name: str) -> str:
    """Extract a numeric label from the filename.

    Tries param_name first (e.g. 'volume_0.50'), then any trailing float.
    """
    patterns = [
        rf'{re.escape(param_name)}[_=]([0-9]+\.?[0-9]*)',
        r'([0-9]+\.[0-9]+)(?:_[a-z]|\.wav|$)',
        r'_([0-9]+\.[0-9]+)',
    ]
    for pat in patterns:
        m = re.search(pat, path.stem, re.IGNORECASE)
        if m:
            try:
                return f"{float(m.group(1)):.2f}"
            except ValueError:
                pass
    return path.stem


def analyze_wav(path: Path, sr_ref: int = None, waveform_t: float = 10.0,
                waveform_dur: float = 0.5, waveform_pts: int = 500) -> dict:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr_ref is not None and sr != sr_ref:
        raise ValueError(f"{path.name}: SR {sr} != expected {sr_ref}")

    n = len(audio)

    # Waveform window
    t0 = min(waveform_t, n / sr * 0.4)
    i0 = int(t0 * sr)
    i1 = min(int((t0 + waveform_dur) * sr), n)
    window = audio[i0:i1]
    stride = max(1, len(window) // waveform_pts)
    waveform = [round(float(v), 4) for v in window[::stride].tolist()]

    # Spectrum: FFT over full signal, log-binned 40–20 kHz
    fft_mag = np.abs(np.fft.rfft(audio))
    fft_freqs = np.fft.rfftfreq(n, 1.0 / sr)
    ref = max(fft_mag.max(), 1e-10)
    fft_db = 20.0 * np.log10(fft_mag / ref + 1e-12)

    freq_bins = np.logspace(np.log10(40), np.log10(20000), 150)
    spec_db = []
    for i, fc in enumerate(freq_bins):
        flo = float(np.sqrt(freq_bins[i - 1] * fc)) if i > 0 else 20.0
        fhi = float(np.sqrt(fc * freq_bins[i + 1])) if i < len(freq_bins) - 1 else 22050.0
        mask = (fft_freqs >= flo) & (fft_freqs < fhi)
        spec_db.append(round(float(fft_db[mask].max()), 2) if mask.sum() else -200.0)

    # Stats
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.abs(audio).max())
    dc = float(audio.mean())
    clipped = int((np.abs(audio) >= 0.9999).sum())

    # Spectral centroid (energy-weighted mean frequency, 40–20kHz)
    mask_range = (fft_freqs >= 40) & (fft_freqs <= 20000)
    w = fft_mag[mask_range] ** 2
    centroid = int(float(np.sum(fft_freqs[mask_range] * w) / (w.sum() + 1e-12)))

    return {
        "audio": audio,
        "sr": sr,
        "waveform": waveform,
        "spec_db": spec_db,
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "dc": round(dc, 5),
        "clipped": clipped,
        "centroid": centroid,
        "freqs": [round(float(f), 1) for f in freq_bins],
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.corrcoef(a, b)[0, 1])


def build_data(analyses: list[dict], labels: list[str], colors: list[str]) -> dict:
    ref = analyses[0]["audio"]
    corrs = [None] + [round(pearson_corr(ref, a["audio"]), 4) for a in analyses[1:]]

    waveforms, spectra, stats = [], [], []
    for i, (a, lbl, col) in enumerate(zip(analyses, labels, colors)):
        waveforms.append({"g": lbl, "c": col, "s": a["waveform"]})
        spectra.append({"g": lbl, "c": col, "db": a["spec_db"]})
        stats.append({
            "g": lbl, "rms": a["rms"], "peak": a["peak"],
            "dc": a["dc"], "clipped": a["clipped"], "centroid": a["centroid"],
        })

    return {
        "sr": analyses[0]["sr"],
        "waveforms": waveforms,
        "spectra": spectra,
        "stats": stats,
        "freqs": analyses[0]["freqs"],
        "corrs": corrs,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:      #0b0c10; --surface: #13151b; --border: #1e2130;
  --grid:    #1a1d28; --text:    #cdd2e0; --muted:  #4f566b;
  --dim:     #2e3347; --good:    #4eb5a0; --warn:   #f4a623;
  --mono: Menlo, 'Cascadia Code', 'Courier New', monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
body {{ background:var(--bg); color:var(--text); font-family:var(--sans);
       font-size:13px; line-height:1.5; padding:24px 20px 48px;
       max-width:1100px; margin:0 auto; }}
header {{ display:flex; align-items:baseline; gap:16px; margin-bottom:28px;
          border-bottom:1px solid var(--border); padding-bottom:16px; }}
.hd-title  {{ font-family:var(--mono); font-size:11px; letter-spacing:.08em;
              color:var(--muted); text-transform:uppercase; }}
.hd-model  {{ font-family:var(--mono); font-size:13px; color:var(--text); }}
.hd-file   {{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto; }}
.legend    {{ display:flex; gap:20px; margin-bottom:16px; align-items:center; flex-wrap:wrap; }}
.legend-label {{ font-family:var(--mono); font-size:10px; letter-spacing:.06em;
                 color:var(--muted); text-transform:uppercase; margin-right:4px; }}
.legend-item  {{ display:flex; align-items:center; gap:7px;
                 font-family:var(--mono); font-size:11px; color:var(--text); }}
.swatch {{ width:24px; height:2px; border-radius:1px; flex-shrink:0; }}
.panels {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }}
@media (max-width:680px) {{ .panels {{ grid-template-columns:1fr; }} }}
.panel {{ background:var(--surface); border:1px solid var(--border); padding:16px 16px 12px; }}
.panel-title {{ font-family:var(--mono); font-size:10px; letter-spacing:.1em;
                text-transform:uppercase; color:var(--muted); margin-bottom:10px; }}
.panel canvas {{ display:block; width:100%; height:auto; }}
.panel-sub {{ font-family:var(--mono); font-size:9px; color:var(--dim);
              margin-top:6px; letter-spacing:.04em; }}
.status-row {{ display:flex; gap:32px; margin-bottom:20px; padding:12px 16px;
               background:var(--surface); border:1px solid var(--border); flex-wrap:wrap; }}
.stat {{ display:flex; flex-direction:column; gap:2px; }}
.stat-label {{ font-family:var(--mono); font-size:9px; letter-spacing:.08em;
               text-transform:uppercase; color:var(--muted); }}
.stat-value {{ font-family:var(--mono); font-size:13px; color:var(--text); }}
.stat-value.good {{ color:var(--good); }}
.findings-title {{ font-family:var(--mono); font-size:10px; letter-spacing:.1em;
                   text-transform:uppercase; color:var(--muted); margin-bottom:12px; }}
table {{ width:100%; border-collapse:collapse; font-family:var(--mono);
         font-size:11px; font-variant-numeric:tabular-nums; }}
th {{ text-align:left; color:var(--muted); letter-spacing:.07em; font-size:9px;
      text-transform:uppercase; padding:0 12px 8px 0;
      border-bottom:1px solid var(--border); font-weight:normal; }}
td {{ padding:7px 12px 7px 0; border-bottom:1px solid var(--grid);
      color:var(--text); vertical-align:middle; }}
td:last-child, th:last-child {{ padding-right:0; }}
.gain-chip {{ display:inline-flex; align-items:center; gap:7px;
              font-family:var(--mono); font-size:11px; }}
.gain-pip {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.tag {{ display:inline-block; padding:1px 6px; font-family:var(--mono);
        font-size:9px; letter-spacing:.05em; border-radius:2px; }}
.tag-ok   {{ background:rgba(78,181,160,.15); color:var(--good); }}
.tag-warn {{ background:rgba(244,166,35,.15);  color:var(--warn); }}
.corr-bar {{ display:inline-block; height:2px; border-radius:1px;
             background:var(--good); vertical-align:middle; margin-right:6px; }}
</style>
</head>
<body>
<header>
  <div>
    <div class="hd-title">Parametric NAM · Inference Analysis</div>
    <div class="hd-model">__MODEL_INFO__</div>
  </div>
  <div class="hd-file">__FILE_INFO__</div>
</header>
<div class="status-row" id="statusRow"></div>
<div class="legend">
  <span class="legend-label">__PARAM_NAME__</span>
  <span id="legendItems"></span>
</div>
<div class="panels">
  <div class="panel">
    <div class="panel-title">Waveform · __WAVEFORM_DUR__s window @ t=__WAVEFORM_T__s</div>
    <canvas id="waveCanvas" width="800" height="280"></canvas>
    <div class="panel-sub" id="waveSub"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Frequency spectrum · log scale · 40 Hz – 20 kHz</div>
    <canvas id="specCanvas" width="800" height="280"></canvas>
    <div class="panel-sub">Spectral centroid rises with parameter value · dB re: peak</div>
  </div>
</div>
<div>
  <div class="findings-title">Per-channel findings</div>
  <table>
    <thead>
      <tr>
        <th>__PARAM_NAME__</th><th>RMS</th><th>Peak</th><th>DC offset</th>
        <th>Clipped</th><th>Centroid</th><th>Corr vs ref</th><th>Status</th>
      </tr>
    </thead>
    <tbody id="findingsBody"></tbody>
  </table>
</div>
<script>
const DATA = __DATA_JSON__;

// Status row
(function() {{
  const s = document.getElementById('statusRow');
  const stats = DATA.stats;
  const rmsVals = stats.map(x => x.rms);
  const mono = rmsVals.every((v, i) => i === 0 || v >= rmsVals[i-1]);
  const maxDc = Math.max(...stats.map(x => Math.abs(x.dc)));
  const totalClip = stats.reduce((a, x) => a + x.clipped, 0);
  const cMin = Math.min(...DATA.corrs.filter(x => x !== null));
  const phaseOk = DATA.corrs.filter(x => x !== null).every(c => c > 0);
  const centroids = stats.map(x => x.centroid);
  s.innerHTML = `
    <div class="stat"><span class="stat-label">Phase flips</span>
      <span class="stat-value ${phaseOk ? 'good' : ''}">${phaseOk ? 'None' : 'DETECTED'}</span></div>
    <div class="stat"><span class="stat-label">Clipping</span>
      <span class="stat-value ${totalClip === 0 ? 'good' : ''}">${totalClip} samples</span></div>
    <div class="stat"><span class="stat-label">DC offset (max)</span>
      <span class="stat-value ${maxDc < 0.01 ? 'good' : ''}">${(maxDc * 1000).toFixed(2)} mFS</span></div>
    <div class="stat"><span class="stat-label">RMS span</span>
      <span class="stat-value">${rmsVals[0].toFixed(3)} → ${rmsVals[rmsVals.length-1].toFixed(3)} (${(rmsVals[rmsVals.length-1]/rmsVals[0]).toFixed(1)}×)</span></div>
    <div class="stat"><span class="stat-label">Monotonic RMS</span>
      <span class="stat-value ${mono ? 'good' : ''}">${mono ? 'Yes' : 'No'}</span></div>
    <div class="stat"><span class="stat-label">Spectral centroid</span>
      <span class="stat-value">${centroids[0]} → ${centroids[centroids.length-1]} Hz</span></div>
  `;
}})();

// Legend
(function() {{
  const el = document.getElementById('legendItems');
  el.style.display = 'flex'; el.style.gap = '20px'; el.style.flexWrap = 'wrap';
  DATA.waveforms.forEach(ch => {{
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML = `<span class="swatch" style="background:${{ch.c}}"></span>${{ch.g}}`;
    el.appendChild(item);
  }});
}})();

// Wave sub
document.getElementById('waveSub').textContent =
  DATA.corrs.filter(x => x !== null).every(c => c > 0)
    ? 'All files in phase · amplitudes scale with parameter'
    : 'WARNING: some files may have phase issues';

// Draw waveform
(function() {{
  const canvas = document.getElementById('waveCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth, H = 280;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const pad = {{ l:36, r:12, t:10, b:28 }};
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const mid = pad.t + iH / 2;
  ctx.strokeStyle = '#1a1d28'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.t + (i / 4) * iH;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
  }}
  ctx.fillStyle = '#4f566b'; ctx.font = '9px Menlo, monospace'; ctx.textAlign = 'right';
  ctx.fillText('+0.5', pad.l - 4, pad.t + 4);
  ctx.fillText('0',    pad.l - 4, mid + 3);
  ctx.fillText('−0.5', pad.l - 4, pad.t + iH + 4);
  DATA.waveforms.forEach(ch => {{
    const n = ch.s.length;
    ctx.strokeStyle = ch.c; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ch.s.forEach((v, i) => {{
      const x = pad.l + (i / (n - 1)) * iW;
      const y = mid - v * iH * 0.9;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#4f566b'; ctx.font = '9px Menlo, monospace'; ctx.textAlign = 'center';
  const dur = __WAVEFORM_DUR__;
  [0, 100, 200, 300, 400, 500].filter(ms => ms <= dur * 1000).forEach(ms => {{
    const x = pad.l + (ms / (dur * 1000)) * iW;
    ctx.beginPath(); ctx.strokeStyle = '#2e3347'; ctx.lineWidth = 1;
    ctx.moveTo(x, pad.t + iH); ctx.lineTo(x, pad.t + iH + 4); ctx.stroke();
    ctx.fillText(ms + 'ms', x, H - 6);
  }});
}})();

// Draw spectrum
(function() {{
  const canvas = document.getElementById('specCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth, H = 280;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const pad = {{ l:40, r:12, t:10, b:28 }};
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const fMin = Math.log10(40), fMax = Math.log10(20000);
  const dbMin = -20, dbMax = 40;
  const fToX = f => pad.l + ((Math.log10(f) - fMin) / (fMax - fMin)) * iW;
  const dToY = d => pad.t + (1 - (d - dbMin) / (dbMax - dbMin)) * iH;
  const freqs = DATA.freqs;
  ctx.strokeStyle = '#1a1d28'; ctx.lineWidth = 1;
  [100,200,500,1000,2000,5000,10000].forEach(f => {{
    const x = fToX(f);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + iH); ctx.stroke();
  }});
  [-20,-10,0,10,20,30,40].forEach(d => {{
    const y = dToY(d);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
  }});
  ctx.fillStyle = '#4f566b'; ctx.font = '9px Menlo, monospace'; ctx.textAlign = 'center';
  [[100,'100'],[500,'500'],[1000,'1k'],[5000,'5k'],[10000,'10k'],[20000,'20k']].forEach(([f,lbl]) => {{
    ctx.fillText(lbl, fToX(f), H - 6);
  }});
  ctx.textAlign = 'right';
  [-10,0,10,20,30,40].forEach(d => {{ ctx.fillText(d + 'dB', pad.l - 4, dToY(d) + 3); }});
  DATA.spectra.forEach(ch => {{
    ctx.strokeStyle = ch.c; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.9;
    ctx.beginPath();
    let started = false;
    ch.db.forEach((d, i) => {{
      if (d < -100) return;
      const x = fToX(freqs[i]);
      const y = Math.max(pad.t, Math.min(pad.t + iH, dToY(d)));
      if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
  ctx.globalAlpha = 1;
}})();

// Findings table
(function() {{
  const tbody = document.getElementById('findingsBody');
  DATA.stats.forEach((s, i) => {{
    const col = DATA.waveforms[i].c;
    const corr = DATA.corrs[i];
    const dcMs = (s.dc * 1000).toFixed(2);
    const corrCell = corr !== null
      ? `<span class="corr-bar" style="width:${{Math.round(corr*48)}}px;background:${{col}}"></span>${{corr.toFixed(4)}}`
      : '— (ref)';
    const ok = s.clipped === 0 && Math.abs(s.dc) < 0.01 && (corr === null || corr > 0);
    tbody.innerHTML += `<tr>
      <td><span class="gain-chip"><span class="gain-pip" style="background:${{col}}"></span>${{s.g}}</span></td>
      <td>${{s.rms.toFixed(4)}}</td>
      <td>${{s.peak.toFixed(4)}}</td>
      <td>${{parseFloat(dcMs) >= 0 ? '+' : ''}}${{dcMs}} mFS</td>
      <td>${{s.clipped}}</td>
      <td>${{s.centroid}} Hz</td>
      <td>${{corrCell}}</td>
      <td><span class="tag ${{ok ? 'tag-ok' : 'tag-warn'}}">${{ok ? 'PASS' : 'WARN'}}</span></td>
    </tr>`;
  }});
}})();
</script>
</body>
</html>
"""


def render_html(data: dict, title: str, model_info: str, file_info: str,
                param_name: str, waveform_t: float, waveform_dur: float) -> str:
    html = HTML_TEMPLATE
    html = html.replace("__TITLE__",        title)
    html = html.replace("__MODEL_INFO__",   model_info)
    html = html.replace("__FILE_INFO__",    file_info)
    html = html.replace("__PARAM_NAME__",   param_name)
    html = html.replace("__WAVEFORM_T__",   str(waveform_t))
    html = html.replace("__WAVEFORM_DUR__", str(waveform_dur))
    html = html.replace("__DATA_JSON__",    json.dumps(data, separators=(",", ":")))
    return html


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("wavs", nargs="+", type=Path, help="WAV files to analyze (sorted by name)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output HTML path (default: <stem of first file>_report.html)")
    ap.add_argument("--title", default="NAM Sweep Analysis")
    ap.add_argument("--model", default="ParametricA2", dest="model_info")
    ap.add_argument("--param-name", default="gain_2", metavar="NAME")
    ap.add_argument("--waveform-t", type=float, default=10.0,
                    help="Start time (s) of waveform window (default: 10.0)")
    ap.add_argument("--waveform-dur", type=float, default=0.5,
                    help="Duration (s) of waveform window (default: 0.5)")
    args = ap.parse_args()

    paths = sorted(args.wavs)
    if not paths:
        ap.error("No WAV files provided.")
    for p in paths:
        if not p.exists():
            ap.error(f"File not found: {p}")

    print(f"Analyzing {len(paths)} files...")
    analyses, labels = [], []
    sr_ref = None
    for i, p in enumerate(paths):
        label = extract_gain_label(p, args.param_name)
        print(f"  [{i+1}/{len(paths)}] {p.name}  →  label={label!r}")
        a = analyze_wav(p, sr_ref=sr_ref,
                        waveform_t=args.waveform_t,
                        waveform_dur=args.waveform_dur)
        if sr_ref is None:
            sr_ref = a["sr"]
        analyses.append(a)
        labels.append(label)

    n = len(analyses)
    colors = [gain_color(i / max(n - 1, 1)) for i in range(n)]

    data = build_data(analyses, labels, colors)

    sr = analyses[0]["sr"]
    dur_s = len(analyses[0]["audio"]) / sr
    file_info = f"{paths[0].stem}  ·  {sr // 1000}kHz  ·  {dur_s:.2f}s"

    html = render_html(
        data=data,
        title=args.title,
        model_info=args.model_info,
        file_info=file_info,
        param_name=args.param_name,
        waveform_t=args.waveform_t,
        waveform_dur=args.waveform_dur,
    )

    out = args.output or paths[0].parent / f"{paths[0].stem.split('_gain')[0]}_report.html"
    out.write_text(html)
    print(f"\nWrote → {out}")

    # Print summary
    print("\nSummary:")
    for s, col in zip(data["stats"], colors):
        print(f"  {s['g']:>6}  RMS={s['rms']:.4f}  peak={s['peak']:.4f}  "
              f"centroid={s['centroid']}Hz  clipped={s['clipped']}")
    corrs = data["corrs"]
    if any(c is not None for c in corrs):
        bad = [labels[i] for i, c in enumerate(corrs) if c is not None and c < 0]
        if bad:
            print(f"\nWARNING: negative correlation vs reference: {bad}")
        else:
            print(f"\nAll in-phase (min corr={min(c for c in corrs if c is not None):.4f})")


if __name__ == "__main__":
    main()
