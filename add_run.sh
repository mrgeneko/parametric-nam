#!/usr/bin/env bash
#
# Package a finished (or killed) training run into a release bundle, VALIDATE the
# .nam payloads, and hand off to parametric-nam-models/add-run.sh.
#
# Nothing about the model shape is hardcoded. Tiers, widths, and the whole training
# config are DERIVED from the run itself:
#   * tier labels + order  <- metrics.csv header ("val_esr_<label>", ascending width)
#   * channel widths       <- the .nam submodels' config.layers
#   * training config      <- best.pt's args_dict (epochs, lr, batch, repeats, ...)
#   * circuit + knobs      <- the dataset's config.json
# So [3,5,8], [3,4,8], [3,8] and [2,4,6,8] all work without edits.
#
# Two things it refuses to publish, because both have already bitten us:
#   * NAM version != 0.7.0   -> the host app rejects the file as "A1"
#   * head_mode != "skip"    -> legacy residual head; renders the wrong function
#
# Usage:
#   ./add_run.sh [--commit] [--push] [--dry-run] [--skip-verify] [--blurb FILE]
#
# Override via env: RUN, CKPT, DS, SCHX, CATEGORY, CIRCUIT, PREFIX, MODELS, PY_BIN
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN="${RUN:-$HOME/work/tmp/bigmuff_v4}"                  # <RUN>.best_<tier>.param.nam + <RUN>.log
CKPT="${CKPT:-${RUN}_ckpt}"                              # best.pt, best_<tier>.pt, metrics.csv
DS="${DS:-$HOME/work/tmp/bigmuff_v4_ds}"
SCHX="${SCHX:-$HOME/work/parametric-devices/pedals/Big Muff Pi V1 (66#5).schx}"
MODELS="${MODELS:-$HOME/work/parametric-nam-models}"
CATEGORY="${CATEGORY:-pedals}"
CIRCUIT="${CIRCUIT:-big-muff-pi-v1-66-5}"
PREFIX="${PREFIX:-bigmuff}"
CONFIG="${CONFIG:-configs/big-muff-v1.toml}"        # training config, for reproduce.sh
PY_BIN="${PY_BIN:-$HERE/.venv/bin/python}"
STAGE="${STAGE:-$HOME/work/tmp/${PREFIX}_release}"
# Moved up from the args-construction site (its only other use) so PRIOR_STATE detection below
# can also see it -- a variant-keyed circuit's prior release lives under CURRENT.<variant>, not
# the bare CURRENT, and PRIOR_STATE needs to run before this var existed to find it.
VARIANT="${VARIANT:-}"

do_commit=0 do_push=0 dry=0 verify=1 BLURB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --commit)      do_commit=1; shift;;
    --push)        do_push=1; do_commit=1; shift;;
    --dry-run)     dry=1; shift;;
    --skip-verify) verify=0; shift;;
    --blurb)       BLURB="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -z "$BLURB" ] && [ -f "$HERE/configs/blurbs/$CIRCUIT.md" ] && BLURB="$HERE/configs/blurbs/$CIRCUIT.md"

[ -f "$HERE/$CONFIG" ] || { echo "no training config at $HERE/$CONFIG (set CONFIG=)" >&2; exit 1; }
[ -f "$CKPT/metrics.csv" ] || { echo "no metrics.csv in $CKPT" >&2; exit 1; }
[ -f "$CKPT/best.pt" ]     || { echo "no best.pt in $CKPT" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. DERIVE the facts. Nothing below is hand-typed.
# ---------------------------------------------------------------------------
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
cat > "$TMPD/facts.py" <<'PY'
import csv, json, re, sys, shlex
from pathlib import Path
import torch

ckpt, ds, logp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, sys.argv[4])          # repo root — for `import param_train`

a = torch.load(str(ckpt / "best.pt"), map_location="cpu", weights_only=False).get("args_dict", {})
# Gate on `widths`, NOT on `slimmable`: the --slimmable flag was removed when the
# non-slimmable path was dropped (8b9bce3), so new checkpoints have no such key.
# A run with no widths is a legacy single-width model — no tiers to compose.
raw_widths = a.get("widths")
if not raw_widths:
    sys.exit("this run has no --widths (legacy single-width model) — no tiers to compose")
widths = [int(w) for w in str(raw_widths).split(",") if w.strip()]

# Tier labels come from the model's OWN rule, so they cannot drift from param_train.
# Do NOT infer them from metrics.csv column order — that order is not guaranteed
# ascending (e.g. jcm800_preamp_v2 wrote "val_esr_full,val_esr_lite").
from param_train import SlimmableParametricA2
tiers = SlimmableParametricA2(1, widths=widths).tier_labels()

# Resolve each tier's metrics column. param_train now writes val_esr_w<width> for EVERY
# tier (the w<N> console/metrics rename, 2f89fee) -- so the endpoints are val_esr_w3 /
# val_esr_w8, NOT val_esr_lite / val_esr_full. Older runs wrote val_esr_<tierlabel>.
# Resolve by width first, then fall back to the label, so both layouts publish.
rows = list(csv.DictReader(open(ckpt / "metrics.csv")))
hdr = rows[0]
def col_for(tier, width):
    for cand in (f"val_esr_w{width}", f"val_esr_{tier}"):
        if cand in hdr:
            return cand
    return None
cols = {t: col_for(t, w) for t, w in zip(tiers, widths)}
missing = [t for t, c in cols.items() if c is None]
if missing:
    sys.exit(f"metrics.csv has no column(s) for tier(s) {missing}; "
             f"header={[c for c in hdr if c.startswith('val_esr_')]}")

esrs, epochs = [], []
for t in tiers:                       # look up BY NAME/width, never by position
    c = cols[t]
    r = min(rows, key=lambda r: float(r[c]))
    esrs.append(f"{float(r[c]):.5f}")
    epochs.append(r["epoch"])
last = max(int(r["epoch"]) for r in rows)

cfg = json.loads((ds / "config.json").read_text())
knobs = cfg.get("knobs", [])
# A capture-sourced dataset (gen_dataset_from_captures.py, .nam or .wav) has no .schx --
# config.json records that as schx=null. Everything schx-specific below (bundling the schematic,
# deriving its git revision, the tone-response ground-truth overlay, the registry handoff) is
# gated on this.
is_capture = cfg.get("schx") is None

def emit(k, v): print(f"{k}={shlex.quote(str(v))}")
def emit_arr(k, vs): print(f"{k}=({' '.join(shlex.quote(str(v)) for v in vs)})")

emit_arr("TIERS", tiers)
emit_arr("WIDTHS", widths)
emit_arr("ESRS", esrs)
emit_arr("EPOCHS", epochs)
emit_arr("KNOBS", knobs)
emit("NTIERS", len(tiers))
emit("LAST", last)
emit("PLANNED", a.get("epochs", "?"))
emit("BATCH", a.get("batch_size", "?"))
emit("LR", a.get("lr", "?"))
emit("CROP", a.get("crop_len", "?"))
emit("REPEATS", a.get("repeats", "?"))
emit("WIDTHS_CSV", a.get("widths", ""))
emit("CIRCUIT_NAME", cfg.get("circuit", "?"))
emit("OVERSAMPLE", cfg.get("oversample", "?"))
emit("BACKEND", cfg.get("backend", "?"))
emit("NPERM", cfg.get("permutation_count", "?"))
emit("INPUT_WAV", Path(str(cfg.get("input_wav", "?"))).name)
# Build-recipe surfacing (2026-08-01): if the input excitation was constructed by
# tools/build_excitation.py, input_provenance() (batch_harness.py) embeds the sidecar recipe
# under cfg["input"]["build_recipe"] -- name the source file here so it's visible in MANIFEST.md
# without opening dataset_config.json; the full recipe (every arg, both hashes) lives there.
_recipe = (cfg.get("input") or {}).get("build_recipe")
emit("RECIPE_NOTE",
     f" (built via `tools/build_excitation.py` from `{_recipe['source']['name']}`; full recipe in `dataset_config.json`'s `input.build_recipe`)"
     if _recipe else "")
emit("FIXED", cfg.get("fixed_params", "") or "none")
emit("IS_CAPTURE", "1" if is_capture else "0")
emit("GEAR_MAKE", cfg.get("gear_make", "?"))
emit("GEAR_MODEL", cfg.get("gear_model", "?"))
emit("GEAR_TYPE", cfg.get("gear_type", "?"))
emit("N_SOURCE_FILES", len(cfg.get("source_files") or []))
# film_gamma_bound (docs/film_runaway_investigation.md "A1") is a forward-pass formula
# parameter, not detectable from weight shapes the way spectral_norm is (export_checkpoint.py's
# own detect_spectral_norm() comment) -- must be read back from args_dict and threaded through
# explicitly, or export_checkpoint.py --compose silently defaults to 0.0 (unbounded gamma),
# baking a DIFFERENT formula than the one actually trained. Confirmed missing (2026-08-04): a
# real run against a spectral_norm=True, film_gamma_bound=32.0 checkpoint composed successfully
# but with only a WARNING, not a failure -- the wrong value would have shipped silently.
emit("FILM_GAMMA_BOUND", a.get("film_gamma_bound", 0.0))
PY
FACTS="$("$PY_BIN" "$TMPD/facts.py" "$CKPT" "$DS" "$RUN.log" "$HERE")"
eval "$FACTS"

echo "==> derived: ${NTIERS} tiers ${TIERS[*]}  widths [${WIDTHS[*]}]  (nothing hardcoded)"

# tier -> checkpoint / .nam filename conventions (param_train.py:1104)
ckpt_of() { [ "$1" = "full" ] && echo "$CKPT/best.pt" || echo "$CKPT/best_$1.pt"; }

for t in "${TIERS[@]}"; do
  [ -f "$RUN.best_$t.param.nam" ] || { echo "MISSING: $RUN.best_$t.param.nam" >&2; exit 1; }
  [ -f "$(ckpt_of "$t")" ]        || { echo "MISSING: $(ckpt_of "$t")" >&2; exit 1; }
done

rm -rf "$STAGE"; mkdir -p "$STAGE"
echo "==> staging $STAGE"

# ---------------------------------------------------------------------------
# 2. VALIDATE the source snapshots, THEN compose, THEN validate the composite.
# ---------------------------------------------------------------------------
cat > "$TMPD/validate.py" <<'PY'
import json, sys
run, names = sys.argv[1], sys.argv[2:]
bad = 0
for t in names:
    p = f"{run}.optimal.param.nam" if t == "optimal" else f"{run}.best_{t}.param.nam"
    d = json.load(open(p))
    errs = []
    if d.get("version") != "0.7.0":
        errs.append(f"version={d.get('version')!r} (the host app rejects <0.7.0 as 'A1')")
    subs = d.get("config", {}).get("submodels", [])
    if not subs:
        errs.append("no submodels")
    for i, s in enumerate(subs):
        m = s.get("model", {})
        par = m.get("config", {}).get("parametric", {})
        if par.get("head_mode") != "skip":
            errs.append(f"tier{i} head_mode={par.get('head_mode')!r} (must be 'skip')")
        if par.get("schema_version") != 1:
            errs.append(f"tier{i} schema_version={par.get('schema_version')!r} (must be 1)")
        if not m.get("weights"):
            errs.append(f"tier{i} has no weights")
    if errs:
        bad = 1
        print(f"  FAIL {t}:")
        for e in errs:
            print(f"        - {e}")
    else:
        l = [s["model"]["config"]["layers"] for s in subs]
        w = [len(s["model"]["weights"]) for s in subs]
        print(f"  ok   {t:<8} v0.7.0  skip  schema=1  widths={l}  weights={w}")
if bad:
    print("\nREFUSING TO PUBLISH: payload validation failed.")
    sys.exit(1)
PY

# Validate the per-tier snapshots FIRST, before composing. They carry the true
# provenance of how the run was trained. export_checkpoint.py rebuilds the model with
# TODAY's code, so composing a legacy residual run would mint a container stamped
# head_mode="skip" over weights trained for the residual forward — correct metadata on
# a wrong model. Refuse before that file can exist.
echo "==> validating source snapshots (before composing) ..."
"$PY_BIN" "$TMPD/validate.py" "$RUN" "${TIERS[@]}" || exit 1

# ---------------------------------------------------------------------------
# 3. Compose the "optimal" container: every tier at ITS OWN best epoch.
#    Same splice param_train.py does at the end of a completed run; we do it here
#    because the run may have been killed before reaching that step. Valid because
#    slimmable tiers share no weights (every state key lives under a tier prefix).
# ---------------------------------------------------------------------------
spec=""
for t in "${TIERS[@]}"; do spec="${spec:+$spec,}$t=$(ckpt_of "$t")"; done
echo "==> composing optimal container: $spec"
"$PY_BIN" "$HERE/export_checkpoint.py" --compose "$spec" --dataset "$DS" \
          --output "$RUN.optimal.param.nam" --film-gamma-bound "$FILM_GAMMA_BOUND"

echo "==> validating the composite ..."
"$PY_BIN" "$TMPD/validate.py" "$RUN" optimal || exit 1
echo "==> payloads validated"

# ---------------------------------------------------------------------------
# 4. Measure what the composite actually buys (so the MANIFEST can't overclaim).
#    Evaluates composite vs the plain best_full snapshot on IDENTICAL fixed val
#    crops. --skip-verify falls back to a weights-only (exact, but ESR-free) check.
# ---------------------------------------------------------------------------
if [ "$verify" -eq 1 ]; then
  echo "==> measuring composite vs best_full on identical fixed val crops ..."
  cat > "$TMPD/verify.py" <<'PY'
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path

sys.path.insert(0, sys.argv[4])          # repo root — MUST precede the param_train import
from param_train import SlimmableParametricA2, ParamDataset, validate

ckpt, ds_dir, run = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
a = torch.load(str(ckpt / "best.pt"), map_location="cpu", weights_only=False)
args = a.get("args_dict", {})
widths = [int(w) for w in str(args["widths"]).split(",")]
dev = "mps" if torch.backends.mps.is_available() else "cpu"

ds = ParamDataset(ds_dir, crop_len=args["crop_len"], repeats=args["repeats"], mmap=True)
n_val = max(1, int(len(ds) * args["val_split"])); n_train = len(ds) - n_val
_, val_ds = torch.utils.data.random_split(
    ds, [n_train, n_val], generator=torch.Generator().manual_seed(args["seed"]))
loader = torch.utils.data.DataLoader(val_ds, batch_size=args["batch_size"],
                                     shuffle=False, num_workers=0)
# Must match the checkpoint's own training-time construction (docs/film_runaway_investigation.md
# "A1+A2") or load_state_dict() can't match keys at all: a spectral_norm-trained checkpoint's
# state dict carries "*.parametrizations.weight.original"/"_u"/"_v" keys that only exist once
# enable_spectral_norm() has wrapped conv/mixin/l1x1 -- a plain (unwrapped) construction has
# "*.weight" directly instead, a hard mismatch, not a numerical difference. Confirmed via a real
# add_run.sh run against a spectral_norm=True checkpoint (2026-08-04): the old unconditional
# `SlimmableParametricA2(..., widths=widths)` here crashed load_state_dict with every conv/
# mixin/l1x1 key mismatched. film_gamma_bound is passed through too since it's an ordinary,
# already-supported constructor arg (see SlimmableParametricA2.__init__) -- the incomplete part
# of A1 is the separate C++ live-playback path, not this pure-Python export/verify step.
m = SlimmableParametricA2(len(ds.param_names), widths=widths,
                          spectral_norm=args.get("spectral_norm", False),
                          film_gamma_bound=args.get("film_gamma_bound", 0.0)).to(dev)
labels = m.tier_labels()

def state(p):
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    return ck["model"] if "model" in ck else ck

full = state(ckpt / "best.pt")
comp = {k: v.clone() for k, v in full.items()}
for i, lbl in enumerate(labels):
    src = full if lbl == "full" else state(ckpt / f"best_{lbl}.pt")
    pref = m.tier_state_prefix(i)
    for k, v in src.items():
        if k.startswith(pref):
            comp[k] = v.clone()

def run_eval(st):
    m.load_state_dict(st); m.to(dev)
    np.random.seed(1234)                     # identical crops for both models
    return validate(m, loader, nn.MSELoss(), dev)[1]

e_full, e_comp = run_eval(full), run_eval(comp)
better = [(l, f - c) for l, c, f in zip(labels, e_comp, e_full) if f - c > 1e-6]
same   = [l for l, c, f in zip(labels, e_comp, e_full) if abs(f - c) <= 1e-6]

if better:
    gains = "; ".join(f"**{l}** by {d:.5f} ESR" for l, d in better)
    s = f"Measured on identical fixed val crops, the composite beats the plain " \
        f"`best_full` snapshot on {gains}"
    if same:
        s += f", and is bit-identical on {', '.join(same)} (those tiers peaked at the same epoch as full)"
    print(s + ".")
else:
    print("Measured on identical fixed val crops, the composite is **indistinguishable from "
          "`best_full`** — every tier peaked at (or effectively at) the full-best epoch.")
PY
  COMPOSITE_NOTE="$("$PY_BIN" "$TMPD/verify.py" "$CKPT" "$DS" "$RUN" "$HERE")"
else
  COMPOSITE_NOTE="(composite not ESR-verified: --skip-verify)"
fi
echo "==> $COMPOSITE_NOTE"

# ---------------------------------------------------------------------------
# 5. Assemble the bundle.
# ---------------------------------------------------------------------------
for t in "${TIERS[@]}"; do
  cp "$RUN.best_$t.param.nam" "$STAGE/${PREFIX}_best_$t.param.nam"
done
cp "$RUN.optimal.param.nam" "$STAGE/${PREFIX}_optimal.param.nam"
cp "$CKPT/metrics.csv" "$STAGE/metrics.csv"
cp "$DS/config.json"   "$STAGE/dataset_config.json"
cp "$DS/params.csv"    "$STAGE/dataset_params.csv"
# Capture-sourced runs (IS_CAPTURE=1, see facts.py) have no .schx to bundle -- config.json's own
# schx=null already says so, and $SCHX is just this script's schx-path default in that case, not
# a real file for this run.
[ "$IS_CAPTURE" -eq 0 ] && cp "$SCHX" "$STAGE/"

# Tone-response documentation: small-signal magnitude frequency response of the shipped
# composite, per tier, at each tone knob's min/max -- rendered through the real C++ product
# path (render_parametric) via Farina deconvolution. With --schx (passed here), it ALSO
# renders the circuit through the LiveSPICE oracle and overlays each tier's knob effect
# against that ground truth + reports the per-knob fidelity error. Best-effort: needs the
# C++ binary (set RENDER_PARAMETRIC, or render_parametric on PATH / the fork's build/tools);
# the schx overlay additionally needs the oracle ($LIVESPICE_CLI / the livespice-cli build). If
# the render binary is unavailable the tool prints a warning and exits 0; if only the oracle
# is missing it falls back to the model-only chart. Non-fatal on error too, so a broken
# chart never blocks a publish.
# Capture-sourced runs have no circuit to overlay against -- plot_tone_response.py already
# falls back to a model-only chart when the schx/oracle side is unavailable, so just omit --schx
# rather than pointing it at a schx that isn't this run's source.
TONE_SCHX_ARGS=()
[ "$IS_CAPTURE" -eq 0 ] && TONE_SCHX_ARGS=(--schx "$SCHX")

TONE_CHART=0
if "$PY_BIN" "$HERE/tools/plot_tone_response.py" \
      --model  "$RUN.optimal.param.nam" \
      --config "$DS/config.json" \
      ${TONE_SCHX_ARGS[@]+"${TONE_SCHX_ARGS[@]}"} \
      --out    "$STAGE/tone_response.svg" \
      --summary "$STAGE/tone_response.md" && [ -f "$STAGE/tone_response.svg" ]; then
  TONE_CHART=1
  echo "==> tone-response chart: tone_response.svg"
else
  echo "==> tone-response chart skipped (render_parametric unavailable or errored)"
fi

if [ "$IS_CAPTURE" -eq 0 ]; then
  SCHX_REPO="$(git -C "$(dirname "$SCHX")" rev-parse --show-toplevel)"
  SCHX_REV="$(git -C "$SCHX_REPO" rev-parse --short HEAD)"
else
  # No schematic for a capture-sourced run -- see PROVENANCE_LINE below for what's shown instead.
  SCHX_REPO=""
  SCHX_REV=""
fi
CODE_REV="$(git -C "$HERE" rev-parse --short HEAD)"

LAST_I=$((NTIERS - 1))
FULL_ESR="${ESRS[$LAST_I]}"
# NB: "${arr[*]}" joins on the FIRST CHARACTER of IFS only, so IFS=', ' gives ",".
# Join explicitly instead.
join_with() { local sep="$1"; shift; local out=""; local x
  for x in "$@"; do out="${out:+$out$sep}$x"; done; printf '%s' "$out"; }
WIDTH_LIST="$(join_with ', ' "${WIDTHS[@]}")"
CH_SUM="$(join_with 'ch + ' "${WIDTHS[@]}")ch"
KNOB_LIST="$(join_with ', ' "${KNOBS[@]}")"
# Open-ended SGDR runs carry epochs=0 (no fixed budget) -- describe them honestly, not "of 0".
STOPPED=""
if [ "$PLANNED" = "0" ] || [ "$PLANNED" = "?" ]; then
  # Don't claim HOW it stopped (manual `touch STOP` vs. the --stale-cycles auto-stop) --
  # param_train.py doesn't persist the stop reason anywhere this script can read (only prints it
  # to the run's own stderr log), so asserting "manually" here would be a guess stated as fact.
  EPOCHS_LINE="open-ended SGDR (restart every 50 epochs) — no fixed budget, stopped at epoch $LAST"
  REACHED="**epoch $LAST** (open-ended SGDR run — no fixed epoch budget)"
else
  EPOCHS_LINE="epochs $PLANNED planned"
  REACHED="**epoch $LAST of $PLANNED**"
  [ "$LAST" != "$PLANNED" ] && STOPPED="**The run was stopped at epoch $LAST of $PLANNED.**"
fi

# per-tier table rows (generated — works for any number of tiers)
tier_rows=""; esr_rows=""
for i in $(seq 0 $LAST_I); do
  t="${TIERS[$i]}"; w="${WIDTHS[$i]}"; e="${ESRS[$i]}"; ep="${EPOCHS[$i]}"
  tier_rows="$tier_rows| \`${PREFIX}_best_$t.param.nam\` | snapshot @ $t-best (${w}ch) | $ep | $e |
"
  esr_rows="$esr_rows| $t | ${w}ch | $e | $ep |
"
done
best_eps="$(IFS='/'; echo "${EPOCHS[*]}")"
esr_inline=""
for i in $(seq 0 $LAST_I); do esr_inline="${esr_inline:+$esr_inline · }${TIERS[$i]} ${ESRS[$i]}"; done

# The framing depends on what the CURRENTLY published bundle (if any) was trained as -- and it is
# DETECTED, not assumed. "Supersedes the residual-head bundles / first correct forward" is only true
# if the prior bundle was actually residual. Once a circuit has migrated to skip (as Big Muff did on
# 2026-07-13), a newer run merely fits the SAME forward better and the prior skip run IS a valid
# baseline; claiming otherwise would be a false provenance claim in a permanent record. So read the
# prior bundle's head_mode rather than treating "a bundle exists" as "residual".
CIRCDIR="$MODELS/$CATEGORY/$CIRCUIT"
PRIOR_STATE=none
if [ -d "$CIRCDIR" ] && [ -n "$(ls -A "$CIRCDIR" 2>/dev/null)" ]; then
  PRIOR_STATE=unknown
  # A variant-keyed circuit's CURRENT pointer is CURRENT.<variant> (models-repo add-run.sh's own
  # convention -- see its "A device with SWITCHES needs a --variant" comment), not the bare
  # CURRENT -- that file doesn't even exist for such a circuit (confirmed: dumble-ots-183-full has
  # only CURRENT.rock-2knob on disk). Falling back to bare CURRENT keeps this working for
  # switchless/no-variant circuits, unchanged.
  _cur="$(cat "$CIRCDIR/CURRENT${VARIANT:+.$VARIANT}" 2>/dev/null || cat "$CIRCDIR/CURRENT" 2>/dev/null || true)"
  _pn=""
  [ -n "$_cur" ] && _pn="$(ls "$CIRCDIR/$_cur"/*optimal*.param.nam "$CIRCDIR/$_cur"/*best_full*.param.nam 2>/dev/null | head -1 || true)"
  if [ -n "$_pn" ]; then
    PRIOR_STATE="$("$PY_BIN" - "$_pn" 2>/dev/null <<'PYX'
import json, sys
d = json.load(open(sys.argv[1]))
subs = d.get("config", {}).get("submodels", [])
modes = set()
for s in subs:
    modes.add(s.get("model", {}).get("config", {}).get("parametric", {}).get("head_mode"))
if not subs:                                   # single-model (non-container) .nam
    modes.add(d.get("config", {}).get("parametric", {}).get("head_mode"))
print("skip" if modes == {"skip"} else "residual" if "residual" in modes else "unknown")
PYX
)" || PRIOR_STATE=unknown
    [ -n "$PRIOR_STATE" ] || PRIOR_STATE=unknown
  fi
fi
echo "==> prior published bundle for $CIRCUIT: $PRIOR_STATE"

CAUSAL='- The head is **strictly causal** (left-padded, K−1) — **zero added latency**, so the model
  stays sample-aligned when summed with parallel paths (no comb filtering).'

case "$PRIOR_STATE" in
  residual)
    supersede_section='## Supersedes the residual-head bundles — this is the first CORRECT forward
Earlier bundles for this circuit were trained with a **residual-only** head while the
product renders **skip-accumulating** (== NAM standard). They compute different functions
from the same weights, so their reported ESRs did not reflect in-product sound. This run
trains the skip head directly:

- `config.parametric.head_mode = "skip"`, `schema_version = 1` on every tier.
'"$CAUSAL"'
- The ESRs below are therefore measured under the **same forward the product runs**.
  Earlier bundles'"'"' ESRs are **not** a valid baseline for these numbers.'
    reading_section='## Reading these numbers
These are the first '"$CIRCUIT_NAME"' ESRs measured under the **correct forward**
(skip-accumulating, matching the product). The residual-head bundles reported numbers under
a forward that never ships, so **they are not a valid baseline** — a tier that looks "worse"
than an older run is not necessarily a regression; the two were never measuring the same thing.'
    ;;
  skip)
    supersede_section='## Supersedes the previous skip-head bundle
The current published bundle for this circuit was **already** trained on the correct
**skip-accumulating** forward (== NAM standard) with a strictly causal head; this run uses the
identical forward and simply fits it better:

- `config.parametric.head_mode = "skip"`, `schema_version = 1` on every tier.
'"$CAUSAL"'
- ESRs are measured under the **same forward** as the previous bundle, so the numbers below are a
  like-for-like comparison — a genuine improvement, not a change in what "ESR" measures.'
    reading_section='## Reading these numbers
Measured under the same **skip-accumulating** forward as the previous '"$CIRCUIT_NAME"' bundle, so
these ESRs are directly comparable to it — lower is a genuine, like-for-like improvement. The
residual-vs-skip migration already happened in an earlier run, so no such caveat applies here.'
    ;;
  none)
    supersede_section='## First release for this circuit
`config.parametric.head_mode = "skip"`, `schema_version = 1` on every tier -- trained
directly against the skip-accumulating forward the product runs (== NAM standard), so
there is no residual-vs-skip baseline mismatch to call out here.

'"$CAUSAL"
    reading_section='## Reading these numbers
First release for this circuit — no earlier bundle exists, so there is no baseline
mismatch to flag. Trained directly against the skip-accumulating forward the product
runs (== NAM standard).'
    ;;
  *)  # bundle exists but head mode unreadable -- make NO residual-vs-skip claim either way
    supersede_section='## Supersedes the previous bundle for this circuit
`config.parametric.head_mode = "skip"`, `schema_version = 1` on every tier -- trained against the
skip-accumulating forward the product runs (== NAM standard). (The previous bundle'"'"'s head mode
could not be read, so no residual-vs-skip baseline claim is made here.)

'"$CAUSAL"
    reading_section='## Reading these numbers
Trained against the skip-accumulating forward the product runs (== NAM standard). The previous
bundle'"'"'s head mode could not be confirmed, so treat cross-run ESR comparisons with care.'
    ;;
esac

# Provenance line: a real .schx + git revision for a schx-sourced run, or the source .nam
# captures' own gear metadata for a capture-sourced one (IS_CAPTURE, see facts.py) -- there is no
# schematic to cite for the latter, config.json already records that as schx=null.
if [ "$IS_CAPTURE" -eq 0 ]; then
  PROVENANCE_LINE="- **Schematic:** \`parametric-devices @ $SCHX_REV\` — \`$(basename "$SCHX")\`."
else
  PROVENANCE_LINE="- **Source:** $N_SOURCE_FILES real captured \`.nam\` files (not a circuit simulation) — $GEAR_MAKE $GEAR_MODEL ($GEAR_TYPE)."
fi

# ---------------------------------------------------------------------------
# 6. MANIFEST.md
# ---------------------------------------------------------------------------
{
cat <<EOF
# $CIRCUIT_NAME — ${NTIERS}-tier slimmable parametric NAM (skip + causal head)

One SlimmableContainer holding **$CH_SUM** tiers, FiLM-conditioned on ${#KNOBS[@]} knobs.

$supersede_section

## Provenance
$PROVENANCE_LINE
- **Training code:** \`parametric-nam @ $CODE_REV\`.
- **Backend:** \`$BACKEND\`, \`--oversample $OVERSAMPLE\`.

## Circuit / knobs
- **Parameters (${#KNOBS[@]}):** \`$KNOB_LIST\`.  **Fixed:** \`$FIXED\`.

## Dataset
- **Grid:** $NPERM permutations. Input \`$INPUT_WAV\`$RECIPE_NOTE.
  See \`dataset_config.json\` / \`dataset_params.csv\`.

## Training
- Slimmable A2, widths **[$WIDTH_LIST]**, FiLM, **skip + causal head**.
- $EPOCHS_LINE, batch $BATCH, lr $LR (cosine), crop-len $CROP, repeats $REPEATS.
  No early stopping (removed 2026-07-19 -- it fired once and cost a 17-22% improvement
  from a cosine tail it cut short). Apple Silicon (MPS).
EOF
[ -n "$STOPPED" ] && echo "- $STOPPED"
cat <<EOF

## Models in this bundle
| file | tier(s) | best epoch(s) | ESR |
|---|---|---|---|
| **\`${PREFIX}_optimal.param.nam\`** | composite ($CH_SUM) | $best_eps | $esr_inline |
$tier_rows
Recommended: **\`${PREFIX}_optimal.param.nam\`** — every tier at its own best epoch.

> **How the composite was made.** $( [ -n "$STOPPED" ] && echo "The run was stopped before its own
> compose step, so this container was spliced afterwards" || echo "This container was spliced" )
> from the per-tier checkpoints (\`export_checkpoint.py --compose\`) — the identical operation
> \`param_train.py\` performs at the end of a completed run, and valid for the same reason
> (slimmable tiers share no weights; every state key lives under a tier prefix).
>
> $COMPOSITE_NOTE

See \`ESR_RECORD.md\` and \`reproduce.sh\`.
EOF
[ -n "$BLURB" ] && { echo; echo "## Circuit notes"; echo; cat "$BLURB"; }
} > "$STAGE/MANIFEST.md"

# Tone-response section (only if the chart was generated above).
if [ "$TONE_CHART" -eq 1 ]; then
  {
    echo
    echo "## Tone response (small-signal)"
    echo
    echo "Magnitude frequency response of \`${PREFIX}_optimal.param.nam\`, measured through the"
    echo "C++ product path with Farina deconvolution (immune to the sweep's HF energy taper;"
    echo "distortion products time-separated out). When the LiveSPICE oracle was available, the"
    echo "chart overlays each tier's knob effect against the REAL CIRCUIT (ground truth) and the"
    echo "table reports per-knob fidelity error; otherwise it is the model's own per-tier response."
    echo "Drive held low so the curves reflect EQ shaping, not distortion. See \`tone_response.svg\`."
    echo
    cat "$STAGE/tone_response.md"
  } >> "$STAGE/MANIFEST.md"
fi

# ---------------------------------------------------------------------------
# 7. ESR_RECORD.md
# ---------------------------------------------------------------------------
cat > "$STAGE/ESR_RECORD.md" <<EOF
# ESR record — $CIRCUIT_NAME, ${NTIERS}-tier [$WIDTH_LIST], skip + causal head

ESR = error-to-signal ratio (val split), from \`metrics.csv\`.
Reached $REACHED.

| tier | width | ESR | best epoch |
|---|---|---:|---|
$esr_rows
Recommended file: **\`${PREFIX}_optimal.param.nam\`** (composite — each tier at its own best epoch).

$COMPOSITE_NOTE

$reading_section

The head is **strictly causal** — 0 added latency (validated: Python↔C++ A/B corr 1.000000,
ESR 2.0e-12, lag 0).
EOF

# ---------------------------------------------------------------------------
# 8. reproduce.sh
# ---------------------------------------------------------------------------
if [ "$IS_CAPTURE" -eq 0 ]; then
  REPRO_PROV_LINE="#   schematic : parametric-devices @ $SCHX_REV  ($(basename "$SCHX"))"
else
  REPRO_PROV_LINE="#   source    : $N_SOURCE_FILES real captured .nam files -- $GEAR_MAKE $GEAR_MODEL"
fi
cat > "$STAGE/reproduce.sh" <<EOF
#!/usr/bin/env bash
# Reproduce $CIRCUIT_NAME — ${NTIERS}-tier slimmable [$WIDTH_LIST], skip + causal head.
#
# Provenance (see MANIFEST.md):
$REPRO_PROV_LINE
#   code      : parametric-nam  @ $CODE_REV
#   backend   : $BACKEND, --oversample $OVERSAMPLE
#
$( [ -n "$STOPPED" ] && echo "# The published bundle stopped at epoch $LAST/$PLANNED; this runs the full schedule." )
# Expect roughly: $esr_inline
set -euo pipefail

SPICE_TO_NAM="\${SPICE_TO_NAM:-\$HOME/work/parametric-nam}"
OUT="\${OUT:-\$HOME/work/tmp/${PREFIX}_rerun}"
cd "\$SPICE_TO_NAM"

python run_pipeline.py --config $CONFIG \\
  --dataset-dir "$DS" \\
  --nam-output  "\$OUT.param.nam" \\
  --checkpoint-dir "\${OUT}_ckpt" \\
  --widths $WIDTHS_CSV \\
  --epochs $PLANNED
EOF
chmod +x "$STAGE/reproduce.sh"

echo "==> bundle ready:"
ls -1 "$STAGE" | sed 's/^/      /'

# ---------------------------------------------------------------------------
# 9. Hand off to the models repo.
# ---------------------------------------------------------------------------
ADD="$MODELS/add-run.sh"
if [ "$IS_CAPTURE" -eq 0 ]; then
  args=(--release "$STAGE" --category "$CATEGORY" --circuit "$CIRCUIT" --schx-repo "$SCHX_REPO")
else
  # No schematic to validate the run dir against -- add-run.sh skips the devices.toml registry
  # check entirely for --capture-source runs (see its own comments for the tradeoff: no automated
  # directory-name-drift protection for this class of device).
  args=(--release "$STAGE" --category "$CATEGORY" --circuit "$CIRCUIT" --capture-source)
fi
# VARIANT: required by the models repo's own add-run.sh whenever the circuit has switches -- a
# switch changes the TOPOLOGY, so its position is part of WHICH MODEL this is, not a training
# detail (see that script's own comment). Also the right place for a switchless device's
# pinned-knob/swept-range/speaker variants, same doc. (Declared up near RUN/CKPT/etc -- PRIOR_STATE
# detection above needs it too.) Left empty by default so switchless single-release circuits
# (e.g. Big Muff) are unaffected.
[ -n "$VARIANT" ] && args+=(--variant "$VARIANT")
if [ "$do_push" -eq 1 ]; then args+=(--push); elif [ "$do_commit" -eq 1 ]; then args+=(--commit); fi

if [ "$dry" -eq 1 ]; then
  echo
  echo "==> DRY RUN — would invoke:"
  echo "    $ADD ${args[*]}"
  echo
  echo "    NOTE: HISTORY.md's table has only 'ESR full' and 'ESR lite' columns, so any"
  echo "    middle tiers (${TIERS[*]}) are recorded in ESR_RECORD.md / MANIFEST.md only."
  exit 0
fi

"$ADD" "${args[@]}"
