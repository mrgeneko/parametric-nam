#!/usr/bin/env bash
#
# One-shot dev setup for parametric-nam. Idempotent — safe to re-run.
#
#   ./setup.sh                # full setup
#   ./setup.sh --no-cli       # skip the oracle build/check (Python-only work)
#   RID=linux-arm64 ./setup.sh   # override the auto-detected .NET runtime id
#
# Does, in order:
#   1. locate (and if needed build) THE ORACLE — livespice_cli
#   2. create .venv and install pinned Python deps
#
# THE ORACLE LIVES IN mrgeneko/livespice-cli, NOT HERE.
# This repo used to vendor its own livespice_cli plus a patched LiveSPICE submodule. The copy
# drifted from hotspice's emitter-testing copy: a fix making an unknown --params name a hard
# error (instead of silently rendering at the defaults, so every "swept" combination comes out
# IDENTICAL and the trainer learns the knob does nothing) landed in one and not the other. Two
# tools built from one schematic, disagreeing about what the device's knobs ARE. There is now
# exactly one, in its own small standalone repo (originally extracted from hotspice/oracle/, which
# has no other functional connection to this project) — it builds against PRISTINE upstream
# LiveSPICE, never a patched fork, because an oracle built from the thing under test is not an
# oracle.
#
# The micro-sign patch is gone too: livespice_cli now normalises U+00B5 MICRO SIGN -> U+03BC GREEK
# MU when it loads a schematic. (Upstream's Quantity parser knows U+03BC but not U+00B5, so it reads
# "4.7µF" written with U+00B5 as 4.7 FARADS — no throw, no warning, every capacitor a dead short.)
# Normalising the input fixes it without forking the simulator.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
ORACLE_REPO="${LIVESPICE_CLI_REPO:-${LIVESPICE_EMITTER:-${HOTSPICE:-$REPO/../livespice-cli}}}"   # LIVESPICE_EMITTER/HOTSPICE: old names, still honoured
BUILD_CLI=1
for a in "$@"; do [ "$a" = "--no-cli" ] && BUILD_CLI=0; done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 1. the oracle ---------------------------------------------------------
if [ "$BUILD_CLI" -eq 1 ]; then
  say "1/3  the oracle (livespice_cli)"

  if [ -n "${LIVESPICE_CLI:-}" ]; then
    if [ ! -x "$LIVESPICE_CLI" ]; then
      echo "ERROR: \$LIVESPICE_CLI is set but not executable: $LIVESPICE_CLI" >&2; exit 1
    fi
    echo "    using \$LIVESPICE_CLI: $LIVESPICE_CLI"
  else
    if [ ! -f "$ORACLE_REPO/build.sh" ]; then
      echo "ERROR: the oracle lives in mrgeneko/livespice-cli, which was not found at:" >&2
      echo "         $ORACLE_REPO" >&2
      echo "       Clone it (recursively — it has its own nested submodule) as a SIBLING of this repo:" >&2
      echo "         git clone --recurse-submodules https://github.com/mrgeneko/livespice-cli" >&2
      echo "       ...or point \$LIVESPICE_CLI_REPO / \$LIVESPICE_CLI at your checkout." >&2
      exit 1
    fi
    if [ -x "$ORACLE_REPO/publish/livespice_cli" ]; then
      echo "    already built: $ORACLE_REPO/publish/livespice_cli"
    else
      if ! command -v dotnet >/dev/null 2>&1; then
        echo "ERROR: 'dotnet' not found. Install the .NET SDK (>=8):" >&2
        echo "       macOS: brew install --cask dotnet-sdk   |   Linux: https://dotnet.microsoft.com/download" >&2
        exit 1
      fi
      echo "    building via $ORACLE_REPO/build.sh"
      ( cd "$ORACLE_REPO" && ./build.sh )
    fi
  fi
else
  say "1/2  the oracle — SKIPPED (--no-cli)"
fi

# --- 2. python venv + deps -------------------------------------------------
say "2/3  Python venv + deps"
# Debian/Ubuntu split venv's ensurepip into a separate apt package. Without it `python3 -m venv`
# fails AFTER creating .venv (no pip inside), and the `[ -d .venv ]` guard below then skips the
# broken venv on every re-run. Check first so nothing half-made is left behind.
if [ ! -d "$REPO/.venv" ] && ! python3 -c 'import ensurepip' 2>/dev/null; then
  echo "ERROR: python3 can't create virtualenvs (ensurepip missing)." >&2
  echo "       Debian/Ubuntu: sudo apt-get install python3-venv" >&2
  exit 1
fi
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/python" -m pip install --quiet --upgrade pip

# An accelerator-specific torch build (ROCm/CUDA, installed per the instructions atop
# requirements.txt) carries a local version segment, e.g. 2.12.1+rocm7.2. pip's resolver does
# NOT treat that as satisfying a bare `torch==2.12.1` pin: a plain `pip install -r
# requirements.txt` silently uninstalls it and reinstalls the generic PyPI build in its place
# (confirmed -- this clobbered a hand-installed ROCm build on a render-fleet box that also does
# local GPU training). If torch already imports in this venv, leave it exactly as it is and
# install everything else from requirements.txt around it.
if EXISTING_TORCH="$("$REPO/.venv/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null)"; then
  echo "    torch $EXISTING_TORCH already installed -- leaving it as-is (pip would otherwise silently swap an accelerator-specific build for the generic PyPI one)"
  grep -vE '^torch==' "$REPO/requirements.txt" | "$REPO/.venv/bin/pip" install --quiet -r /dev/stdin
else
  "$REPO/.venv/bin/pip" install --quiet -r "$REPO/requirements.txt"
fi
echo "    installed into .venv:"
"$REPO/.venv/bin/python" -c "import torch,numpy,scipy,soundfile as sf; print(f'      torch {torch.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, soundfile {sf.__version__}')"

# ngspice is a SYSTEM binary, not a pip package. requirements.txt installs spicelib, the Python
# wrapper that drives it, and spicelib imports fine without the simulator -- so setup used to look
# complete and the first --backend ngspice render was what failed (found on a fresh Ubuntu render
# worker). Warn, don't fail: the default livespice backend never needs it.
if command -v ngspice >/dev/null 2>&1; then
  NGSPICE_VER="$( { ngspice -v </dev/null 2>&1 || true; } | grep -oE 'ngspice-[0-9][0-9.]*' | head -n 1 || true)"
  echo "    ngspice: $(command -v ngspice) (${NGSPICE_VER:-version unknown})"
else
  echo "    WARNING: 'ngspice' not found on PATH -- only needed for --backend ngspice / ngspice-deck." >&2
  echo "             Linux: sudo apt-get install ngspice   |   macOS: brew install ngspice" >&2
fi

# --- 3. the product path (render_parametric) -------------------------------
# WHY THIS EXISTS. Two checks run the model through the REAL C++ product path rather than the
# Python forward pass: ab_realtime_playback.py (FiLM parity) and plot_tone_response.py (the
# release tone chart). Both need the `render_parametric` binary from NeuralAmpModelerCore, and
# until now NOTHING in this repo located, built, or even mentioned how to obtain it -- setup.sh
# provisioned the oracle only.
#
# That gap is how ab_realtime_playback.py stayed broken for months: it raised TypeError on every
# invocation from parametric-nam@19b9f57 until @2076305, no test covers it (it needs this binary
# to run at all), and release_run.sh treats the tone chart as best-effort and exits 0 when the
# binary is missing. A check nobody can run is indistinguishable from a check that passes.
#
# WARN, DO NOT FAIL: the binary is a nice-to-have for training and dataset work, and is only
# required to validate a release. Same policy as ngspice above.
say "3/3  the product path (render_parametric)"
RP=""
if [ -n "${RENDER_PARAMETRIC:-}" ]; then
  if [ -x "$RENDER_PARAMETRIC" ]; then RP="$RENDER_PARAMETRIC"; echo "    using \$RENDER_PARAMETRIC: $RP"
  else echo "    WARNING: \$RENDER_PARAMETRIC is set but not executable: $RENDER_PARAMETRIC" >&2; fi
elif command -v render_parametric >/dev/null 2>&1; then
  RP="$(command -v render_parametric)"; echo "    found on PATH: $RP"
else
  # Look for a NeuralAmpModelerCore checkout as a sibling, or one directory up alongside other
  # products that vendor it. First hit wins; a built binary is preferred over an unbuilt source.
  for cand in "$REPO/../NeuralAmpModelerCore" "$REPO"/../*/NeuralAmpModelerCore; do
    [ -d "$cand" ] || continue
    if [ -x "$cand/build/tools/render_parametric" ]; then RP="$cand/build/tools/render_parametric"; break; fi
    [ -z "$NAMCORE_SRC" ] && NAMCORE_SRC="$cand"
  done
  if [ -n "$RP" ]; then
    echo "    already built: $RP"
  elif [ -n "${NAMCORE_SRC:-}" ]; then
    if command -v cmake >/dev/null 2>&1; then
      echo "    building render_parametric in $NAMCORE_SRC ..."
      if cmake -S "$NAMCORE_SRC" -B "$NAMCORE_SRC/build" >/dev/null 2>&1 \
         && cmake --build "$NAMCORE_SRC/build" --target render_parametric >/dev/null 2>&1 \
         && [ -x "$NAMCORE_SRC/build/tools/render_parametric" ]; then
        RP="$NAMCORE_SRC/build/tools/render_parametric"; echo "    built: $RP"
      else
        echo "    WARNING: build failed in $NAMCORE_SRC -- build it by hand and set \$RENDER_PARAMETRIC." >&2
      fi
    else
      echo "    WARNING: 'cmake' not found; cannot build render_parametric from $NAMCORE_SRC." >&2
    fi
  fi
fi
if [ -z "$RP" ]; then
  echo "    WARNING: 'render_parametric' not found -- Python<->C++ parity (ab_realtime_playback.py)" >&2
  echo "             and the release tone chart (plot_tone_response.py) CANNOT RUN without it." >&2
  echo "             Clone NeuralAmpModelerCore as a sibling of this repo and re-run setup, or" >&2
  echo "             point \$RENDER_PARAMETRIC at an existing build." >&2
else
  echo "    export RENDER_PARAMETRIC=$RP   # release_run.sh and plot_tone_response.py read this"
fi

say "done. Activate with:  . .venv/bin/activate"
if [ "$BUILD_CLI" -eq 1 ]; then
  "$REPO/.venv/bin/python" -c "
from gen_dataset_from_schx import LIVESPICE_CLI
print(f'Oracle: {LIVESPICE_CLI}  (exists={LIVESPICE_CLI.exists()})')"
fi
exit 0
