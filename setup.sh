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
# THE ORACLE LIVES IN hotspice, NOT HERE.
# This repo used to vendor its own livespice_cli plus a patched LiveSPICE submodule. The copy
# drifted from the emitter's: a fix making an unknown --params name a hard error (instead of
# silently rendering at the defaults, so every "swept" permutation comes out IDENTICAL and the
# trainer learns the knob does nothing) landed in one and not the other. Two tools built from one
# schematic, disagreeing about what the device's knobs ARE. There is now exactly one, and it builds
# against PRISTINE upstream LiveSPICE — an oracle built from the thing under test is not an oracle.
#
# The micro-sign patch is gone too: livespice_cli now normalises U+00B5 MICRO SIGN -> U+03BC GREEK
# MU when it loads a schematic. (Upstream's Quantity parser knows U+03BC but not U+00B5, so it reads
# "4.7µF" written with U+00B5 as 4.7 FARADS — no throw, no warning, every capacitor a dead short.)
# Normalising the input fixes it without forking the simulator.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
EMITTER="${HOTSPICE:-${LIVESPICE_EMITTER:-$REPO/../hotspice}}"   # LIVESPICE_EMITTER: old name, still honoured
BUILD_CLI=1
for a in "$@"; do [ "$a" = "--no-cli" ] && BUILD_CLI=0; done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 1. the oracle ---------------------------------------------------------
if [ "$BUILD_CLI" -eq 1 ]; then
  say "1/2  the oracle (livespice_cli)"

  if [ -n "${LIVESPICE_CLI:-}" ]; then
    if [ ! -x "$LIVESPICE_CLI" ]; then
      echo "ERROR: \$LIVESPICE_CLI is set but not executable: $LIVESPICE_CLI" >&2; exit 1
    fi
    echo "    using \$LIVESPICE_CLI: $LIVESPICE_CLI"
  else
    if [ ! -d "$EMITTER/oracle" ]; then
      echo "ERROR: the oracle lives in hotspice, which was not found at:" >&2
      echo "         $EMITTER" >&2
      echo "       Clone it as a SIBLING of this repo:" >&2
      echo "         git clone https://github.com/mrgeneko/hotspice" >&2
      echo "       ...or point \$LIVESPICE_EMITTER / \$LIVESPICE_CLI at your checkout." >&2
      exit 1
    fi
    if [ -x "$EMITTER/oracle/publish/livespice_cli" ]; then
      echo "    already built: $EMITTER/oracle/publish/livespice_cli"
    else
      if ! command -v dotnet >/dev/null 2>&1; then
        echo "ERROR: 'dotnet' not found. Install the .NET SDK (>=8):" >&2
        echo "       macOS: brew install --cask dotnet-sdk   |   Linux: https://dotnet.microsoft.com/download" >&2
        exit 1
      fi
      echo "    building via $EMITTER/oracle/build.sh"
      ( cd "$EMITTER/oracle" && ./build.sh )
    fi
  fi
else
  say "1/2  the oracle — SKIPPED (--no-cli)"
fi

# --- 2. python venv + deps -------------------------------------------------
say "2/2  Python venv + deps"
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/python" -m pip install --quiet --upgrade pip
"$REPO/.venv/bin/pip" install --quiet -r "$REPO/requirements.txt"
echo "    installed into .venv:"
"$REPO/.venv/bin/python" -c "import torch,numpy,scipy,soundfile as sf; print(f'      torch {torch.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, soundfile {sf.__version__}')"

say "done. Activate with:  . .venv/bin/activate"
if [ "$BUILD_CLI" -eq 1 ]; then
  "$REPO/.venv/bin/python" -c "
from batch_harness import LIVESPICE_CLI
print(f'Oracle: {LIVESPICE_CLI}  (exists={LIVESPICE_CLI.exists()})')"
fi
exit 0
