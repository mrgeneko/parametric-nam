#!/usr/bin/env bash
#
# One-shot dev setup for spice-to-nam. Idempotent — safe to re-run.
#
#   ./setup.sh                # full setup
#   ./setup.sh --no-cli       # skip the .NET livespice_cli build (Python-only work)
#   RID=linux-arm64 ./setup.sh   # override the auto-detected .NET runtime id
#
# Does, in order:
#   1. init submodules (LiveSPICE, recursive)
#   2. apply the vendored micro-sign patch to LiveSPICE  (idempotent; see below)
#   3. build the livespice_cli simulator for this platform  (needs .NET SDK)
#   4. create .venv and install pinned Python deps
#
# The micro-sign patch is REQUIRED and easy to forget: without it LiveSPICE parses
# "µF" (U+00B5) as FARADS, silently corrupting every µ-valued circuit. This script
# applies it automatically and refuses to leave it unapplied.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PATCH="$REPO/patches/livespice-microsign-prefix.patch"
LS="$REPO/extern/LiveSPICE"
BUILD_CLI=1
for a in "$@"; do [ "$a" = "--no-cli" ] && BUILD_CLI=0; done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 1. submodules ---------------------------------------------------------
say "1/4  submodules (LiveSPICE, recursive)"
git -C "$REPO" submodule update --init --recursive

# --- 2. micro-sign patch (idempotent) -------------------------------------
say "2/4  LiveSPICE micro-sign patch"
if [ ! -f "$PATCH" ]; then
  echo "ERROR: patch not found: $PATCH" >&2; exit 1
fi
if git -C "$LS" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "    already applied — ok"
elif git -C "$LS" apply --check "$PATCH" >/dev/null 2>&1; then
  git -C "$LS" apply "$PATCH"; echo "    applied."
else
  echo "ERROR: micro-sign patch does not apply cleanly to extern/LiveSPICE." >&2
  echo "       (submodule may be at an unexpected revision — inspect manually.)" >&2
  exit 1
fi

# --- 3. build livespice_cli ------------------------------------------------
if [ "$BUILD_CLI" -eq 1 ]; then
  say "3/4  build livespice_cli (.NET)"
  if ! command -v dotnet >/dev/null 2>&1; then
    echo "ERROR: 'dotnet' not found. Install the .NET SDK (>=8):" >&2
    echo "       macOS: brew install --cask dotnet-sdk   |   Linux: https://dotnet.microsoft.com/download" >&2
    exit 1
  fi
  # auto-detect .NET runtime id unless overridden via $RID
  if [ -z "${RID:-}" ]; then
    case "$(uname -s)-$(uname -m)" in
      Darwin-arm64)  RID=osx-arm64 ;;
      Darwin-x86_64) RID=osx-x64 ;;
      Linux-x86_64)  RID=linux-x64 ;;
      Linux-aarch64|Linux-arm64) RID=linux-arm64 ;;
      *) echo "ERROR: unknown platform $(uname -s)-$(uname -m); set RID=... manually." >&2; exit 1 ;;
    esac
  fi
  echo "    runtime id: $RID"
  ( cd "$REPO/livespice_cli" && dotnet publish -c Release -r "$RID" --self-contained -o publish/ )
  echo "    built: livespice_cli/publish/livespice_cli"
else
  say "3/4  build livespice_cli — SKIPPED (--no-cli)"
fi

# --- 4. python venv + deps -------------------------------------------------
say "4/4  Python venv + deps"
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/python" -m pip install --quiet --upgrade pip
"$REPO/.venv/bin/pip" install --quiet -r "$REPO/requirements.txt"
echo "    installed into .venv:"
"$REPO/.venv/bin/python" -c "import torch,numpy,scipy,soundfile as sf; print(f'      torch {torch.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, soundfile {sf.__version__}')"

say "done. Activate with:  . .venv/bin/activate"
[ "$BUILD_CLI" -eq 1 ] && echo "Smoke-test the CLI:  ./livespice_cli/publish/livespice_cli --help"
exit 0
