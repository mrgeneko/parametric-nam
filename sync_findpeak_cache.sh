#!/usr/bin/env bash
#
# Union the fleet's ~/.cache/parametric-nam/findpeak/ across machines: gather every host's
# cache files into this machine, then scatter the union back out, so every host ends up
# holding what any host has ever measured. Onset probing is embarrassingly parallel and
# content-addressed but was never shared -- see docs/fleet-deployment-proposal.md §4a and
# docs/per-item-sharding-proposal.md's closing section: three workers on the same device
# independently ran the same 25-corner coverage gate, ~50 minutes spent computing identical
# numbers, because nothing synced what any one of them had already measured.
#
# SAFE TO RUN ANY TIME, INCLUDING MID-RENDER. Every cache entry's filename is
# sha256(ONSET_METHOD + schx/module bytes + params + backend/oversample/min-start-v/
# solver-build/... extra)[:16].json -- see find_saturation_point.py's findpeak_cache_key().
# Two machines can never produce a DIFFERENT file under the SAME name: same content, or no
# file at all. Unioning by filename is therefore always correct, never a conflict to resolve
# -- rsync only ever ADDS files here, never overwrites one that already exists remotely or
# locally (--ignore-existing on every leg), so nothing already present is re-transferred or
# clobbered.
#
# PREREQUISITE (closed 2026-09-21, see that commit): findpeak_cache_key's `extra` string must
# include the renderer's own build version (solver_identity()) at every call site, or a
# rebuilt oracle's fresh measurements collide under the SAME filename as the old build's
# stale ones -- syncing would then spread one machine's wrong entry to the whole fleet. Ran
# on any parametric-nam checkout from before that fix, this script is not wrong, just less
# useful: entries that SHOULD collide (same solver) will merge correctly; entries that
# happened to share a name across different solver builds (the bug) would already have been
# silently colliding locally on each machine, same as before this script existed.
#
# Usage:
#   ./sync_findpeak_cache.sh --workers optiplex7010,blackbox
#
# Cross-machine reproducibility of the underlying renders is the other assumption here --
# see docs/per-item-sharding-proposal.md's "Determinism" section: bit-identical across five
# repeats on ONE machine (Mesa Orange, oversample 8), circumstantial but not yet
# full-precision-verified agreement across ARM/x86. If a rendering bug is ever traced to
# platform-specific float behavior, this script is exactly the mechanism that would have
# spread it fleet-wide -- keep that in mind before scheduling this unattended.
set -euo pipefail

WORKERS="" DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$WORKERS" ] || { echo "usage: $0 --workers host1,host2,... [--dry-run]" >&2; exit 2; }

IFS=',' read -r -a WORKER_ARR <<< "$WORKERS"

# Same keepalive convention as distribute_gen.sh -- an unattended sync across a fleet is
# exactly the shape of run that silently hangs on a dropped connection otherwise.
SSH_OPTS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=3)
export RSYNC_RSH="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

LOCAL_CACHE="$HOME/.cache/parametric-nam/findpeak"
mkdir -p "$LOCAL_CACHE"

RSYNC_FLAGS=(-az --ignore-existing)
[ "$DRY_RUN" -eq 1 ] && RSYNC_FLAGS+=(--dry-run -v)

before=$(find "$LOCAL_CACHE" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "==> local cache before: $before entries"

echo "==> gathering from ${#WORKER_ARR[@]} worker(s) into local cache ..."
for w in "${WORKER_ARR[@]}"; do
  # Discover the remote HOME so this works across the fleet's mixed /Users/chewie vs
  # /home/gene layout without hardcoding either -- same probe distribute_gen.sh uses.
  rhome="$(ssh "${SSH_OPTS[@]}" "$w" 'echo $HOME')"
  remote_cache="$rhome/.cache/parametric-nam/findpeak"
  n="$(ssh "${SSH_OPTS[@]}" "$w" "mkdir -p '$remote_cache' && find '$remote_cache' -name '*.json' 2>/dev/null | wc -l" | tr -d ' ')"
  echo "    $w: $n entries"
  rsync "${RSYNC_FLAGS[@]}" "$w:$remote_cache/" "$LOCAL_CACHE/" 2>&1 | sed "s/^/    [$w <- ] /" || true
done

after_gather=$(find "$LOCAL_CACHE" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "==> local cache after gather: $after_gather entries (+$((after_gather - before)))"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> --dry-run: skipping scatter phase"
  exit 0
fi

echo "==> scattering the union back out to every worker ..."
for w in "${WORKER_ARR[@]}"; do
  rhome="$(ssh "${SSH_OPTS[@]}" "$w" 'echo $HOME')"
  remote_cache="$rhome/.cache/parametric-nam/findpeak"
  rsync "${RSYNC_FLAGS[@]}" "$LOCAL_CACHE/" "$w:$remote_cache/" 2>&1 | sed "s/^/    [ -> $w] /" || true
done

echo "==> done. Every worker (and this machine) now holds the same $after_gather-entry union."
