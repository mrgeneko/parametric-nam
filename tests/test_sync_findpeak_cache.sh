#!/usr/bin/env bash
# sync_findpeak_cache.sh -- smoke tests via `ssh localhost` and --dry-run, never touching a
# real fleet or writing outside the local ~/.cache/parametric-nam/findpeak the script itself
# already owns (the script has no injectable cache-path override, and --dry-run's gather
# phase still `mkdir -p`s the real local cache dir -- harmless, it already exists on any
# machine that has ever run this tool or find_saturation_point.py).
#
# Uses `grep ... <<< "$out"` (here-string), never `echo "$out" | grep -q ...`: with a large
# captured $out (this tool's own verbose --dry-run output over a real, many-thousand-entry
# cache is 100+ KB), `grep -q` exits the instant it finds its match, closing its end of the
# pipe -- `echo`, still writing the rest of $out, gets SIGPIPE, and `pipefail` (set below)
# reports the PIPELINE's exit status as 141, not grep's real 0. That looked like a real
# assertion failure the first time this file was written; it was a test-harness bug, not a
# tool bug. A here-string has no live inter-process pipe to break this way.
#
# What this does NOT test: a genuine two-machine differential union (host A contributes
# entries host B didn't have). `ssh localhost` is the same filesystem as the invoking shell,
# so both "sides" of any local+remote comparison resolve to the identical path -- there is no
# way to fake two independently-seeded caches with only one real machine. That property was
# instead verified live against real fleet hosts when this script was built (see its own
# commit message: optiplex7010 + blackbox, 291 new entries merged into a 1671-entry union,
# confirmed identical on all three machines afterward). What's covered here is the CLI
# contract: argument parsing, comma-separated --workers, --dry-run never reaching the scatter
# phase, and the required-argument error path -- all of it exercisable without a second host.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fails=0
check() { if [ "$1" -eq 0 ]; then echo "  ok   $2"; else echo "  FAIL $2"; fails=$((fails+1)); fi; }

if ! ssh -o ConnectTimeout=3 -o BatchMode=yes localhost true >/dev/null 2>&1; then
  echo "SKIP: ssh localhost not available in this environment (needs passwordless self-SSH)"
  exit 0
fi

echo "== --workers is required =="
out="$("$HERE/sync_findpeak_cache.sh" 2>&1)"; rc=$?
[ "$rc" -eq 2 ]
check "$?" "missing --workers exits 2, not a crash"
grep -qi "usage:" <<< "$out"
check "$?" "missing --workers prints a usage message"

echo "== --dry-run never reaches the scatter phase =="
out="$("$HERE/sync_findpeak_cache.sh" --workers localhost --dry-run 2>&1)"; rc=$?
check "$rc" "--dry-run exits 0"
grep -q "gathering from 1 worker" <<< "$out"
check "$?" "single comma-separated worker parses to a 1-worker list"
grep -q "skipping scatter phase" <<< "$out"
check "$?" "--dry-run stops before scattering (never writes to a worker)"
if grep -q "scattering the union back out" <<< "$out"; then
  check 1 "--dry-run must not print the scatter banner"
else
  check 0 "--dry-run must not print the scatter banner"
fi

echo "== comma-separated multi-worker list parses correctly =="
out="$("$HERE/sync_findpeak_cache.sh" --workers localhost,localhost --dry-run 2>&1)"
grep -q "gathering from 2 worker" <<< "$out"
check "$?" "two comma-separated entries parse to a 2-worker list (localhost twice is fine -- it's just exercising IFS splitting)"

echo; [ "$fails" -eq 0 ] && echo "PASS" || { echo "$fails FAILURE(S)"; exit 1; }
