#!/usr/bin/env bash
#
# Split ONE dataset generation across multiple machines over SSH, weighted
# by each machine's own core count, and merge the results back into one local directory ready
# for --combine. This is orchestration only -- it does not know or care what circuit/knobs/
# backend it's rendering, since every gen_dataset_from_schx.py flag after `--` is forwarded
# to each worker verbatim, unmodified, with only --output and --shard appended per worker.
#
# Written for bash 3.2 (macOS's stock /bin/bash, which release_run.sh also targets) --
# no associative arrays, only indexed arrays kept in lockstep by position.
#
# WHY WEIGHTED, NOT AN EQUAL N-WAY SPLIT: a naive equal split makes total wall-clock time
# equal to the SLOWEST worker's shard time, wasting a fast machine's idle time after it
# finishes early. Probing each worker's own `nproc` (Linux) / `sysctl -n hw.ncpu` (macOS) and
# sizing shards proportionally is a heuristic, not a guarantee -- it doesn't account for CPU
# generation, thermal throttling, or a backend binary performing differently across
# architectures -- but it is a real improvement over equal split for a known-heterogeneous
# fleet, at the cost of only one extra SSH round-trip per worker before rendering starts.
#
# WHY EACH WORKER GETS ITS OWN --output, NEVER A SHARED LIVE DIRECTORY: gen_dataset_from_schx.py's
# acquire_generation_lock() uses fcntl.flock, which is reliable between processes on ONE machine
# but not across machines over a network mount. Two workers writing into one shared --output
# at once corrupts it silently (params.csv desyncs from the .npy files with nothing catching it
# until much later) -- see that function's own docstring. So results are always merged AFTER
# each worker finishes into its own private directory, never written concurrently.
#
# Usage:
#   ./distribute_gen.sh --workers host1,host2,host3 --job large_muffin_ds \
#       --sync-file examples/large_muffin/large_muffin.schx \
#       --sync-file examples/T3K-sweep-v3.wav \
#       -- \
#       --backend livespice --schx examples/large_muffin/large_muffin.schx \
#       --knobs SUSTAIN,TONE \
#       --range SUSTAIN=0.001,0.008,0.03,0.12,0.4,0.55,0.7,0.8,0.88,0.94,0.97,1.0 \
#       --range TONE=0.0,0.25,0.5,0.75,1.0 \
#       --fixed-params VOLUME=1.0 --oversample 16 \
#       --input examples/T3K-sweep-v3.wav
#
# Then, once it reports success:
#   python gen_dataset_from_schx.py --combine ~/work/tmp/large_muffin_ds
#
# --sync-file PATH   repeatable. Any input the render needs that isn't already on every worker
#                    (the .schx, the excitation wav, a --pedal-dir module for ngspice-deck/
#                    ltspice-deck) -- pushed to $REMOTE_DIR on every worker before rendering
#                    starts. Provisioning (repo clone, venv, oracle build) is a one-time,
#                    per-machine setup step this script deliberately does NOT automate --
#                    run setup.sh on each worker yourself first.
#
# Override via env: REMOTE_DIR (repo path on every worker, default ~/work/parametric-nam),
#                    LOCAL_DIR (merged output on this machine, default ~/work/tmp/$JOB),
#                    PY (python invocation on the worker, default ". .venv/bin/activate && python")
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKERS=""
JOB=""
RESERVE_CORES=0   # --reserve-cores N: give each worker (its cores - N) --parallel-sims instead
# of one uniform value chosen for the smallest machine. CAVEAT: --parallel-sims caps concurrent
# PROCESSES, not cores -- LTspice threads internally (measured: 10 sims on a 12-core M4 Pro ran
# at load 18.9), so "reserve N" eases pressure, it does not idle N cores.
WEIGHTS=""        # --weights a,b,c: split the grid by THESE numbers instead of probed core
# counts. Core count is a poor proxy for throughput on a heterogeneous fleet -- measured on this
# one, a 10-core fanless MacBook Air rendered 9.0 caps/h against 17.9 for a 12-core M4 Pro mini
# and 17.9 for a 14-core M3 Pro, i.e. HALF the rate its core count implied. Weighting by cores
# gave it a share it needed 20.1 h to finish while the others idled from 11.6 h. Render a small
# grid first, read caps/h per worker, then pass those numbers here.
RENDERER="gen_dataset_from_schx.py"   # --renderer: which entry point each worker runs.
# The deck renderers (render_ltspice_deck.py / render_ngspice_deck.py) take the SAME
# --shard LOW-HIGH/TOTAL contract (shard.py is shared), but name their output dir
# --outdir, not --output, and merge differently: concatenate cap_NNNN.wav plus the
# manifest.jsonl lines and mapping.csv bodies, then run gen_dataset_from_captures.py
# against the merged directory instead of gen_dataset_from_schx.py --combine.
SYNC_FILES=()
DRY_RUN=0
GEN_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --workers)    WORKERS="$2"; shift 2 ;;
    --job)        JOB="$2"; shift 2 ;;
    --renderer)  RENDERER="$2"; shift 2 ;;
    --reserve-cores) RESERVE_CORES="$2"; shift 2 ;;
    --weights)   WEIGHTS="$2"; shift 2 ;;
    --sync-file)  SYNC_FILES+=("$2"); shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --)           shift; GEN_ARGS=("$@"); break ;;
    *)            echo "unknown flag: $1 (generation args must come after --)" >&2; exit 1 ;;
  esac
done

[ -n "$WORKERS" ]           || { echo "ERROR: --workers host1,host2,... is required" >&2; exit 1; }
[ -n "$JOB" ]               || { echo "ERROR: --job NAME is required (names the shard/output dirs)" >&2; exit 1; }
[ "${#GEN_ARGS[@]}" -gt 0 ] || { echo "ERROR: no generation args given after --" >&2; exit 1; }

# %q-quote each forwarded arg individually before joining -- a plain "${GEN_ARGS[*]}" join loses
# quoting entirely, so any value containing a space (e.g. --gear-model "My Cool Amp") would be
# split back into two separate argv entries on the remote end. printf %q survives the round trip
# through the remote shell that `ssh host "command string"` implies.
GEN_ARGS_Q=""
for a in "${GEN_ARGS[@]}"; do
  printf -v _q '%q' "$a"
  GEN_ARGS_Q="$GEN_ARGS_Q $_q"
done

IFS=',' read -r -a WORKER_ARR <<< "$WORKERS"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/work/parametric-nam}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/work/tmp/$JOB}"
PY="${PY:-. .venv/bin/activate && python}"

# A silent network hang (machine sleeps, path black-holes packets -- NOT a clean disconnect,
# which ssh already detects and exits nonzero for on its own) has no default timeout: an
# already-established ssh connection that just goes quiet blocks `wait` on that one PID
# forever, and since workers are checked in order, even a LATER worker that already finished
# successfully never gets reported until the stuck one ahead of it resolves. ServerAlive*
# makes the ssh client itself probe the connection and give up after ~90s of silence,
# turning a silent hang into the same cleanly-detected failure the FAILED_IDX/wait logic
# below already handles correctly. Applies to both direct ssh calls (SSH_OPTS) and every
# rsync transfer, which uses ssh as its transport too (RSYNC_RSH, which rsync reads
# automatically -- no per-call -e flag needed).
SSH_OPTS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=3)
export RSYNC_RSH="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

N=${#WORKER_ARR[@]}

echo "==> probing core count and \$HOME on $N worker(s)"
CORES=(); REMOTE_HOME=()
TOTAL_CORES=0
for w in "${WORKER_ARR[@]}"; do
  # One round trip for both: resolving $HOME explicitly HERE (rather than passing a literal
  # "\$HOME" string into a later rsync host:path argument) avoids depending on whether rsync's
  # own remote-shell invocation expands it the same way a plain `ssh host "command"` does --
  # that's true for ssh command strings (used below for REMOTE_DIR), but is a murkier assumption
  # for rsync's host:path argument specifically, and this script needs the pull-back paths right
  # every time, not "usually."
  info="$(ssh "${SSH_OPTS[@]}" "$w" 'echo "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) $HOME"')"
  n="${info%% *}"; home="${info#* }"
  n="${n//[!0-9]/}"; [ -n "$n" ] || n=1
  CORES+=("$n")
  REMOTE_HOME+=("$home")
  TOTAL_CORES=$((TOTAL_CORES + n))
  echo "    $w: $n cores, \$HOME=$home"
done

# --weights overrides the probed core counts for SPLITTING only. CORES[] is left alone so
# --reserve-cores still sizes --parallel-sims from what the machine actually has.
SPLIT=("${CORES[@]}")
if [ -n "$WEIGHTS" ]; then
  IFS=',' read -r -a SPLIT <<< "$WEIGHTS"
  [ "${#SPLIT[@]}" -eq "${#WORKER_ARR[@]}" ] || {
    echo "ERROR: --weights has ${#SPLIT[@]} value(s) but there are ${#WORKER_ARR[@]} worker(s)" >&2; exit 2; }
  TOTAL_CORES=0
  for _w in "${SPLIT[@]}"; do
    case "$_w" in ''|*[!0-9]*) echo "ERROR: --weights must be positive integers (got '$_w')" >&2; exit 2;; esac
    [ "$_w" -ge 1 ] || { echo "ERROR: --weights values must be >= 1 (got '$_w')" >&2; exit 2; }
    TOTAL_CORES=$((TOTAL_CORES + _w))
  done
  echo "==> --weights: splitting by $WEIGHTS (total $TOTAL_CORES) instead of probed core counts"
fi

# Weighted [LOW,HIGH] ranges out of TOTAL_CORES, cumulative -- worker i's share is proportional
# to its own core count, not an equal split. The whole [0, TOTAL_CORES) range is tiled with no
# gaps or overlaps by construction (each worker's LOW is the previous worker's HIGH+1), so every
# permutation index lands in exactly one worker's shard. LOW/HIGH/PIDS/REMOTE_OUT stay in
# lockstep with WORKER_ARR by POSITION (bash 3.2 has no associative arrays).
LOW=(); HIGH=(); REMOTE_OUT=(); PSIMS=()
cum=0
for i in "${!WORKER_ARR[@]}"; do
  LOW+=("$cum")
  cum=$((cum + SPLIT[i] - 1))
  HIGH+=("$cum")
  cum=$((cum + 1))
  REMOTE_OUT+=("${REMOTE_HOME[i]}/work/tmp/${JOB}_shard_${i}")
  if [ "$RESERVE_CORES" -gt 0 ]; then
    _n=$(( CORES[i] - RESERVE_CORES )); [ "$_n" -lt 1 ] && _n=1
    PSIMS+=(" --parallel-sims $_n")
  else
    PSIMS+=("")
  fi
done

case "$RENDERER" in
  *_deck.py) OUTFLAG="--outdir";;
  *)         OUTFLAG="--output";;
esac

echo "==> shard plan (TOTAL=$TOTAL_CORES)"
for i in "${!WORKER_ARR[@]}"; do
  echo "    ${WORKER_ARR[$i]}: ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES  (${SPLIT[$i]}/$TOTAL_CORES of the grid)"
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> --dry-run: not touching any worker. Commands that would run:"
  for i in "${!WORKER_ARR[@]}"; do
    echo "    ssh ${SSH_OPTS[*]} ${WORKER_ARR[$i]} \"cd $REMOTE_DIR && $PY $RENDERER $GEN_ARGS_Q $OUTFLAG ${REMOTE_OUT[$i]} --shard ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES${PSIMS[$i]}\""
  done
  exit 0
fi

mkdir -p "$LOCAL_DIR/logs"

if [ "${#SYNC_FILES[@]}" -gt 0 ]; then
  echo "==> pushing ${#SYNC_FILES[@]} input file(s) to every worker"
  for w in "${WORKER_ARR[@]}"; do
    for f in "${SYNC_FILES[@]}"; do
      # preserve the relative directory structure under REMOTE_DIR so a --schx/--input path
      # that's relative to the repo root resolves the same way on the worker as it does here.
      dest_dir="$(dirname "$f")"
      ssh "${SSH_OPTS[@]}" "$w" "mkdir -p $REMOTE_DIR/$dest_dir"
      rsync -az "$HERE/$f" "$w:$REMOTE_DIR/$f"
    done
  done
fi

echo "==> launching $N worker(s) in parallel (logs: $LOCAL_DIR/logs/<worker>.log)"
PIDS=()
for i in "${!WORKER_ARR[@]}"; do
  w="${WORKER_ARR[$i]}"
  ssh "${SSH_OPTS[@]}" "$w" "cd $REMOTE_DIR && $PY $RENDERER $GEN_ARGS_Q \
      $OUTFLAG ${REMOTE_OUT[$i]} --shard ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES${PSIMS[$i]}" \
      > "$LOCAL_DIR/logs/$w.log" 2>&1 &
  PIDS+=("$!")
  echo "    $w: pid $!"
done

FAILED_IDX=()
for i in "${!WORKER_ARR[@]}"; do
  w="${WORKER_ARR[$i]}"
  if wait "${PIDS[$i]}"; then
    echo "    $w: OK"
  else
    echo "    $w: FAILED (see $LOCAL_DIR/logs/$w.log)"
    FAILED_IDX+=("$i")
  fi
done

if [ "${#FAILED_IDX[@]}" -gt 0 ]; then
  echo "==> ${#FAILED_IDX[@]} worker(s) failed"
  echo "    Re-run just a failed worker's shard once fixed (resume picks up where it left off):"
  for i in "${FAILED_IDX[@]}"; do
    echo "      ssh ${SSH_OPTS[*]} ${WORKER_ARR[$i]} \"cd $REMOTE_DIR && $PY $RENDERER $GEN_ARGS_Q $OUTFLAG ${REMOTE_OUT[$i]} --shard ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES${PSIMS[$i]}\""
  done
  echo "    Not merging while any worker is outstanding -- a partial shard would just be reported"
  echo "    as a missing permutation by --combine anyway, so fix the worker and re-run this script"
  echo "    (already-done permutations resume-skip on every machine, nothing is re-rendered)."
  exit 1
fi

echo "==> merging $N shard(s) into $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

if [ "$OUTFLAG" = "--outdir" ]; then
  # ---- DECK renderers (render_ltspice_deck.py / render_ngspice_deck.py) ----------------
  # Different artifacts from the schx path: cap_NNNN.wav files plus manifest.jsonl and
  # mapping.csv, and no config.json. cap_NNNN keeps the GLOBAL grid index (shard.select
  # filters on it rather than renumbering), so merging the wavs is filename-safe exactly the
  # way merging sig/ is. manifest.jsonl is headerless -- concatenate every line. mapping.csv
  # HAS a header, so take it once and append only body rows, or the merged file gets a header
  # buried mid-table and gen_dataset_from_captures.py reads it as a permutation.
  first=1
  for i in "${!WORKER_ARR[@]}"; do
    w="${WORKER_ARR[$i]}"
    if ! ssh "${SSH_OPTS[@]}" "$w" "[ -f ${REMOTE_OUT[$i]}/mapping.csv ]"; then
      echo "    $w: shard was empty (0 permutations in its range) -- nothing to merge"
      continue
    fi
    rsync -az --include='cap_*.wav' --exclude='*' "$w:${REMOTE_OUT[$i]}/" "$LOCAL_DIR/"
    rsync -az "$w:${REMOTE_OUT[$i]}/manifest.jsonl" "$LOCAL_DIR/.shard_manifest.jsonl"
    cat "$LOCAL_DIR/.shard_manifest.jsonl" >> "$LOCAL_DIR/manifest.jsonl"
    rm -f "$LOCAL_DIR/.shard_manifest.jsonl"
    rsync -az "$w:${REMOTE_OUT[$i]}/mapping.csv" "$LOCAL_DIR/.shard_mapping.csv"
    if [ "$first" -eq 1 ]; then
      cp "$LOCAL_DIR/.shard_mapping.csv" "$LOCAL_DIR/mapping.csv"; first=0
    else
      tail -n +2 "$LOCAL_DIR/.shard_mapping.csv" >> "$LOCAL_DIR/mapping.csv"
    fi
    rm -f "$LOCAL_DIR/.shard_mapping.csv"
  done
  n_wav=$(ls "$LOCAL_DIR"/cap_*.wav 2>/dev/null | wc -l | tr -d ' ')
  n_man=$(wc -l < "$LOCAL_DIR/manifest.jsonl" 2>/dev/null | tr -d ' ')
  echo "==> merged: $n_wav capture(s), $n_man manifest row(s) in $LOCAL_DIR"
  echo "==> done. Next:  python gen_dataset_from_captures.py --captures $LOCAL_DIR/'*.wav' \\"
  echo "                     --mapping-csv $LOCAL_DIR/mapping.csv --input <excitation.wav> \\"
  echo "                     --output <dataset-dir> --v0dbfs 1.0"
else
  # ---- schx path (gen_dataset_from_schx.py) --------------------------------------------
  mkdir -p "$LOCAL_DIR/sig"
  first=1
  for i in "${!WORKER_ARR[@]}"; do
    w="${WORKER_ARR[$i]}"
    # sig/ filenames are the GLOBAL permutation index (not shard-relative), so merging every
    # shard's sig/ tree into one directory is filename-safe by construction -- disjoint shards
    # never produce the same filename twice.
    #
    # A worker whose weighted [LOW,HIGH] range happened to match none of this job's permutation
    # indices (small grid, fine-grained core-weighted split) legitimately renders zero files and
    # never creates sig/ at all -- that's a valid empty shard, not a failure, so check first
    # rather than letting rsync error out on a missing remote directory.
    if ssh "${SSH_OPTS[@]}" "$w" "[ -d ${REMOTE_OUT[$i]}/sig ]"; then
      rsync -az "$w:${REMOTE_OUT[$i]}/sig/" "$LOCAL_DIR/sig/"
    else
      echo "    $w: shard was empty (0 permutations in its range) -- nothing to merge"
    fi
    if [ "$first" -eq 1 ]; then
      rsync -az "$w:${REMOTE_OUT[$i]}/params.csv" "$LOCAL_DIR/params.csv"
      rsync -az "$w:${REMOTE_OUT[$i]}/config.json" "$LOCAL_DIR/config.json"
      first=0
    else
      # params.csv must be CONCATENATED, never overwritten -- every shard has its own file with
      # the same name and a full header, so appending body rows (skip line 1) is what makes this
      # a merge instead of a last-writer-wins clobber that silently drops every earlier shard.
      rsync -az "$w:${REMOTE_OUT[$i]}/params.csv" "$LOCAL_DIR/.shard_params.csv"
      tail -n +2 "$LOCAL_DIR/.shard_params.csv" >> "$LOCAL_DIR/params.csv"
      rm -f "$LOCAL_DIR/.shard_params.csv"
    fi
  done
  echo "==> done. Next:  python gen_dataset_from_schx.py --combine $LOCAL_DIR"
fi
