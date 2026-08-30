#!/usr/bin/env bash
#
# Split ONE gen_dataset_from_schx.py generation across multiple machines over SSH, weighted
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
SYNC_FILES=()
DRY_RUN=0
GEN_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --workers)    WORKERS="$2"; shift 2 ;;
    --job)        JOB="$2"; shift 2 ;;
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
  info="$(ssh "$w" 'echo "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) $HOME"')"
  n="${info%% *}"; home="${info#* }"
  n="${n//[!0-9]/}"; [ -n "$n" ] || n=1
  CORES+=("$n")
  REMOTE_HOME+=("$home")
  TOTAL_CORES=$((TOTAL_CORES + n))
  echo "    $w: $n cores, \$HOME=$home"
done

# Weighted [LOW,HIGH] ranges out of TOTAL_CORES, cumulative -- worker i's share is proportional
# to its own core count, not an equal split. The whole [0, TOTAL_CORES) range is tiled with no
# gaps or overlaps by construction (each worker's LOW is the previous worker's HIGH+1), so every
# permutation index lands in exactly one worker's shard. LOW/HIGH/PIDS/REMOTE_OUT stay in
# lockstep with WORKER_ARR by POSITION (bash 3.2 has no associative arrays).
LOW=(); HIGH=(); REMOTE_OUT=()
cum=0
for i in "${!WORKER_ARR[@]}"; do
  LOW+=("$cum")
  cum=$((cum + CORES[i] - 1))
  HIGH+=("$cum")
  cum=$((cum + 1))
  REMOTE_OUT+=("${REMOTE_HOME[i]}/work/tmp/${JOB}_shard_${i}")
done

echo "==> shard plan (TOTAL=$TOTAL_CORES)"
for i in "${!WORKER_ARR[@]}"; do
  echo "    ${WORKER_ARR[$i]}: ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES  (${CORES[$i]}/$TOTAL_CORES of the grid)"
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> --dry-run: not touching any worker. Commands that would run:"
  for i in "${!WORKER_ARR[@]}"; do
    echo "    ssh ${WORKER_ARR[$i]} \"cd $REMOTE_DIR && $PY gen_dataset_from_schx.py $GEN_ARGS_Q --output ${REMOTE_OUT[$i]} --shard ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES\""
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
      ssh "$w" "mkdir -p $REMOTE_DIR/$dest_dir"
      rsync -az "$HERE/$f" "$w:$REMOTE_DIR/$f"
    done
  done
fi

echo "==> launching $N worker(s) in parallel (logs: $LOCAL_DIR/logs/<worker>.log)"
PIDS=()
for i in "${!WORKER_ARR[@]}"; do
  w="${WORKER_ARR[$i]}"
  ssh "$w" "cd $REMOTE_DIR && $PY gen_dataset_from_schx.py $GEN_ARGS_Q \
      --output ${REMOTE_OUT[$i]} --shard ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES" \
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
    echo "      ssh ${WORKER_ARR[$i]} \"cd $REMOTE_DIR && $PY gen_dataset_from_schx.py $GEN_ARGS_Q --output ${REMOTE_OUT[$i]} --shard ${LOW[$i]}-${HIGH[$i]}/$TOTAL_CORES\""
  done
  echo "    Not merging while any worker is outstanding -- a partial shard would just be reported"
  echo "    as a missing permutation by --combine anyway, so fix the worker and re-run this script"
  echo "    (already-done permutations resume-skip on every machine, nothing is re-rendered)."
  exit 1
fi

echo "==> merging $N shard(s) into $LOCAL_DIR"
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
  if ssh "$w" "[ -d ${REMOTE_OUT[$i]}/sig ]"; then
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
