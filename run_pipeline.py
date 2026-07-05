#!/usr/bin/env python3
"""
Dataset generation → combine → training pipeline.

Streams real-time timestamped output from each step and logs everything
to <dataset-dir>/pipeline.log (or --log).

Steps auto-skip when their outputs already exist; use --force-generate
to wipe and regenerate, or --skip-*/--only-* for manual control.

Example — full Dumble clean pipeline:
    python run_pipeline.py \\
        --dataset-dir /Volumes/DATA/work/dumble_clean \\
        --nam-output   /Volumes/DATA/work/dumble_clean.param.nam \\
        --checkpoint-dir /Volumes/DATA/work/dumble_clean_ckpt \\
        --backend livespice \\
        --schx "$HOME/work/LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx" \\
        --knobs volume,mid,treble,middle,bass,clean_master \\
        --range volume=0.1,0.3,0.5,0.7,0.9 \\
        --range mid=0.0,1.0 \\
        --range treble=0.2,0.4,0.6,0.8 \\
        --range middle=0.2,0.4,0.6,0.8 \\
        --range bass=0.2,0.4,0.6,0.8 \\
        --range clean_master=0.1,0.3,0.5,0.7,0.9 \\
        --fixed-params "Rock=0" \\
        --speaker S1 \\
        --input /Volumes/DATA/work/sweep120s.wav \\
        --slimmable --mmap --epochs 200
"""

import argparse, os, platform, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE    = Path(__file__).resolve().parent
PYTHON  = sys.executable
BATCH   = HERE / "batch_harness.py"
TRAIN   = HERE / "param_train.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, fh):
    line = f"[{ts()}] {msg}\n"
    print(line, end="", flush=True)
    fh.write(line)
    fh.flush()


def section(title, fh):
    bar = "=" * 64
    msg = f"\n{bar}\n[{ts()}] {title}\n{bar}\n"
    print(msg, flush=True)
    fh.write(msg)
    fh.flush()


def stream_run(cmd, fh, label):
    """Run cmd, stream each output line with a timestamp to stdout and log."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log(f"CMD: {' '.join(str(c) for c in cmd)}", fh)
    fh.write("\n")
    fh.flush()

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        out = f"[{ts()}] {stripped}\n"
        print(out, end="", flush=True)
        fh.write(out)
        fh.flush()
    proc.wait()

    if proc.returncode != 0:
        log(f"ERROR: {label} exited with code {proc.returncode} — aborting pipeline.", fh)
        sys.exit(proc.returncode)

    log(f"{label} finished OK.", fh)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate dataset, combine, and train a parametric NAM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- pipeline control ---
    g = ap.add_argument_group("pipeline")
    g.add_argument("--dataset-dir",    required=True, type=Path,
                   help="Dataset directory (generation output / training input)")
    g.add_argument("--nam-output",     required=True, type=Path,
                   help="Output .param.nam path")
    g.add_argument("--checkpoint-dir", required=True, type=Path,
                   help="Directory for epoch checkpoints and metrics.csv")
    g.add_argument("--log",            type=Path, default=None,
                   help="Log file (default: <dataset-dir>/pipeline.log)")
    g.add_argument("--force-generate", action="store_true",
                   help="Re-run generation even if outputs.npy already exists")
    g.add_argument("--skip-generate",  action="store_true")
    g.add_argument("--skip-combine",   action="store_true")
    g.add_argument("--skip-train",     action="store_true")
    g.add_argument("--no-caffeinate",  action="store_true",
                   help="Disable caffeinate (macOS sleep prevention, on by default)")

    # --- generation (batch_harness.py) ---
    g = ap.add_argument_group("generation")
    g.add_argument("--backend",      choices=["cpp", "livespice"], default="livespice")
    g.add_argument("--schx",         type=Path, help="Path to .schx (livespice)")
    g.add_argument("--circuit",      help="Circuit name (cpp)")
    g.add_argument("--knobs",        help="Comma-separated knob names")
    g.add_argument("--input",        type=Path, help="Input sweep WAV")
    g.add_argument("--workers",      type=int,  default=os.cpu_count())
    g.add_argument("--values",       help="Default sweep values for all knobs")
    g.add_argument("--range",        action="append", metavar="KNOB=v1,v2,...",
                   dest="ranges",    help="Per-knob value list (repeatable)")
    g.add_argument("--gang",         action="append", metavar="KNOB=Name1,Name2,...")
    g.add_argument("--steps",        action="append", metavar="KNOB=N")
    g.add_argument("--fixed-params", help="Fixed k=v,... for every permutation")
    g.add_argument("--speaker",      help="Speaker name (e.g. S1)")

    # --- training (param_train.py) ---
    g = ap.add_argument_group("training")
    g.add_argument("--epochs",         type=int,   default=100)
    g.add_argument("--batch-size",     type=int,   default=16)
    g.add_argument("--lr",             type=float, default=3e-4)
    g.add_argument("--crop-len",       type=int,   default=44100)
    g.add_argument("--repeats",        type=int,   default=1)
    g.add_argument("--mrstft-weight",  type=float, default=0.1)
    g.add_argument("--channels",       type=int,   default=8)
    g.add_argument("--slimmable",      action="store_true")
    g.add_argument("--mmap",           action="store_true")
    g.add_argument("--resume",         type=Path,  default=None)
    g.add_argument("--device",         default="auto")
    g.add_argument("--seed",           type=int,   default=42)
    g.add_argument("--param-sensitivity", action="store_true")
    g.add_argument("--val-split",      type=float, default=0.1)

    args = ap.parse_args()

    dataset_dir  = args.dataset_dir
    outputs_npy  = dataset_dir / "outputs.npy"
    sig_dir      = dataset_dir / "sig"
    log_path     = args.log or (dataset_dir / "pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    use_caffeinate = (platform.system() == "Darwin") and not args.no_caffeinate

    def wrap(cmd):
        return (["caffeinate", "-i"] + cmd) if use_caffeinate else cmd

    with open(log_path, "a") as fh:
        t0 = time.time()
        header = (f"\n{'#'*64}\n"
                  f"Pipeline started {datetime.now().isoformat()}\n"
                  f"Dataset : {dataset_dir}\n"
                  f"NAM out : {args.nam_output}\n"
                  f"Ckpt dir: {args.checkpoint_dir}\n"
                  f"{'#'*64}\n")
        print(header, flush=True)
        fh.write(header)
        fh.flush()

        # ------------------------------------------------------------------
        # Step 1: Generate
        # ------------------------------------------------------------------
        run_generate = not args.skip_generate
        if run_generate and not args.force_generate:
            if outputs_npy.exists() and not sig_dir.exists():
                log("SKIP generate: outputs.npy exists and no sig/ dir found. "
                    "Use --force-generate to redo.", fh)
                run_generate = False

        if run_generate:
            section("STEP 1 / 3 — Dataset Generation", fh)
            gen_cmd = [
                PYTHON, BATCH,
                "--backend", args.backend,
                "--output",  dataset_dir,
                "--workers", args.workers,
            ]
            if args.schx:          gen_cmd += ["--schx",         args.schx]
            if args.circuit:       gen_cmd += ["--circuit",      args.circuit]
            if args.knobs:         gen_cmd += ["--knobs",        args.knobs]
            if args.input:         gen_cmd += ["--input",        args.input]
            if args.values:        gen_cmd += ["--values",       args.values]
            if args.fixed_params:  gen_cmd += ["--fixed-params", args.fixed_params]
            if args.speaker:       gen_cmd += ["--speaker",      args.speaker]
            for r in (args.ranges or []):
                gen_cmd += ["--range", r]
            for g in (args.gang or []):
                gen_cmd += ["--gang", g]
            for s in (args.steps or []):
                gen_cmd += ["--steps", s]
            stream_run(wrap(gen_cmd), fh, "Generation")

        # ------------------------------------------------------------------
        # Step 2: Combine
        # ------------------------------------------------------------------
        run_combine = not args.skip_combine
        if run_combine and not sig_dir.exists():
            if outputs_npy.exists():
                log("SKIP combine: sig/ dir gone and outputs.npy exists — already combined.", fh)
            else:
                log("ERROR: sig/ dir not found and no outputs.npy — "
                    "did generation complete?", fh)
                sys.exit(1)
            run_combine = False

        if run_combine:
            section("STEP 2 / 3 — Combine", fh)
            stream_run(wrap([PYTHON, BATCH, "--combine", dataset_dir]), fh, "Combine")

        # ------------------------------------------------------------------
        # Step 3: Train
        # ------------------------------------------------------------------
        if not args.skip_train:
            section("STEP 3 / 3 — Training", fh)
            train_cmd = [
                PYTHON, TRAIN,
                "--dataset",         dataset_dir,
                "--output",          args.nam_output,
                "--checkpoint-dir",  args.checkpoint_dir,
                "--epochs",          args.epochs,
                "--batch-size",      args.batch_size,
                "--lr",              args.lr,
                "--crop-len",        args.crop_len,
                "--repeats",         args.repeats,
                "--mrstft-weight",   args.mrstft_weight,
                "--val-split",       args.val_split,
                "--device",          args.device,
                "--seed",            args.seed,
            ]
            if args.slimmable:           train_cmd.append("--slimmable")
            elif args.channels != 8:     train_cmd += ["--channels", args.channels]
            if args.mmap:                train_cmd.append("--mmap")
            if args.resume:              train_cmd += ["--resume", args.resume]
            if args.param_sensitivity:   train_cmd.append("--param-sensitivity")
            stream_run(wrap(train_cmd), fh, "Training")

        elapsed = time.time() - t0
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        footer = f"\n{'#'*64}\nPipeline complete in {h}h {m}m {s}s\n{'#'*64}\n"
        print(footer, flush=True)
        fh.write(footer)


if __name__ == "__main__":
    main()
