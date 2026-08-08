#!/usr/bin/env python3
"""Opt-in post-processing step: soft-limit a rendered (wet) training-target WAV so the
NAM trainer sees an explicit output ceiling near the top of its own training range, instead
of a purely linear relationship that a WaveNet then extrapolates unboundedly past the
training input peak. See internal engineering notes for why input-side calibration alone
doesn't fix this for a gain setting whose own real saturation point sits far above any
realistic input level (e.g. SLO-100 Normal Gain=0.1, ~20V input needed -- never reachable
in real use, so its whole training range stays linear with no hint of a ceiling).

This does NOT touch the dry input file -- only the rendered wet target. Not run by default;
invoke explicitly and point a data_config.json's y_path at the output.

Usage:
  python3 tools/apply_output_limiter.py IN.wav OUT.wav --threshold 0.35 --ceiling 0.6

Below --threshold, samples pass through unchanged. Above it, a smooth tanh knee compresses
toward an asymptote at --ceiling -- never a hard clip, so no new harmonic-distortion
artifact is added; it only caps how far growth continues at the loudest few percent of
samples (matching the [redacted] sweep's rare max-excitation segments, given its ~24:1 crest
factor -- most content sits far below the threshold and is untouched).
"""
import argparse
import math
import struct
import sys


def read_wav16(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 12
    sr = None
    audio = None
    while pos < len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            sr = fmt[2]
        elif cid == b"data":
            audio = body
        pos += 8 + size + (size % 2)
    n = len(audio) // 2
    vals = struct.unpack("<%dh" % n, audio[:n * 2])
    return sr, [v / 32768.0 for v in vals]


def write_wav16(path, sr, samples):
    clamped = [max(-1.0, min(1.0, v)) for v in samples]
    ints = [int(round(v * 32767.0)) for v in clamped]
    data = struct.pack("<%dh" % len(ints), *ints)
    fmt_chunk = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 4 + (8 + 16) + (8 + len(data))))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def soft_limit(x, threshold, ceiling):
    ax = abs(x)
    if ax <= threshold:
        return x
    span = ceiling - threshold
    y = threshold + span * math.tanh((ax - threshold) / span)
    return math.copysign(y, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("output_wav")
    ap.add_argument("--threshold", type=float, required=True,
                     help="normalized level (0-1) below which samples pass through unchanged")
    ap.add_argument("--ceiling", type=float, required=True,
                     help="normalized asymptote (0-1) that louder samples compress toward")
    args = ap.parse_args()
    if not (0 < args.threshold < args.ceiling <= 1.0):
        sys.exit("require 0 < threshold < ceiling <= 1.0")

    sr, y = read_wav16(args.input_wav)
    out = [soft_limit(v, args.threshold, args.ceiling) for v in y]

    touched = sum(1 for v in y if abs(v) > args.threshold)
    print("samples above threshold: %d / %d (%.2f%%)" % (touched, len(y), 100.0 * touched / len(y)))
    print("peak before=%.4f after=%.4f" % (max(abs(v) for v in y), max(abs(v) for v in out)))

    write_wav16(args.output_wav, sr, out)
    print("wrote", args.output_wav)


if __name__ == "__main__":
    main()
