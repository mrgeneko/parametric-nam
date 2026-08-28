#!/usr/bin/env python3
"""Name-based knob role classification -- shared between preflight.py (direction/EQ-swamp
checks), scaffold_config.py ([knob-kind] auto-tagging), and gen_dataset_from_schx.py's own
knob-sensitivity check, so all three agree on what a knob's name implies without needing to
import preflight.py's heavier machinery (render_backends.py pulls in spicelib, which isn't a
dependency scaffold_config.py or gen_dataset_from_schx.py should need just to classify a
knob's role by name). Deliberately dependency-free: no numpy, no backend imports.

Categories: "hi"/"lo"/"mid" (an EQ/tone-shaping knob), "drive" (gain/distortion), "rms"
(volume/level). None ("unknown") means the name didn't match anything -- see classify()'s own
docstring for what that implies downstream.
"""

# (name keywords, kind, human label). First match wins; order specific->general.
DIRECTION_RULES = [
    (("treble", "tone", "bright", "presence", "high", "top", "tref"), "hi",  "treble/high-freq"),
    (("bass", "low", "depth", "body", "bottom", "sub"),               "lo",  "bass/low-freq"),
    (("mid", "middle"),                                               "mid", "midrange"),
    (("gain", "drive", "dist", "overdrive", "fuzz", "sustain", "sat", "od", "pre"), "drive", "gain/distortion"),
    (("volume", "level", "master", "output", "vol", "loud", "post"),  "rms", "output level"),
]


def classify(knob):
    """Returns (kind, label). kind is one of "hi"/"lo"/"mid"/"drive"/"rms", or None if the
    name matched nothing -- an unclassified knob gets NO special treatment anywhere that
    consults this: preflight.py checks responsiveness only (no direction check, no
    --eq-check-drive-level protection when ANOTHER knob is checked), and
    gen_dataset_from_schx.py's sensitivity check falls back to plain RMS-spread. A knob named
    unconventionally (or in another language) silently gets the least scrutiny, not the most
    -- always review scaffold_config.py's [knob-kind] guesses by hand, don't just trust them."""
    n = knob.lower()
    for keys, kind, label in DIRECTION_RULES:
        if any(k in n for k in keys):
            return kind, label
    return None, "unknown"


TONE_LIKE_KINDS = ("hi", "lo", "mid")
