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
    -- always review scaffold_config.py's [knob-kind] guesses by hand, don't just trust them.

    A knob's actual TYPE (drive/gain vs. volume/level) is checked as a SUFFIX before falling
    back to plain substring-anywhere matching, and wins even when a tone keyword also appears
    in the name: found via BrightVol on a dual-channel (Bright/Normal) tweed amp, which was
    getting classified "hi" (treble) purely because "bright" is a treble/presence keyword --
    but on this amp (and any similar vintage Fender-style dual-channel design), "Bright" names
    the CHANNEL, and BrightVol is that channel's volume control, same role as NormalVol. A
    prefix naming a channel/mode ("Bright", "Normal", "OD", "Red", "Or(ange)", ...) is common
    across this fleet's devices; a name ENDING in a level/gain word ("...Vol", "...Level",
    "...Gain") is a much stronger, position-anchored signal for the knob's real role than an
    incidental substring match anywhere else in the name. Confirmed against every knob name
    currently in parametric-devices/devices.toml: this reclassifies exactly two, both
    corrections in the same direction -- BrightVol (was "hi", now "rms") and OD Level/ODLevel
    (was "drive" via the "od" substring, now "rms", since "OD" is likewise a channel prefix
    here, not a role by itself) -- everything else is unaffected."""
    n = knob.lower()
    for keys, kind, label in DIRECTION_RULES:
        if kind in ("drive", "rms") and any(n.endswith(k) for k in keys):
            return kind, label
    for keys, kind, label in DIRECTION_RULES:
        if any(k in n for k in keys):
            return kind, label
    return None, "unknown"


TONE_LIKE_KINDS = ("hi", "lo", "mid")
