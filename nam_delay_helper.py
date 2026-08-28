#!/usr/bin/env python3
"""Internal subprocess helper for gen_dataset_from_captures.py's detect_delay() -- not meant
to be run directly by users.

WHY THIS IS A SUBPROCESS, NOT A LAZY IMPORT.
nam.train.core depends on numba, which checks numpy's version AT IMPORT TIME. Importing
nam.train.core from inside parametric-nam's own already-running process (even after adding
the sibling neural-amp-modeler venv's site-packages to sys.path) hands numba whatever numpy
module is ALREADY LOADED in that process -- parametric-nam's own (2.5.0), not the sibling
venv's own compatible one (2.4.6) -- because Python caches modules by name in sys.modules,
and only the FIRST import of a given name in a process wins, regardless of which sys.path
entry a later import would have preferred. Confirmed hitting this directly: "ImportError:
Numba needs NumPy 2.4 or less. Got NumPy 2.5." A fresh subprocess, run under the sibling
venv's OWN interpreter, has no such contamination -- its numpy and numba were never touched
by parametric-nam's venv at all.

Usage: <sibling-venv-python> nam_delay_helper.py <dry_wav> <wet_wav>
Prints exactly one line of JSON to stdout and exits 0 -- including for an EXPECTED
"couldn't calibrate" outcome (not a standard file, no impulse response found, etc). A
non-JSON-parseable line or a nonzero exit means something is broken badly enough that the
caller's own fallback should apply, not this script's.

    {"delay": int, "source": "nam_standard_calibration"}  -- real calibration succeeded
    {"delay": null, "source": "..."}                      -- no calibration; "source" explains why
    {"delay": null, "source": "...", "detail": "..."}     -- with the causing exception's text
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")  # never open a blocking GUI window -- NAM's own calibrate_latency_v_all
                        # can pop one even with silent=True, when it can't find an impulse
                        # response and no manual fallback was given (manual_available=False
                        # here, always) -- confirmed hitting this directly, it hangs the
                        # subprocess waiting for a window nothing will ever close.


def _emit(delay, source=None, detail=None):
    payload = {"delay": delay}
    if source is not None:
        payload["source"] = source
    if detail is not None:
        payload["detail"] = detail
    print(json.dumps(payload))


def main():
    dry_wav, wet_wav = sys.argv[1], sys.argv[2]

    try:
        from nam.train.core import _detect_input_version, _InputValidationError, _analyze_latency
    except ImportError as e:
        _emit(None, "nam_unavailable", f"{type(e).__name__}: {e}")
        return

    try:
        version, _strong_match = _detect_input_version(dry_wav)
    except _InputValidationError:
        version = None
    except Exception as e:
        # Despite the (version, strong_match) signature implying a graceful None for "no
        # match", it can also RAISE (confirmed on a non-standard sweep) or hit an unrelated
        # read failure (e.g. a FLOAT-subtype WAV NAM's own reader doesn't support). Either
        # way: not a standard file, not this process's problem to solve further.
        _emit(None, "delay_zero_fallback", f"{type(e).__name__}: {e}")
        return
    if version is None:
        _emit(None, "delay_zero_fallback")
        return

    try:
        result = _analyze_latency(user_latency=None, input_version=version,
                                  input_path=dry_wav, output_path=wet_wav, silent=True,
                                  _override_suppress_plots=True)
        delay = result.calibration.recommended
    except Exception as e:
        _emit(None, "delay_zero_fallback", f"{type(e).__name__}: {e}")
        return

    if delay is None:
        _emit(None, "delay_zero_fallback")
        return
    _emit(int(delay), "nam_standard_calibration")


if __name__ == "__main__":
    main()
