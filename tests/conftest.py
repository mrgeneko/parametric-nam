"""Test fixtures / import shims.

The round-trip acceptance test needs the official `neural-amp-modeler` package,
but its full dependency tree (librosa→numba, pytorch-lightning, …) won't install
on Python 3.13 and is training-only. We install it inference-only:

    pip install neural-amp-modeler --no-deps
    pip install pydantic          # REAL — used in nam type annotations

and fabricate the heavy training-only third-party deps (plus `nam.train`, which
subclasses a Lightning base) via the meta-path finder below, so `nam.models`
imports for inference. If neither nam nor this shim is present, the round-trip
test skips (see importorskip).
"""
import sys, types, importlib.abc, importlib.machinery
from unittest.mock import MagicMock

_BLOCKED_TOP = {"librosa", "numba", "auraloss", "pytorch_lightning", "torchaudio",
                "wavio", "tqdm", "matplotlib", "onnx", "transformers", "sounddevice"}


def _blocked(name: str) -> bool:
    if name.split(".")[0] in _BLOCKED_TOP:
        return True
    return name == "nam.train" or name.startswith("nam.train.")


class _MockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Return a mock module for any name under a blocked prefix (any depth)."""
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self) if _blocked(name) else None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__dict__["__getattr__"] = lambda n: MagicMock()
        m.__path__ = []  # mark as package so submodule imports resolve
        return m

    def exec_module(self, module):
        pass


# Install once, before any `import nam` in the test session.
if not any(isinstance(f, _MockFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _MockFinder())
