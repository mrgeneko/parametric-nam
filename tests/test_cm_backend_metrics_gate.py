"""CmBackend._render_one must not return audio from a render cm_run itself flagged as untrustworthy."""
import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import render_backends  # noqa: E402

FAKE = """#!{py}
import json, sys
import numpy as np
import soundfile as sf
a = sys.argv
out = a[3]
metrics = a[a.index("--metrics") + 1]
sf.write(out, np.full(480, 0.25, dtype="float32"), 48000, subtype="FLOAT")
m = {"solves": 1000000, "unconverged": 0, "severe": 0, "divergences": 0, "dc_converged": 1, "first_bad_sample": -1}
m.update(json.load(open(sys.argv[0] + ".cfg")))
json.dump(m, open(metrics, "w"))
sys.exit(m.pop("rc", 0))
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "cm_run"
    exe.write_text(FAKE.replace("{py}", sys.executable, 1))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CM_RUN", str(exe))

    def configure(**kw):
        Path(str(exe) + ".cfg").write_text(json.dumps(kw))

    configure()
    return configure


def _render(tmp_path):
    be = render_backends.CmBackend(tmp_path / "amp.schx")
    return be._render_one({"Gain": 0.5}, str(tmp_path / "in.wav"), str(tmp_path), "t")


def test_clean_render_returns_audio(fake, tmp_path):
    y = _render(tmp_path)
    assert y is not None and len(y) == 480


@pytest.mark.parametrize("bad", [{"divergences": 3, "first_bad_sample": 77}, {"dc_converged": 0}, {"rc": 3}])
def test_untrustworthy_render_is_rejected(fake, tmp_path, capsys, bad):
    fake(**bad)
    assert _render(tmp_path) is None
    assert "[t]" in capsys.readouterr().err


def test_unconverged_fraction_warns_but_returns_audio(fake, tmp_path, capsys):
    fake(unconverged=100)
    assert _render(tmp_path) is not None
    assert "unconverged" in capsys.readouterr().err
