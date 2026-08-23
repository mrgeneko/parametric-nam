"""Properties tools/check_input_headroom.py's read_peak_v() must have. It's a hand-rolled WAV
parser (no soundfile/scipy) because this tool needs to read the exact peak sample value under
this fleet's V0dBFS=1V convention across all three formats this fleet actually produces (16-bit
PCM, 24-bit PCM, 32-bit float) -- a wrong peak here silently changes whether check_input_headroom
warns that a device's training excitation never reaches its own saturation ceiling.

See tools/check_input_headroom.py.
"""
import struct

import pytest

from tools.check_input_headroom import read_peak_v


def _wav_bytes(audio_format, channels, sr, bits, data):
    byte_rate = sr * channels * (bits // 8)
    block_align = channels * (bits // 8)
    fmt_chunk = struct.pack("<HHIIHH", audio_format, channels, sr, byte_rate, block_align, bits)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data))
    return (b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
            + b"data" + struct.pack("<I", len(data)) + data)


def _int24_bytes(samples):
    return b"".join(struct.pack("<i", s)[:3] for s in samples)


class TestReadPeakVInt16:
    def test_peak_matches_the_loudest_sample_normalized_by_32768(self, tmp_path):
        data = struct.pack("<3h", 0, 16384, -32768)
        wav = tmp_path / "t.wav"
        wav.write_bytes(_wav_bytes(1, 1, 48000, 16, data))
        assert read_peak_v(str(wav)) == pytest.approx(1.0)

    def test_uses_absolute_value_of_the_most_negative_sample(self, tmp_path):
        data = struct.pack("<2h", 0, -16384)
        wav = tmp_path / "t.wav"
        wav.write_bytes(_wav_bytes(1, 1, 48000, 16, data))
        assert read_peak_v(str(wav)) == pytest.approx(16384 / 32768.0)


class TestReadPeakVFloat32:
    def test_peak_matches_the_loudest_sample_directly_no_rescaling(self, tmp_path):
        data = struct.pack("<3f", 0.0, 0.5, -0.75)
        wav = tmp_path / "t.wav"
        wav.write_bytes(_wav_bytes(3, 1, 48000, 32, data))
        assert read_peak_v(str(wav)) == pytest.approx(0.75)

    def test_a_float_wav_can_exceed_1v_and_is_not_clamped(self, tmp_path):
        """This fleet's V0dBFS=1V convention means a float WAV can genuinely carry >1V samples
        (a hot excitation segment) -- read_peak_v must report that value as-is."""
        data = struct.pack("<2f", 0.0, 3.5)
        wav = tmp_path / "t.wav"
        wav.write_bytes(_wav_bytes(3, 1, 48000, 32, data))
        assert read_peak_v(str(wav)) == pytest.approx(3.5)


class TestReadPeakVInt24:
    def test_peak_matches_the_loudest_sample_normalized_by_2_pow_23(self, tmp_path):
        data = _int24_bytes([0, 4194304, -8388608])
        wav = tmp_path / "t.wav"
        wav.write_bytes(_wav_bytes(1, 1, 48000, 24, data))
        assert read_peak_v(str(wav)) == pytest.approx(1.0)


class TestReadPeakVUnsupportedFormat:
    def test_unhandled_format_raises_instead_of_silently_misreading(self, tmp_path):
        # audio_format=6 (A-law), 8 bits -- not one of this fleet's three real formats.
        data = bytes([0, 128, 255])
        wav = tmp_path / "t.wav"
        wav.write_bytes(_wav_bytes(6, 1, 48000, 8, data))
        with pytest.raises(ValueError):
            read_peak_v(str(wav))
