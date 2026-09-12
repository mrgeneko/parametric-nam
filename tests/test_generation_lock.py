"""The generation lock must survive deletion of the directory it protects.

flock is held on an INODE. When the lock file lived at out_dir/.generation.lock, `rm -rf
out_dir` unlinked it without releasing anything, and the next process created a fresh inode
at the same path and locked it happily -- two generations into one directory, silently
desyncing params.csv from the .npy files. That is not hypothetical: it happened during the
Mesa Orange gain/master shard run on 2026-09-12, when an orphaned renderer held the lock and
a `rm -rf shard1` aimed at clearing the shard for a restart unlinked it out from under itself.
"""
import shutil
import pytest
from pathlib import Path

from gen_dataset_from_schx import acquire_generation_lock


class TestGenerationLockSurvivesOutputDeletion:

    def test_second_acquire_on_the_same_dir_is_refused(self, tmp_path):
        out = tmp_path / "shard0"; out.mkdir()
        held = acquire_generation_lock(out)
        with pytest.raises(SystemExit) as e:
            acquire_generation_lock(out)
        assert "already generating" in str(e.value)
        held.close()

    def test_deleting_out_dir_does_not_release_the_lock(self, tmp_path):
        """THE REGRESSION. Clearing a shard for a restart must not silently hand the lock to
        a second generation while the first is still alive."""
        out = tmp_path / "shard1"; out.mkdir()
        held = acquire_generation_lock(out)
        shutil.rmtree(out)                      # exactly what broke it in the field
        assert not out.exists()
        with pytest.raises(SystemExit) as e:
            acquire_generation_lock(out)        # would have SUCCEEDED before the fix
        assert "already generating" in str(e.value)
        held.close()

    def test_lock_file_lives_outside_out_dir(self, tmp_path):
        out = tmp_path / "shard2"; out.mkdir()
        held = acquire_generation_lock(out)
        lock_dir = Path.home() / ".cache" / "parametric-nam" / "locks"
        holders = [p for p in lock_dir.glob("*.lock") if str(out.resolve()) in p.read_text()]
        assert len(holders) == 1, "lock not found outside out_dir"
        assert lock_dir not in out.parents and out not in holders[0].parents
        held.close()

    def test_breadcrumb_points_at_the_real_lock(self, tmp_path):
        out = tmp_path / "shard3"; out.mkdir()
        held = acquire_generation_lock(out)
        info = (out / ".generation.lock.info").read_text()
        assert "lock=" in info and "pid=" in info
        # Breadcrumb is informational ONLY -- deleting it must not free the lock.
        (out / ".generation.lock.info").unlink()
        with pytest.raises(SystemExit):
            acquire_generation_lock(out)
        held.close()

    def test_releases_normally_when_the_holder_closes(self, tmp_path):
        out = tmp_path / "shard4"; out.mkdir()
        acquire_generation_lock(out).close()
        acquire_generation_lock(out).close()     # a finished run must not block the next one

    def test_different_dirs_do_not_contend(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
        ha = acquire_generation_lock(a)
        hb = acquire_generation_lock(b)          # distinct outputs are independent
        ha.close(); hb.close()
