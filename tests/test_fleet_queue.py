import json
import threading

import pytest

from fleet_queue import FleetQueue


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def q(tmp_path, clock):
    fq = FleetQueue(tmp_path / "q.db", clock=clock, lease_s=100, quarantine_after=3)
    yield fq
    fq.close()


PI = {"per_item": True}


def specs(n):
    return [f"{i}-{i}/{n}" for i in range(n)]


def reg(q, *names):
    for n in names:
        q.register_worker(n, f"/work/{n}", 2)


class TestLease:
    def test_fifo_and_exhaustion(self, q):
        reg(q, "a")
        q.submit_job({"tool": "x", "per_item": True}, specs(2))
        assert q.lease("a", 0)["spec"] == "0-0/2"
        assert q.lease("a", 1)["spec"] == "1-1/2"
        assert q.lease("a", 2) is None

    def test_unregistered_worker_rejected(self, q):
        q.submit_job(PI, specs(1))
        with pytest.raises(KeyError):
            q.lease("ghost")

    def test_lease_carries_job_spec_and_attempt(self, q):
        reg(q, "a")
        jid = q.submit_job({"tool": "gen_dataset", "output": "o"}, specs(1))
        got = q.lease("a")
        assert got["job_id"] == jid and got["job"]["output"] == "o" and got["attempt"] == 1

    def test_jobs_served_in_submission_order(self, q):
        reg(q, "a")
        j1 = q.submit_job(PI, ["0-0/1"])
        j2 = q.submit_job(PI, ["0-0/1"])
        assert q.lease("a", 0)["job_id"] == j1
        assert q.lease("a", 1)["job_id"] == j2

    def test_two_workers_never_get_same_chunk(self, q):
        reg(q, "a", "b")
        q.submit_job(PI, specs(2))
        x, y = q.lease("a"), q.lease("b")
        assert x["chunk_id"] != y["chunk_id"]

    def test_concurrent_leases_are_unique(self, tmp_path):
        fq = FleetQueue(tmp_path / "c.db", lease_s=100)
        for i in range(8):
            fq.register_worker(f"w{i}", "/d", 1)
        fq.submit_job(PI, specs(40))
        got, lk = [], threading.Lock()

        def run(name):
            while True:
                r = fq.lease(name, 0)
                if r is None:
                    return
                with lk:
                    got.append(r["chunk_id"])
                fq.complete(r["chunk_id"], name)
        ts = [threading.Thread(target=run, args=(f"w{i}",)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert sorted(got) == sorted(set(got)) and len(got) == 40
        fq.close()


class TestSlots:
    def test_whole_chunk_job_only_goes_to_slot_zero(self, q):
        reg(q, "a")
        q.submit_job({"per_item": False}, specs(2))
        assert q.lease("a", 1) is None
        assert q.lease("a", 0) is not None

    def test_per_item_job_goes_to_any_slot(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1))
        assert q.lease("a", 3) is not None

    def test_slot_one_skips_whole_chunk_job_for_later_per_item_job(self, q):
        reg(q, "a")
        q.submit_job({"per_item": False}, specs(1))
        q.submit_job(PI, specs(1))
        assert q.lease("a", 1)["job_id"] == 2


class TestRetry:
    def test_failed_chunk_goes_to_different_worker(self, q):
        reg(q, "a", "b")
        q.submit_job(PI, specs(2), retries=1)
        c0 = q.lease("a")
        assert q.fail(c0["chunk_id"], "a", "boom") == "requeued"
        # a is offered chunk 1 (untried) before the one it failed
        assert q.lease("a")["spec"] == "1-1/2"
        assert q.lease("b")["spec"] == "0-0/2"

    def test_falls_back_to_tried_chunk_when_nothing_else(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1), retries=1)
        c = q.lease("a")
        q.fail(c["chunk_id"], "a", "x")
        again = q.lease("a")
        assert again["chunk_id"] == c["chunk_id"] and again["attempt"] == 2

    def test_exhausted_retries_fail_for_good(self, q):
        reg(q, "a", "b")
        jid = q.submit_job(PI, specs(1), retries=1)
        c = q.lease("a")
        assert q.fail(c["chunk_id"], "a", "e1") == "requeued"
        c = q.lease("b")
        assert q.fail(c["chunk_id"], "b", "e2") == "failed"
        assert q.job_counts(jid)["failed"] == 1 and q.job_finished(jid)

    def test_retries_zero_never_requeues(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1), retries=0)
        assert q.fail(q.lease("a")["chunk_id"], "a") == "failed"

    def test_fail_from_non_holder_ignored(self, q):
        reg(q, "a", "b")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        assert q.fail(c["chunk_id"], "b", "not mine") == "ignored"
        assert q.job_counts(1)["leased"] == 1

    def test_error_tail_kept(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1), retries=0)
        c = q.lease("a")
        q.fail(c["chunk_id"], "a", "x" * 5000 + "TAIL")
        err = q.status()["problems"][0]["error"]
        assert err.endswith("TAIL") and len(err) == 2000


class TestQuarantine:
    def test_benched_after_consecutive_failures_with_no_success(self, q):
        reg(q, "bad", "good")
        q.submit_job(PI, specs(10), retries=5)
        for _ in range(3):
            c = q.lease("bad")
            q.fail(c["chunk_id"], "bad", "import error")
        assert q.lease("bad") is None
        assert q.lease("good") is not None
        w = {x["name"]: x for x in q.status()["workers"]}
        assert w["bad"]["quarantined"] and not w["good"]["quarantined"]

    def test_success_resets_and_prevents_quarantine(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(10), retries=9)
        q.fail(q.lease("a")["chunk_id"], "a")
        q.fail(q.lease("a")["chunk_id"], "a")
        q.complete(q.lease("a")["chunk_id"], "a")
        for _ in range(5):
            q.fail(q.lease("a")["chunk_id"], "a")
        assert q.lease("a") is not None       # has successes -> never benched

    def test_two_failures_do_not_quarantine(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(5), retries=5)
        q.fail(q.lease("a")["chunk_id"], "a")
        q.fail(q.lease("a")["chunk_id"], "a")
        assert q.lease("a") is not None

    def test_unquarantine(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(10), retries=9)
        for _ in range(3):
            q.fail(q.lease("a")["chunk_id"], "a")
        q.unquarantine("a")
        assert q.lease("a") is not None
        with pytest.raises(KeyError):
            q.unquarantine("nobody")

    def test_disabled_with_zero(self, tmp_path):
        fq = FleetQueue(tmp_path / "z.db", quarantine_after=0)
        fq.register_worker("a", "/d", 1)
        fq.submit_job(PI, specs(10), retries=9)
        for _ in range(6):
            fq.fail(fq.lease("a")["chunk_id"], "a")
        assert fq.lease("a") is not None
        fq.close()


class TestLeaseExpiry:
    def test_expired_lease_requeues_and_is_reassigned(self, q, clock):
        reg(q, "a", "b")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        clock.t += 101
        got = q.lease("b")
        assert got["chunk_id"] == c["chunk_id"] and got["attempt"] == 2

    def test_heartbeat_extends_lease(self, q, clock):
        reg(q, "a", "b")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        clock.t += 90
        assert q.heartbeat(c["chunk_id"], "a", 3)["ok"]
        clock.t += 90
        assert q.lease("b") is None            # 180s elapsed but renewed at 90
        clock.t += 20
        assert q.lease("b") is not None

    def test_expiry_does_not_blame_worker(self, q, clock):
        reg(q, "a")
        q.submit_job(PI, specs(10), retries=9)
        for _ in range(4):
            q.lease("a", 0)
            clock.t += 101
            q.expire_leases()
        assert q.lease("a") is not None
        assert not q.status()["workers"][0]["quarantined"]

    def test_expiry_consumes_an_attempt(self, q, clock):
        reg(q, "a", "b")
        jid = q.submit_job(PI, specs(1), retries=1)
        q.lease("a"); clock.t += 101
        q.lease("b"); clock.t += 101
        q.expire_leases()
        assert q.job_counts(jid)["failed"] == 1

    def test_stale_holder_heartbeat_told_to_stop(self, q, clock):
        reg(q, "a", "b")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        clock.t += 101
        q.lease("b")
        assert q.heartbeat(c["chunk_id"], "a") == {"ok": False, "cancel": True}

    def test_restarted_agent_slot_reclaims(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(2))
        first = q.lease("a", 0)
        again = q.lease("a", 0)     # same slot asks again -> the old lease is abandoned
        assert again["chunk_id"] != first["chunk_id"] or again["attempt"] == 2
        assert q.job_counts(1)["leased"] == 1

    def test_lease_survives_restart_of_queue(self, tmp_path, clock):
        p = tmp_path / "r.db"
        a = FleetQueue(p, clock=clock, lease_s=100)
        a.register_worker("w", "/d", 1)
        a.submit_job(PI, specs(1))
        c = a.lease("w")
        a.close()
        b = FleetQueue(p, clock=clock, lease_s=100)
        assert b.heartbeat(c["chunk_id"], "w", 1)["ok"]
        assert b.complete(c["chunk_id"], "w")
        b.close()


class TestComplete:
    def test_complete_updates_counters(self, q):
        reg(q, "a")
        jid = q.submit_job(PI, specs(1))
        c = q.lease("a")
        assert q.complete(c["chunk_id"], "a", duration=1800, combos=5)
        w = q.status()["workers"][0]
        assert w["done"] == 1 and w["combos"] == 5 and w["chunks_per_hour"] == pytest.approx(2.0)
        assert q.job_finished(jid)

    def test_late_complete_accepted_when_chunk_still_pending(self, q, clock):
        reg(q, "a")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        clock.t += 101
        q.expire_leases()
        assert q.complete(c["chunk_id"], "a")

    def test_late_complete_rejected_when_someone_else_holds_it(self, q, clock):
        reg(q, "a", "b")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        clock.t += 101
        q.lease("b")
        assert q.complete(c["chunk_id"], "a") is False

    def test_complete_twice_second_rejected(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        assert q.complete(c["chunk_id"], "a")
        assert q.complete(c["chunk_id"], "a") is False

    def test_unknown_chunk(self, q):
        with pytest.raises(KeyError):
            q.complete(999, "a")


class TestRelease:
    def test_release_costs_no_attempt_and_no_blame(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1), retries=0)
        c = q.lease("a")
        assert q.release(c["chunk_id"], "a")
        again = q.lease("a")
        assert again["attempt"] == 1            # retries=0 would have failed it otherwise
        assert q.status()["workers"][0]["failed"] == 0

    def test_block_stops_offers_of_that_job_only(self, q):
        reg(q, "a", "b")
        j1 = q.submit_job(PI, ["0-0/1"])
        j2 = q.submit_job(PI, ["0-0/1"])
        c = q.lease("a")
        assert c["job_id"] == j1
        q.release(c["chunk_id"], "a", block=True, reason="version mismatch")
        assert q.lease("a")["job_id"] == j2      # j1 blocked for a
        assert q.lease("b")["job_id"] == j1

    def test_block_is_absolute_even_when_nothing_else_is_pending(self, q):
        reg(q, "a")
        q.submit_job(PI, ["0-0/1"])
        c = q.lease("a")
        q.release(c["chunk_id"], "a", block=True, reason="version mismatch")
        assert q.lease("a") is None     # tried_on alone would let it retake the last chunk

    def test_plain_release_leaves_the_worker_eligible_to_retake(self, q):
        reg(q, "a", "b")
        q.submit_job(PI, ["0-0/1"])
        c = q.lease("a")
        q.release(c["chunk_id"], "a")
        assert q.lease("a")["chunk_id"] == c["chunk_id"]

    def test_plain_release_does_not_mark_the_worker_as_having_tried(self, q):
        reg(q, "a", "b")
        j = q.submit_job(PI, ["0-0/1", "1-1/1"])
        c = q.lease("a")
        q.release(c["chunk_id"], "a")
        # a is not in tried_on, so it is offered the earlier chunk, not skipped past it
        assert q.lease("a")["chunk_id"] == c["chunk_id"]

    def test_release_by_non_holder_is_noop(self, q):
        reg(q, "a", "b")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        assert q.release(c["chunk_id"], "b") is False


class TestCancel:
    def test_cancel_stops_offers_and_signals_holders(self, q):
        reg(q, "a", "b")
        jid = q.submit_job(PI, specs(3))
        c = q.lease("a")
        q.cancel_job(jid)
        assert q.lease("b") is None
        assert q.heartbeat(c["chunk_id"], "a") == {"ok": True, "cancel": True}
        assert q.release(c["chunk_id"], "a")
        assert q.job_finished(jid)
        assert q.job_counts(jid)["failed"] == 3

    def test_fail_after_cancel_does_not_requeue(self, q):
        reg(q, "a")
        jid = q.submit_job(PI, specs(1), retries=5)
        c = q.lease("a")
        q.cancel_job(jid)
        assert q.fail(c["chunk_id"], "a", "killed") == "failed"

    def test_cancel_unknown(self, q):
        with pytest.raises(KeyError):
            q.cancel_job(42)


class TestReporting:
    def test_worker_slots_only_lists_completed(self, q):
        reg(q, "a", "b")
        jid = q.submit_job(PI, specs(3))
        q.complete(q.lease("a", 0)["chunk_id"], "a")
        q.complete(q.lease("a", 1)["chunk_id"], "a")
        q.lease("b", 0)                          # in flight, not done
        assert q.worker_slots(jid) == [("a", 0, "/work/a"), ("a", 1, "/work/a")]

    def test_status_shape(self, q):
        reg(q, "a")
        q.submit_job({"tool": "gen_dataset", "output": "o", "per_item": True}, specs(2))
        q.lease("a", 1)
        st = q.status()
        assert st["jobs"][0]["counts"] == {"pending": 1, "leased": 1, "done": 0, "failed": 0,
                                           "total": 2}
        assert st["inflight"][0]["worker"] == "a" and st["inflight"][0]["slot"] == 1
        assert st["workers"][0]["busy"] == 1 and st["workers"][0]["online"]

    def test_worker_goes_offline_when_silent(self, q, clock):
        reg(q, "a")
        clock.t += 500
        assert not q.status(stale_s=90)["workers"][0]["online"]

    def test_empty_job_rejected(self, q):
        with pytest.raises(ValueError):
            q.submit_job(PI, [])

    def test_progress_recorded(self, q):
        reg(q, "a")
        q.submit_job(PI, specs(1))
        c = q.lease("a")
        q.heartbeat(c["chunk_id"], "a", 7)
        assert q.status()["inflight"][0]["progress"] == 7

    def test_job_spec_roundtrip(self, q):
        jid = q.submit_job({"gen_args": ["--x", "a b"]}, specs(1))
        assert q.job_spec(jid) == {"gen_args": ["--x", "a b"]}
        with pytest.raises(KeyError):
            q.job_spec(99)
