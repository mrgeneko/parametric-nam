"""Durable work queue for the pull-agent fleet (docs/implementation-roadmap.md item 11).

The coordinator (fleet_coordinator.py) owns ONE of these; agents never touch the database, they
speak HTTP. Everything here is plain SQLite (WAL) so the queue survives a coordinator restart:
leased chunks keep their leases, and an agent that finishes while the coordinator is down simply
retries its report.

The scheduling rules are distribute_pull.py's, moved from in-memory dicts into rows:

  * a chunk is a `--shard i-i/N` spec; a retry goes to a DIFFERENT worker where possible
    (`tried_on`), falling back to one that already failed it only when nothing else is pending;
  * retries: a failed chunk is requeued while attempts <= retries, then fails for good;
  * quarantine: N consecutive failures with no successes benches a worker;
  * a lease that is not renewed in time is treated as a failed attempt on that worker -- the
    agent vanished (reboot, network). Safe because renderers resume-skip finished items and
    write outputs atomically.

Deliberately NOT here: process supervision, pacing statistics, HTTP. This module is pure state
so every rule can be tested with a fake clock.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec TEXT NOT NULL,
    retries INTEGER NOT NULL,
    cancelled INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    spec TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',      -- pending | leased | done | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    tried_on TEXT NOT NULL DEFAULT '[]',
    worker TEXT, slot INTEGER,
    lease_expires REAL,
    progress INTEGER NOT NULL DEFAULT 0,
    started REAL, last_beat REAL, finished REAL,
    duration REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS chunks_by_state ON chunks(job_id, state);
CREATE TABLE IF NOT EXISTS workers (
    name TEXT PRIMARY KEY,
    dir TEXT, slots INTEGER, info TEXT NOT NULL DEFAULT '{}',
    last_seen REAL,
    done INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
    combos INTEGER NOT NULL DEFAULT 0, secs REAL NOT NULL DEFAULT 0,
    consec_fail INTEGER NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS blocks (
    job_id INTEGER NOT NULL, worker TEXT NOT NULL, reason TEXT,
    PRIMARY KEY (job_id, worker)
);
"""


class FleetQueue:
    def __init__(self, path, clock=time.time, lease_s: float = 120.0,
                 quarantine_after: int = 3):
        self.clock = clock
        self.lease_s = lease_s
        self.quarantine_after = quarantine_after
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def _tx(self):
        return _Tx(self)

    # ---- jobs --------------------------------------------------------------------------
    def submit_job(self, spec: dict, chunk_specs: "list[str]", retries: int = 2) -> int:
        if not chunk_specs:
            raise ValueError("a job needs at least one chunk")
        with self._tx() as db:
            cur = db.execute("INSERT INTO jobs(spec, retries, created) VALUES (?,?,?)",
                             (json.dumps(spec), int(retries), self.clock()))
            jid = cur.lastrowid
            db.executemany("INSERT INTO chunks(job_id, spec) VALUES (?,?)",
                           [(jid, c) for c in chunk_specs])
        return jid

    def job_spec(self, job_id: int) -> dict:
        row = self.db.execute("SELECT spec FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such job {job_id}")
        return json.loads(row["spec"])

    def cancel_job(self, job_id: int) -> None:
        """Stop handing out chunks. A chunk already leased is told to stop on its next
        heartbeat; what it already rendered stays on disk (resume-skip)."""
        with self._tx() as db:
            if db.execute("UPDATE jobs SET cancelled=1 WHERE id=?", (job_id,)).rowcount == 0:
                raise KeyError(f"no such job {job_id}")
            db.execute("UPDATE chunks SET state='failed', error='job cancelled', finished=? "
                       "WHERE job_id=? AND state='pending'", (self.clock(), job_id))

    # ---- workers -----------------------------------------------------------------------
    def register_worker(self, name: str, dir: str, slots: int, info: "dict | None" = None):
        with self._tx() as db:
            db.execute("INSERT INTO workers(name, dir, slots, info, last_seen) VALUES (?,?,?,?,?) "
                       "ON CONFLICT(name) DO UPDATE SET dir=excluded.dir, slots=excluded.slots, "
                       "info=excluded.info, last_seen=excluded.last_seen",
                       (name, dir, int(slots), json.dumps(info or {}), self.clock()))

    def unquarantine(self, name: str) -> None:
        with self._tx() as db:
            if db.execute("UPDATE workers SET quarantined=0, consec_fail=0 WHERE name=?",
                          (name,)).rowcount == 0:
                raise KeyError(f"no such worker {name}")

    def _touch(self, db, name):
        db.execute("UPDATE workers SET last_seen=? WHERE name=?", (self.clock(), name))

    # ---- leasing -----------------------------------------------------------------------
    def expire_leases(self) -> int:
        with self._tx() as db:
            return self._expire(db)

    def _expire(self, db) -> int:
        now = self.clock()
        rows = db.execute("SELECT id, job_id, worker FROM chunks WHERE state='leased' "
                          "AND lease_expires < ?", (now,)).fetchall()
        for r in rows:
            self._attempt_failed(db, r["id"], r["worker"],
                                 "lease expired: agent stopped heartbeating", count_worker=False)
        return len(rows)

    def lease(self, worker: str, slot: int = 0) -> "dict | None":
        """The next chunk for `worker`/`slot`, or None. Returns
        {"chunk_id", "job_id", "spec", "job", "attempt"}."""
        with self._tx() as db:
            self._expire(db)
            w = db.execute("SELECT quarantined FROM workers WHERE name=?", (worker,)).fetchone()
            if w is None:
                raise KeyError(f"worker {worker!r} is not registered")
            self._touch(db, worker)
            if w["quarantined"]:
                return None
            # A slot holds one chunk at a time: an agent that restarted and asks again has
            # abandoned whatever it held here.
            for r in db.execute("SELECT id FROM chunks WHERE state='leased' AND worker=? "
                                "AND slot=?", (worker, slot)).fetchall():
                self._attempt_failed(db, r["id"], worker, "agent restarted", count_worker=False)
            rows = db.execute(
                "SELECT c.id, c.job_id, c.spec, c.attempts, c.tried_on, j.spec AS job_spec "
                "FROM chunks c "
                "JOIN jobs j ON j.id=c.job_id "
                "WHERE c.state='pending' AND j.cancelled=0 AND NOT EXISTS "
                "(SELECT 1 FROM blocks b WHERE b.job_id=c.job_id AND b.worker=?) "
                "ORDER BY c.job_id, c.id", (worker,)).fetchall()
            # Whole-chunk jobs run one renderer using every core of the agent, so only slot 0
            # takes them; extra slots exist for per-item jobs (one core each).
            rows = [r for r in rows if slot == 0 or json.loads(r["job_spec"]).get("per_item")]
            pick = next((r for r in rows if worker not in json.loads(r["tried_on"])), None)
            if pick is None and rows:
                pick = rows[0]     # only chunks this worker already failed are left
            if pick is None:
                return None
            now = self.clock()
            db.execute("UPDATE chunks SET state='leased', worker=?, slot=?, attempts=attempts+1, "
                       "lease_expires=?, started=?, last_beat=?, progress=0, error=NULL "
                       "WHERE id=?", (worker, slot, now + self.lease_s, now, now, pick["id"]))
            return {"chunk_id": pick["id"], "job_id": pick["job_id"], "spec": pick["spec"],
                    "job": json.loads(pick["job_spec"]),
                    "attempt": pick["attempts"] + 1}

    def heartbeat(self, chunk_id: int, worker: str, progress: int = 0) -> dict:
        """{"ok": bool, "cancel": bool}. ok=False means this worker no longer holds the chunk
        (lease expired and it was reassigned): the agent must stop rendering it."""
        with self._tx() as db:
            self._touch(db, worker)
            row = db.execute("SELECT c.state, c.worker, j.cancelled FROM chunks c "
                             "JOIN jobs j ON j.id=c.job_id WHERE c.id=?", (chunk_id,)).fetchone()
            if row is None or row["state"] != "leased" or row["worker"] != worker:
                return {"ok": False, "cancel": True}
            now = self.clock()
            db.execute("UPDATE chunks SET lease_expires=?, last_beat=?, progress=? WHERE id=?",
                       (now + self.lease_s, now, int(progress), chunk_id))
            return {"ok": True, "cancel": bool(row["cancelled"])}

    # ---- results -----------------------------------------------------------------------
    def complete(self, chunk_id: int, worker: str, duration: float = 0.0, combos: int = 0) -> bool:
        """Accepted when the worker still holds the lease, or when the lease lapsed but nobody
        else has taken the chunk (its output is valid: outputs are written atomically). False
        when another worker now owns or finished it."""
        with self._tx() as db:
            row = db.execute("SELECT state, worker FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            if row is None:
                raise KeyError(f"no such chunk {chunk_id}")
            mine = row["state"] == "leased" and row["worker"] == worker
            orphan = row["state"] == "pending"
            if not (mine or orphan):
                return False
            db.execute("UPDATE chunks SET state='done', worker=?, finished=?, duration=?, "
                       "lease_expires=NULL WHERE id=?", (worker, self.clock(), duration, chunk_id))
            db.execute("UPDATE workers SET done=done+1, combos=combos+?, secs=secs+?, "
                       "consec_fail=0, last_seen=? WHERE name=?",
                       (int(combos), float(duration), self.clock(), worker))
            return True

    def fail(self, chunk_id: int, worker: str, error: str = "") -> str:
        """Record a failed attempt. Returns 'requeued', 'failed' (retries exhausted), or
        'ignored' (this worker did not hold the chunk)."""
        with self._tx() as db:
            row = db.execute("SELECT state, worker FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            if row is None:
                raise KeyError(f"no such chunk {chunk_id}")
            if row["state"] != "leased" or row["worker"] != worker:
                return "ignored"
            return self._attempt_failed(db, chunk_id, worker, error, count_worker=True)

    def release(self, chunk_id: int, worker: str, block: bool = False, reason: str = "") -> bool:
        """Hand a chunk back WITHOUT costing it an attempt or blaming the worker: an agent
        shutting down, or one that refuses the job (version mismatch). block=True also stops
        this worker being offered any more of the job."""
        with self._tx() as db:
            row = db.execute("SELECT c.state, c.worker, c.job_id, c.tried_on, j.cancelled "
                             "FROM chunks c JOIN jobs j ON j.id=c.job_id WHERE c.id=?",
                             (chunk_id,)).fetchone()
            if row is None or row["state"] != "leased" or row["worker"] != worker:
                return False
            if row["cancelled"]:
                db.execute("UPDATE chunks SET state='failed', error='job cancelled', "
                           "lease_expires=NULL, finished=? WHERE id=?", (self.clock(), chunk_id))
                return True
            tried = json.loads(row["tried_on"])
            if block and worker not in tried:
                tried.append(worker)
            db.execute("UPDATE chunks SET state='pending', worker=NULL, slot=NULL, "
                       "lease_expires=NULL, attempts=attempts-1, tried_on=?, error=? WHERE id=?",
                       (json.dumps(tried), reason or None, chunk_id))
            if block:
                db.execute("INSERT OR REPLACE INTO blocks(job_id, worker, reason) VALUES (?,?,?)",
                           (row["job_id"], worker, reason))
            return True

    def _attempt_failed(self, db, chunk_id, worker, error, count_worker) -> str:
        row = db.execute("SELECT c.attempts, c.tried_on, j.retries, j.cancelled FROM chunks c "
                         "JOIN jobs j ON j.id=c.job_id WHERE c.id=?", (chunk_id,)).fetchone()
        tried = json.loads(row["tried_on"])
        if worker and worker not in tried:
            tried.append(worker)
        if count_worker and worker:
            db.execute("UPDATE workers SET failed=failed+1, consec_fail=consec_fail+1 WHERE name=?",
                       (worker,))
            w = db.execute("SELECT done, consec_fail FROM workers WHERE name=?",
                           (worker,)).fetchone()
            if (self.quarantine_after and w["done"] == 0
                    and w["consec_fail"] >= self.quarantine_after):
                db.execute("UPDATE workers SET quarantined=1 WHERE name=?", (worker,))
        requeue = row["attempts"] <= row["retries"] and not row["cancelled"]
        db.execute("UPDATE chunks SET state=?, tried_on=?, error=?, lease_expires=NULL, "
                   "finished=? WHERE id=?",
                   ("pending" if requeue else "failed", json.dumps(tried), (error or "")[-2000:],
                    None if requeue else self.clock(), chunk_id))
        return "requeued" if requeue else "failed"

    # ---- reporting ---------------------------------------------------------------------
    def job_counts(self, job_id: int) -> dict:
        c = {"pending": 0, "leased": 0, "done": 0, "failed": 0}
        for r in self.db.execute("SELECT state, COUNT(*) n FROM chunks WHERE job_id=? "
                                 "GROUP BY state", (job_id,)):
            c[r["state"]] = r["n"]
        c["total"] = sum(c.values())
        return c

    def job_finished(self, job_id: int) -> bool:
        c = self.job_counts(job_id)
        return c["total"] > 0 and c["pending"] == 0 and c["leased"] == 0

    def worker_slots(self, job_id: int) -> "list[tuple[str, int, str]]":
        """[(worker, slot, worker_dir)] that completed at least one chunk of the job -- exactly
        the places its output lives, for --collect."""
        rows = self.db.execute(
            "SELECT DISTINCT c.worker, c.slot, w.dir FROM chunks c "
            "LEFT JOIN workers w ON w.name=c.worker "
            "WHERE c.job_id=? AND c.state='done' AND c.worker IS NOT NULL "
            "ORDER BY c.worker, c.slot", (job_id,)).fetchall()
        return [(r["worker"], r["slot"] or 0, r["dir"]) for r in rows]

    def status(self, stale_s: float = 90.0) -> dict:
        with self._tx() as db:
            self._expire(db)
        now = self.clock()
        jobs = []
        for j in self.db.execute("SELECT * FROM jobs ORDER BY id"):
            spec = json.loads(j["spec"])
            c = self.job_counts(j["id"])
            first = self.db.execute("SELECT MIN(started) s, MAX(finished) f FROM chunks "
                                    "WHERE job_id=?", (j["id"],)).fetchone()
            jobs.append({
                "id": j["id"], "tool": spec.get("tool"), "output": spec.get("output"),
                "label": spec.get("label"), "counts": c, "cancelled": bool(j["cancelled"]),
                "finished": self.job_finished(j["id"]), "created": j["created"],
                "elapsed": (now - j["created"]),
            })
        inflight = [dict(r) for r in self.db.execute(
            "SELECT id, job_id, spec, worker, slot, attempts, progress, started, last_beat "
            "FROM chunks WHERE state='leased' ORDER BY worker, slot")]
        problems = [dict(r) for r in self.db.execute(
            "SELECT id, job_id, spec, state, attempts, tried_on, error, finished FROM chunks "
            "WHERE error IS NOT NULL AND error != '' AND state IN ('pending','failed') "
            "ORDER BY COALESCE(finished, started) DESC LIMIT 25")]
        workers = []
        busy = {}
        for r in inflight:
            busy[r["worker"]] = busy.get(r["worker"], 0) + 1
        for w in self.db.execute("SELECT * FROM workers ORDER BY name"):
            age = None if w["last_seen"] is None else now - w["last_seen"]
            workers.append({
                "name": w["name"], "dir": w["dir"], "slots": w["slots"], "busy": busy.get(w["name"], 0),
                "done": w["done"], "failed": w["failed"], "combos": w["combos"],
                "chunks_per_hour": (w["done"] / (w["secs"] / 3600)) if w["secs"] else 0.0,
                "quarantined": bool(w["quarantined"]), "last_seen_age": age,
                "online": age is not None and age <= stale_s,
                "info": json.loads(w["info"]),
            })
        return {"now": now, "jobs": jobs, "inflight": inflight, "workers": workers,
                "problems": problems}


class _Tx:
    """One serialized IMMEDIATE transaction: every state change in this module is a
    read-modify-write, and the HTTP server is multi-threaded."""

    def __init__(self, q: FleetQueue):
        self.q = q

    def __enter__(self):
        self.q._lock.acquire()
        self.q.db.execute("BEGIN IMMEDIATE")
        return self.q.db

    def __exit__(self, et, ev, tb):
        try:
            self.q.db.execute("ROLLBACK" if et else "COMMIT")
        finally:
            self.q._lock.release()
        return False
