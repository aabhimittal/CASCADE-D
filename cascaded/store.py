"""Shared proxy state: request log and rate-limit hits, in one SQLite file.

Point every worker process at the same file (e.g. `uvicorn --workers 4`) and they share one ledger,
one drift view and one rate limit. ":memory:" keeps state private to one process.
"""

import sqlite3
import threading
import time

# shortcut: one host only (SQLite over a network filesystem is unsafe); put this behind Redis or Postgres
# when the proxy runs on several machines


class Store:
    def __init__(self, path: str = ":memory:", maxlen: int = 100_000):
        self.maxlen = maxlen
        self._lock = threading.Lock()  # one connection per Store, shared by FastAPI's worker threads
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30, isolation_level=None)
        self._read, self._read_lock = self._db, self._lock
        if path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            # separate read connection: a big dashboard read never holds the lock request logging needs
            self._read = sqlite3.connect(path, check_same_thread=False, timeout=30, isolation_level=None)
            self._read_lock = threading.Lock()
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT,
                tier INTEGER NOT NULL, cost REAL NOT NULL, latency_ms REAL NOT NULL, task TEXT);
            CREATE TABLE IF NOT EXISTS hits (client TEXT NOT NULL, ts REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS hits_by_client ON hits (client, ts);
        """)

    def add(self, tier: int, cost: float, latency_ms: float, task: str | None) -> None:
        with self._lock:
            cur = self._db.execute("INSERT INTO requests (tier, cost, latency_ms, task) VALUES (?, ?, ?, ?)",
                                   (tier, cost, latency_ms, task))
            self._db.execute("DELETE FROM requests WHERE id <= ?", (cur.lastrowid - self.maxlen,))  # keep a window

    def requests(self) -> list[tuple]:
        """(tier, cost, latency_ms, task) for the most recent `maxlen` requests from every process, oldest first."""
        with self._read_lock:
            rows = self._read.execute("SELECT tier, cost, latency_ms, task FROM requests ORDER BY id DESC LIMIT ?",
                                    (self.maxlen,)).fetchall()
        return rows[::-1]

    def allow(self, client: str, per_minute: int, now: float | None = None) -> bool:
        """Sliding one-minute window per client, checked and recorded in one transaction so concurrent
        workers can't both admit the request that should have been the last one."""
        now = time.time() if now is None else now  # wall clock: comparable across processes, unlike monotonic
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM hits WHERE ts <= ?", (now - 60,))
                (count,) = self._db.execute("SELECT COUNT(*) FROM hits WHERE client = ?", (client,)).fetchone()
                ok = count < per_minute
                if ok:
                    self._db.execute("INSERT INTO hits VALUES (?, ?)", (client, now))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return ok
