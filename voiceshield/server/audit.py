"""Audit log in SQLite. Stores scores, decisions and the names of scam tactics seen.
Never stores audio, transcripts or voiceprints."""
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started REAL, ended REAL, source TEXT, mode TEXT, profile TEXT, threshold REAL,
  verdict TEXT, max_risk REAL, windows INTEGER, audio_s REAL, speech_s REAL, mean_llr REAL
);
CREATE TABLE IF NOT EXISTS windows (
  call_id INTEGER, t REAL, speech_s REAL, llr REAL, p_fake REAL, window_score REAL,
  risk REAL, level TEXT, latency_ms REAL
);
CREATE TABLE IF NOT EXISTS events (call_id INTEGER, t REAL, kind TEXT, detail TEXT);
"""


class Audit:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(windows)")}
            if "similarity" not in cols:                       # added in Step 6 (voiceprint check)
                self.db.execute("ALTER TABLE windows ADD COLUMN similarity REAL")
                self.db.commit()

    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def start_call(self, source: str, mode: str, profile: str, threshold: float) -> int:
        return self._exec("INSERT INTO calls (started, source, mode, profile, threshold, verdict, max_risk, windows) "
                          "VALUES (?, ?, ?, ?, ?, 'PASS', 0, 0)", (time.time(), source, mode, profile, threshold)).lastrowid

    def window(self, call_id: int, **w) -> None:
        self._exec("INSERT INTO windows (call_id, t, speech_s, llr, p_fake, window_score, risk, level, latency_ms, "
                   "similarity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (call_id, w["t"], w["speech_s"], w["llr"], w["p_fake"], w.get("window_score"),
                    w.get("risk"), w.get("level"), w["latency_ms"], w.get("similarity")))

    def event(self, call_id: int, t: float, kind: str, detail: str = "") -> None:
        self._exec("INSERT INTO events VALUES (?, ?, ?, ?)", (call_id, t, kind, detail))

    def progress(self, call_id: int, verdict: str, max_risk: float) -> None:
        """While a call is open, keep its row's verdict and peak risk current, so the Incidents and History views
        show a live attack as it happens instead of 'PASS, risk 0' until the call ends. Escalation only."""
        rank = "CASE verdict WHEN 'HOLD' THEN 2 WHEN 'VERIFY' THEN 1 ELSE 0 END"
        new = {"HOLD": 2, "VERIFY": 1}.get(verdict, 0)
        self._exec(f"UPDATE calls SET verdict = CASE WHEN ? > {rank} THEN ? ELSE verdict END, "
                   "max_risk = MAX(COALESCE(max_risk, 0), ?) WHERE id=? AND ended IS NULL", (new, verdict, max_risk, call_id))

    def end_call(self, call_id: int, verdict: str, max_risk: float, audio_s: float, speech_s: float) -> dict:
        with self.lock:
            row = self.db.execute("SELECT COUNT(*) n, AVG(llr) m FROM windows WHERE call_id=?", (call_id,)).fetchone()
        self._exec("UPDATE calls SET ended=?, verdict=?, max_risk=?, windows=?, audio_s=?, speech_s=?, mean_llr=? WHERE id=?",
                   (time.time(), verdict, max_risk, row["n"], audio_s, speech_s, row["m"], call_id))
        return self.call(call_id)

    def call(self, call_id: int) -> dict | None:
        with self.lock:
            c = self.db.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
            if c is None:
                return None
            ws = self.db.execute("SELECT * FROM windows WHERE call_id=? ORDER BY t", (call_id,)).fetchall()
            ev = self.db.execute("SELECT * FROM events WHERE call_id=? ORDER BY t", (call_id,)).fetchall()
        return {**dict(c), "window_log": [dict(w) for w in ws], "events": [dict(e) for e in ev]}

    def recent(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def incidents(self, limit: int = 30) -> list[dict]:
        """Recent monitored calls with their events (alerts, scam tactics, prevention actions), newest first."""
        with self.lock:
            rows = self.db.execute("SELECT * FROM calls WHERE mode='monitor' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                ev = self.db.execute("SELECT t, kind, detail FROM events WHERE call_id=? ORDER BY rowid", (r["id"],)).fetchall()
                out.append({**dict(r), "events": [dict(e) for e in ev]})
        return out
