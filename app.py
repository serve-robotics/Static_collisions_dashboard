#!/usr/bin/env python3
"""
app.py

Minimal local web dashboard for the static-collisions pipeline. Pick a day,
it runs query.txt against BigQuery for that day, feeds the resulting impact
events into build_grid_videos_from_prefix.process_event (unchanged) to
download + camera-sync + impact-highlight + grid-stitch each event, and
displays the finished 2x2 grid videos in the browser as they complete.

Run: ./.venv/bin/python app.py
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory
from google.cloud import bigquery

from build_grid_videos_from_prefix import process_event

# ── config ────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
QUERY_PATH    = ROOT_DIR / "query.txt"
WORK_DIR      = ROOT_DIR / "work"
GRID_DIR      = ROOT_DIR / "grid_videos"
BQ_PROJECT    = "serve-robotics"   # billed project for the query
EVENT_WORKERS = 4                  # concurrent events in flight, same as the CLI default

WORK_DIR.mkdir(exist_ok=True)
GRID_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


# ── BigQuery ──────────────────────────────────────────────────────────────────

_DECLARE_RE = re.compile(r"^\s*DECLARE\s+(start_ts|end_ts)\b", re.IGNORECASE)


_BARE_START_TS_RE = re.compile(r"\bstart_ts\b")
_BARE_END_TS_RE   = re.compile(r"\bend_ts\b")


def _parameterized_query() -> str:
    """query.txt hardcodes the day via two DECLARE...DEFAULT lines and then
    references the bare scripting variables `start_ts`/`end_ts`. Strip the
    DECLAREs (the day becomes caller-supplied), then rewrite those bare
    references to `@start_ts`/`@end_ts` -- BigQuery query parameters must be
    referenced with an `@` prefix, unlike DECLAREd scripting variables."""
    lines = QUERY_PATH.read_text().splitlines()
    kept = [line for line in lines if not _DECLARE_RE.match(line)]
    sql = "\n".join(kept)
    sql = _BARE_START_TS_RE.sub("@start_ts", sql)
    sql = _BARE_END_TS_RE.sub("@end_ts", sql)
    return sql


def _row_to_event(row: bigquery.table.Row) -> dict:
    d = dict(row)
    ts = d.get("event_ts")
    # process_event -> parse_event_ts only accepts "%Y-%m-%d %H:%M:%S(.%f) UTC";
    # BigQuery gives back a datetime whose str() ("...+00:00") does NOT match
    # that format, so it must be formatted explicitly rather than stringified.
    if isinstance(ts, datetime):
        d["event_ts"] = ts.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    speed_ts = d.get("speed_ts")
    if isinstance(speed_ts, datetime):
        d["speed_ts"] = speed_ts.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    prefixes = d.get("candidate_camera_prefixes")
    if prefixes is not None:
        d["candidate_camera_prefixes"] = list(prefixes)
    return d


def run_query_for_day(day: date) -> list[dict]:
    sql = _parameterized_query()
    client = bigquery.Client(project=BQ_PROJECT)
    start_dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start_dt),
            bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end_dt),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [_row_to_event(r) for r in rows]


# ── job manager ───────────────────────────────────────────────────────────────

class JobManager:
    """Drives one active day's worth of events through process_event in
    bounded batches ("Load more"), streaming results into self.state so the
    browser can poll for progress. Only one query/batch runs at a time."""

    def __init__(self, work_dir: Path, output_dir: Path, event_workers: int = EVENT_WORKERS):
        self.work_dir = str(work_dir)
        self.output_dir = str(output_dir)
        self.event_workers = event_workers
        self.pool = ThreadPoolExecutor(max_workers=event_workers)
        self.lock = threading.Lock()
        self.state = self._fresh_state(None)

    @staticmethod
    def _fresh_state(day: Optional[date]) -> dict:
        return {
            "status": "idle",  # idle | querying | running | done | error
            "day": day.isoformat() if day else None,
            "query_count": 0,
            "events": [],
            "cursor": 0,
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "videos": [],
            "error": None,
            "exhausted": False,
            "force_rebuild": False,
        }

    def status(self) -> dict:
        with self.lock:
            s = self.state
            return {
                "status": s["status"],
                "day": s["day"],
                "query_count": s["query_count"],
                "attempted": s["attempted"],
                "succeeded": s["succeeded"],
                "failed": s["failed"],
                "exhausted": s["exhausted"],
                "error": s["error"],
                "videos": list(s["videos"]),
            }

    def start_new_day(self, day: date, batch_size: int, force_rebuild: bool) -> bool:
        with self.lock:
            if self.state["status"] in ("querying", "running"):
                return False
            self.state = self._fresh_state(day)
            self.state["status"] = "querying"
            self.state["force_rebuild"] = force_rebuild
        threading.Thread(target=self._run_day_job, args=(day, batch_size, force_rebuild), daemon=True).start()
        return True

    def continue_batch(self, batch_size: int) -> bool:
        with self.lock:
            if self.state["status"] in ("querying", "running"):
                return False
            if self.state["day"] is None or self.state["status"] == "error":
                return False
            if self.state["exhausted"]:
                return False
            self.state["status"] = "running"
            force_rebuild = self.state["force_rebuild"]
        threading.Thread(target=self._process_batch, args=(batch_size, force_rebuild), daemon=True).start()
        return True

    def _run_day_job(self, day: date, batch_size: int, force_rebuild: bool) -> None:
        try:
            events = run_query_for_day(day)
        except Exception as e:
            with self.lock:
                self.state["status"] = "error"
                self.state["error"] = str(e)
            return
        with self.lock:
            self.state["events"] = events
            self.state["query_count"] = len(events)
            self.state["exhausted"] = len(events) == 0
            self.state["status"] = "running"
        if events:
            self._process_batch(batch_size, force_rebuild)
        else:
            with self.lock:
                self.state["status"] = "done"

    def _process_one(self, event: dict, force_rebuild: bool):
        try:
            result = process_event(event, self.work_dir, self.output_dir, skip_existing=not force_rebuild)
            return result, None
        except Exception as e:
            return None, e

    @staticmethod
    def _video_meta(event: dict, result_path: str) -> dict:
        return {
            "filename": os.path.basename(result_path),
            "robot_id": event.get("robot_id") or event.get("rover_id"),
            "event_ts": event.get("event_ts"),
            "impact_severity": event.get("impact_severity"),
            "nearest_speed": event.get("nearest_speed"),
            "max_abs_speed": event.get("max_abs_speed"),
            "event_id": event.get("event_id"),
        }

    def _process_batch(self, batch_size: int, force_rebuild: bool) -> None:
        with self.lock:
            events = self.state["events"]
            cursor = self.state["cursor"]

        event_iter = iter(events[cursor:])
        consumed = 0
        batch_succeeded = 0
        pending: dict = {}

        def submit_next() -> bool:
            nonlocal consumed
            event = next(event_iter, None)
            if event is None:
                return False
            consumed += 1
            pending[self.pool.submit(self._process_one, event, force_rebuild)] = event
            return True

        def top_up() -> None:
            # Only ever keep enough in flight to plausibly hit batch_size --
            # mirrors build_grid_videos_from_prefix.main()'s top_up, so a
            # burst of successes doesn't overshoot with already-launched work.
            while len(pending) < self.event_workers:
                if batch_succeeded + len(pending) >= batch_size:
                    break
                if not submit_next():
                    break

        top_up()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                event = pending.pop(future)
                result, err = future.result()
                with self.lock:
                    self.state["attempted"] += 1
                    if err is not None or not result:
                        self.state["failed"] += 1
                    else:
                        self.state["succeeded"] += 1
                        self.state["videos"].append(self._video_meta(event, result))
                if err is None and result:
                    batch_succeeded += 1
            if batch_succeeded >= batch_size:
                break
            top_up()

        with self.lock:
            self.state["cursor"] = cursor + consumed
            self.state["exhausted"] = self.state["cursor"] >= len(events)
            self.state["status"] = "done"


job_manager = JobManager(WORK_DIR, GRID_DIR)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    body = request.get_json(force=True, silent=True) or {}
    day_str = body.get("day")
    if not day_str:
        return jsonify(error="day is required (YYYY-MM-DD)"), 400
    try:
        day = date.fromisoformat(day_str)
    except ValueError:
        return jsonify(error="invalid day format, expected YYYY-MM-DD"), 400
    try:
        batch_size = int(body.get("batch_size", 20))
    except (TypeError, ValueError):
        return jsonify(error="batch_size must be an integer"), 400
    force_rebuild = bool(body.get("force_rebuild", False))

    if not job_manager.start_new_day(day, batch_size, force_rebuild):
        return jsonify(error="a job is already running"), 409
    return jsonify(job_manager.status()), 202


@app.route("/more", methods=["POST"])
def more():
    body = request.get_json(force=True, silent=True) or {}
    try:
        batch_size = int(body.get("batch_size", 20))
    except (TypeError, ValueError):
        return jsonify(error="batch_size must be an integer"), 400

    if not job_manager.continue_batch(batch_size):
        return jsonify(error="no active day, a batch is already running, or events are exhausted"), 409
    return jsonify(job_manager.status()), 202


@app.route("/status")
def status():
    return jsonify(job_manager.status())


@app.route("/videos/<path:filename>")
def videos(filename):
    return send_from_directory(str(GRID_DIR), filename, conditional=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static collisions dashboard")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
