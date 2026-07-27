#!/usr/bin/env python3
"""
build_grid_videos.py

Reads rover collision events from a job JSON file (list of dicts with
rover_id, event_ts, event_id, camera_files), downloads each event's camera
.ts segments from GCS, merges the segments per camera into one video per
camera, crops each to a shared ±10s window around event_ts (using the same
shared-trim-origin camera-sync logic as build_clip_dataset.py's ex.py, so the
event lands at the same offset across all cameras), draws a one-frame red
border at the event instant, then stitches the four cameras (front, left,
right, rear) into a single 2x2 grid video per event:

    front | left
    right | rear

Output: {output_dir}/{rover_id}_{sanitized_event_ts}_{event_id}.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from google.cloud import storage
from tqdm import tqdm

# ── binary paths ───────────────────────────────────────────────────────────
FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

# ── constants ────────────────────────────────────────────────────────────────
CAMERAS    = ["front", "left", "right", "rear"]  # grid order: top-left, top-right, bottom-left, bottom-right
GCS_BUCKET = "rovers-data"
GCS_PREFIX = "cameras/"
CELL_W     = 640
FPS        = 10     # matches build_clip_dataset.py; used to size the highlighted window
FRAME_DUR  = 1.0 / FPS
PRE_POST   = 10.0   # seconds before/after event_ts to keep in the cropped clip
CELL_H     = 360

# ── GCS helpers ───────────────────────────────────────────────────────────────

_storage_client: Optional[storage.Client] = None
_storage_client_lock = threading.Lock()


def get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        with _storage_client_lock:
            if _storage_client is None:
                _storage_client = storage.Client()
    return _storage_client


def normalize_gcs_path(raw_path: str) -> str:
    path = raw_path.strip()
    gs_prefix = f"gs://{GCS_BUCKET}/"
    if path.startswith(gs_prefix):
        path = path[len(gs_prefix):]
    if not path.startswith(GCS_PREFIX):
        path = GCS_PREFIX + path
    return path


def download_from_gcs(gcs_path: str, local_dir: str) -> str:
    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    object_path = normalize_gcs_path(gcs_path)
    blob = bucket.blob(object_path)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "__".join(object_path.split("/")))
    if os.path.exists(local_path):
        return local_path

    tmp_path = f"{local_path}.{os.getpid()}.{threading.get_ident()}.part"
    try:
        blob.download_to_filename(tmp_path)
        os.replace(tmp_path, local_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return local_path


def download_segments_concurrent(
    by_camera: dict[str, list[str]], event_dir: str, max_workers: int = 8,
) -> dict[str, list[str]]:
    """Download every segment across every camera for ONE event concurrently."""
    tasks = [(cam, p) for cam, paths in by_camera.items() for p in paths]
    if not tasks:
        return {cam: [] for cam in by_camera}

    cam_tmp_dirs = {cam: os.path.join(event_dir, "tmp", cam) for cam in by_camera if by_camera[cam]}
    for d in cam_tmp_dirs.values():
        os.makedirs(d, exist_ok=True)

    local_by_gcs_path: dict[tuple[str, str], str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_from_gcs, gcs_path, cam_tmp_dirs[cam]): (cam, gcs_path)
            for cam, gcs_path in tasks
        }
        for future in as_completed(futures):
            cam, gcs_path = futures[future]
            try:
                local_by_gcs_path[(cam, gcs_path)] = future.result()
            except Exception as e:
                tqdm.write(f"  WARNING [{cam}] download failed {gcs_path}: {e}")

    return {
        cam: [local_by_gcs_path[(cam, p)] for p in paths if (cam, p) in local_by_gcs_path]
        for cam, paths in by_camera.items()
    }


# ── timestamp helpers ─────────────────────────────────────────────────────────

def parse_event_ts(s: str) -> datetime:
    s = s.strip()
    fmt = "%Y-%m-%d %H:%M:%S.%f UTC" if "." in s else "%Y-%m-%d %H:%M:%S UTC"
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)


def parse_cam_ts(local_path: str) -> Optional[datetime]:
    m = re.search(
        r"(front|rear|left|right)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-(\d{3})\.",
        os.path.basename(local_path),
    )
    if not m:
        return None
    dt = datetime.strptime(m.group(2), "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc)
    return dt + timedelta(milliseconds=int(m.group(3)))


# ── video helpers ─────────────────────────────────────────────────────────────

def camera_from_filename(path: str) -> Optional[str]:
    m = re.search(r"(front|rear|left|right)-", os.path.basename(path))
    return m.group(1) if m else None


def merge_segments(camera: str, segment_paths: list[str], out_dir: str) -> Optional[str]:
    segment_paths = [p for p in segment_paths if os.path.exists(p)]
    if not segment_paths:
        return None
    concat_file = os.path.join(out_dir, f"{camera}.concat.txt")
    output_mp4  = os.path.join(out_dir, f"{camera}.mp4")
    with open(concat_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        FFMPEG_BIN,
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        "-y", "-loglevel", "error",
        output_mp4,
    ]
    subprocess.run(cmd, check=True)
    return output_mp4


def trim_video(input_path: str, output_path: str, start: float, end: float) -> str:
    duration = end - start
    cmd = [
        FFMPEG_BIN,
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
        "-y", "-loglevel", "error",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def get_video_duration(video_path: str) -> float:
    ffprobe_bin = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "/usr/bin/ffprobe"
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def build_grid(
    camera_videos: dict[str, str],
    out_path: str,
    camera_offsets: Optional[dict[str, float]] = None,
) -> None:
    """Stitch up to 4 camera videos into a 2x2 grid: front|left / right|rear.
    A missing camera is replaced with a black filler cell of matching duration.

    `camera_offsets` maps camera -> seconds-from-clip-start of the event
    timestamp for that camera's own footage; that camera's quadrant gets a
    red border drawn for one frame's duration centered there.
    """
    camera_offsets = camera_offsets or {}
    durations = [get_video_duration(p) for p in camera_videos.values()]
    duration = max(durations) if durations else 1.0

    inputs: list[str] = []
    for cam in CAMERAS:
        path = camera_videos.get(cam)
        if path:
            inputs += ["-i", path]
        else:
            inputs += [
                "-f", "lavfi",
                "-i", f"color=c=black:s={CELL_W}x{CELL_H}:rate=10:d={duration}",
            ]

    filter_parts = []
    for idx, cam in enumerate(CAMERAS):
        stage = f"[{idx}:v]scale={CELL_W}:{CELL_H},setsar=1"
        offset = camera_offsets.get(cam)
        if offset is not None and 0 <= offset <= duration:
            start = max(0.0, offset - FRAME_DUR / 2)
            end   = offset + FRAME_DUR / 2
            stage += (
                f",drawbox=x=0:y=0:w=iw:h=ih:color=red:t=8:"
                f"enable='between(t,{start:.3f},{end:.3f})'"
            )
        filter_parts.append(f"{stage}[v{idx}]")
    filter_parts.append("[v0][v1]hstack=inputs=2[top]")
    filter_parts.append("[v2][v3]hstack=inputs=2[bottom]")
    filter_parts.append("[top][bottom]vstack=inputs=2[grid]")
    filter_complex = ";".join(filter_parts)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[grid]",
        "-y", "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)


# ── per-event pipeline ────────────────────────────────────────────────────────

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def load_events(path: str) -> list[dict]:
    """Load events from either a plain JSON array or BigQuery's
    newline-delimited JSON export (one JSON object per line)."""
    with open(path) as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def process_event(event: dict, work_dir: str, output_dir: str, skip_existing: bool = True) -> Optional[str]:
    rover_id     = str(event.get("rover_id") or event.get("robot_id") or "unknown")
    event_ts_str = str(event.get("event_ts", ""))
    event_id     = str(event.get("event_id", "unknown"))
    camera_files = event.get("camera_files", [])

    sanitized_ts = sanitize(event_ts_str)
    event_label  = f"{rover_id}_{sanitized_ts}_{event_id}"
    event_dir    = os.path.join(work_dir, event_label)
    out_path     = os.path.join(output_dir, f"{event_label}.mp4")

    if skip_existing and os.path.exists(out_path):
        return out_path

    by_camera: dict[str, list[str]] = {c: [] for c in CAMERAS}
    for cf in camera_files:
        cam = camera_from_filename(cf.split("/")[-1])
        if cam and cam in by_camera:
            by_camera[cam].append(cf)
    for cam in by_camera:
        by_camera[cam].sort()

    if not any(by_camera.values()):
        tqdm.write(f"  {event_label}: no recognizable camera files, skipping")
        return None

    try:
        event_ts = parse_event_ts(event_ts_str)
    except ValueError:
        tqdm.write(f"  {event_label}: unparseable event_ts {event_ts_str!r}, skipping")
        return None

    downloaded = download_segments_concurrent(by_camera, event_dir)

    # ── Phase A: per camera, merge + compute collision_in_merged (no trim yet;
    # trimming needs the shared origin computed in Phase B). Exact mirror of
    # build_clip_dataset.py's (ex.py) process_event Phase A.
    merged_info: dict[str, tuple[str, float]] = {}  # cam -> (merged_path, collision_in_merged)
    for cam in CAMERAS:
        local_segments = downloaded.get(cam, [])
        if not local_segments:
            continue
        cam_tmp_dir = os.path.join(event_dir, "tmp", cam)
        try:
            merged_path = merge_segments(cam, local_segments, cam_tmp_dir)
        except Exception as e:
            tqdm.write(f"  {event_label} ERROR [{cam}] merge failed: {e}")
            continue
        if not merged_path:
            continue

        seg_times = [t for t in (parse_cam_ts(p) for p in local_segments) if t is not None]
        if not seg_times:
            tqdm.write(f"  {event_label} WARNING [{cam}] cannot parse segment timestamps, skipping")
            continue
        min_seg_time = min(seg_times)
        collision_in_merged = (event_ts - min_seg_time).total_seconds()
        merged_info[cam] = (merged_path, collision_in_merged)

    if not merged_info:
        tqdm.write(f"  {event_label}: no merged camera videos, skipping")
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    # ── Phase B: shared trim origin across all cameras with footage, so the
    # event timestamp lands at the SAME offset in every camera's trimmed clip.
    # Trimming each camera independently with trim_start = max(0.0,
    # collision_in_merged - PRE_POST) lets a camera whose footage doesn't
    # reach back PRE_POST seconds before the event "clamp" to trim_start=0
    # while other cameras don't -- so frame 0 of that camera's trimmed video
    # lands at a different real-world instant than frame 0 of an unclamped
    # camera, silently desyncing the grid quadrants by up to ~PRE_POST
    # seconds. Using one shared_pre -- the tightest lead-in actually
    # available across the group -- for every camera's trim guarantees
    # collision_in_trimmed comes out identical for all cameras, at the cost
    # of a uniformly shorter lead-in whenever any single camera is the
    # constraining one. (Exact mirror of build_clip_dataset.py's Phase B;
    # the post-collision side is intentionally NOT shared -- each camera
    # keeps up to PRE_POST seconds of its own available footage after the
    # event, same as ex.py's WINDOW_DUR post-window.)
    shared_pre = min([PRE_POST] + [t for _, t in merged_info.values()])
    if shared_pre < PRE_POST - 0.05:
        constraining = [c for c, (_, t) in merged_info.items() if abs(t - shared_pre) < 1e-6]
        tqdm.write(f"  {event_label}: lead-in reduced to {shared_pre:.3f}s "
                   f"(from {PRE_POST}s) by camera(s) {constraining}")

    # ── Phase C: per camera, trim using the shared origin
    trimmed: dict[str, str] = {}
    collision_times: dict[str, float] = {}
    for cam, (merged_path, collision_in_merged) in merged_info.items():
        trimmed_path = os.path.join(event_dir, f"{cam}_trimmed.mp4")
        trim_start = collision_in_merged - shared_pre  # always >= 0 by construction
        trim_end   = collision_in_merged + PRE_POST
        collision_in_trimmed = collision_in_merged - trim_start  # == shared_pre, every camera
        try:
            trim_video(merged_path, trimmed_path, trim_start, trim_end)
        except Exception as e:
            tqdm.write(f"  {event_label} ERROR [{cam}] trim failed: {e}")
            continue
        trimmed[cam] = trimmed_path
        collision_times[cam] = collision_in_trimmed

    if not trimmed:
        tqdm.write(f"  {event_label}: no trimmed camera videos, skipping")
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    assert all(abs(t - shared_pre) < 1e-6 for t in collision_times.values()), (
        f"{event_label}: collision_times should all equal shared_pre={shared_pre} "
        f"after shared-origin trim, got {collision_times}"
    )
    camera_offsets = collision_times

    try:
        build_grid(trimmed, out_path, camera_offsets)
    except Exception as e:
        tqdm.write(f"  {event_label} ERROR building grid: {e}")
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    shutil.rmtree(event_dir, ignore_errors=True)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2x2 grid videos (front/left/right/rear) from rover collision events")
    parser.add_argument("--events-json", default=str(Path(__file__).parent / "script_job_c5147e2c1ebeb9d8ecfc77f254da49dd_0.json"))
    parser.add_argument("--work-dir",    default=str(Path(__file__).parent / "work"))
    parser.add_argument("--output-dir",  default=str(Path(__file__).parent / "grid_videos"))
    parser.add_argument("--max-events",  type=int, default=None,
                        help="Keep pulling events from --events-json until this many grid "
                             "videos have actually been produced (events that fail, e.g. "
                             "missing GCS segments, don't count against this and are skipped "
                             "over). Default: process every event in the file.")
    parser.add_argument("--event-workers", type=int, default=4)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    events = load_events(args.events_json)
    print(f"Loaded {len(events)} events from {args.events_json}")

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    skip_existing = not args.no_skip_existing
    target = args.max_events

    def _process_one(event: dict):
        try:
            result = process_event(event, args.work_dir, args.output_dir, skip_existing)
            return result, None
        except Exception as e:
            return None, e

    event_iter = iter(events)
    attempted = 0
    succeeded = 0

    with ThreadPoolExecutor(max_workers=args.event_workers) as pool:
        pending: dict = {}

        def submit_next() -> bool:
            event = next(event_iter, None)
            if event is None:
                return False
            pending[pool.submit(_process_one, event)] = event
            return True

        def top_up() -> None:
            # Keep only as many tasks in flight as are actually needed: enough
            # to cover --max-events assuming everything still pending
            # succeeds, so a burst of successes doesn't overshoot the target
            # with already-launched extra work.
            while len(pending) < args.event_workers:
                if target and succeeded + len(pending) >= target:
                    break
                if not submit_next():
                    break

        top_up()

        pbar = tqdm(total=target, desc="Videos", unit="video") if target else tqdm(desc="Events processed", unit="event")
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    event = pending.pop(future)
                    attempted += 1
                    result, err = future.result()
                    if err is not None:
                        tqdm.write(f"ERROR rover={event.get('rover_id') or event.get('robot_id')}: {err}")
                    elif result:
                        succeeded += 1
                        if target:
                            pbar.update(1)
                    if not target:
                        pbar.update(1)

                if target and succeeded >= target:
                    break
                top_up()
        finally:
            pbar.close()

    if target and succeeded < target:
        print(f"\nDone. Only {succeeded}/{target} grid videos produced -- ran out of events "
              f"after attempting all {attempted} available.")
    else:
        print(f"\nDone. {succeeded} grid videos written to {args.output_dir} "
              f"(attempted {attempted} event(s)).")


if __name__ == "__main__":
    main()
