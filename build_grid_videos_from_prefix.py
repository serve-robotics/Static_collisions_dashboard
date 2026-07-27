#!/usr/bin/env python3
"""
build_grid_videos_from_prefix.py

Like build_grid_videos.py, but for query results that don't carry an explicit
camera_files list -- instead each event gives a GCS prefix (a per-minute
"directory") that its camera footage should live under:

    {
      "event_id": "...",
      "event_ts": "2026-07-23 23:56:11.000000 UTC",
      "robot_id": "1903",
      "projected_camera_prefix":   "gs://rovers-data/cameras/s1-8-0-1903/2026/07/23/23.56/",
      "candidate_camera_prefixes": ["gs://.../23.55/", "gs://.../23.56/", "gs://.../23.57/"]
    }

For each event this script:
  1. Lists every camera segment under the event's candidate prefixes (the
     event can land right at a minute-folder boundary, so the projected
     prefix alone may not contain all the footage needed -- see the segment
     listing in the docstring below).
  2. Keeps only the segments whose filename timestamp falls within the
     window actually needed to build the cropped clip (event_ts ± HALF_WINDOW,
     with a generous look-back buffer since segment length isn't known until
     downloaded).
  3. Downloads + merges per camera, then applies the exact same shared-trim-
     origin camera-sync logic and one-frame red impact highlight as
     build_grid_videos.py / ex.py (build_clip_dataset.py), so the event lands
     at the identical offset in every camera's clip.
  4. Stitches front/left/right/rear into a single 2x2 grid video, cropped to
     ~2*HALF_WINDOW seconds (default: ~10s total).

Output: {output_dir}/{rover_id}_{sanitized_event_ts}_{event_id}.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import timedelta
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from build_grid_videos import (
    CAMERAS,
    GCS_BUCKET,
    build_grid,
    camera_from_filename,
    download_segments_concurrent,
    get_storage_client,
    load_events,
    merge_segments,
    normalize_gcs_path,
    parse_cam_ts,
    parse_event_ts,
    sanitize,
    trim_video,
)

# ── constants ────────────────────────────────────────────────────────────────
HALF_WINDOW             = 5.0   # seconds before/after event_ts -- total clip ~= 2*HALF_WINDOW (~10s)
SEGMENT_LOOKBACK_BUFFER = 15.0  # generous guess at max segment duration: a segment that STARTS
                                # this far before our window could still extend into it, so include it


# ── GCS prefix listing ───────────────────────────────────────────────────────

def list_segments_under_prefixes(prefixes: list[str]) -> list[str]:
    """Return every blob name (deduped, in listing order) found under any of
    the given gs:// prefixes."""
    client = get_storage_client()
    seen: set[str] = set()
    names: list[str] = []
    for raw_prefix in prefixes:
        object_prefix = normalize_gcs_path(raw_prefix)
        for blob in client.list_blobs(GCS_BUCKET, prefix=object_prefix):
            if blob.name not in seen:
                seen.add(blob.name)
                names.append(blob.name)
    return names


# ── per-event pipeline ────────────────────────────────────────────────────────

def process_event(
    event: dict,
    work_dir: str,
    output_dir: str,
    skip_existing: bool = True,
    half_window: float = HALF_WINDOW,
) -> Optional[str]:
    rover_id     = str(event.get("robot_id") or event.get("rover_id") or "unknown")
    event_ts_str = str(event.get("event_ts", ""))
    event_id     = str(event.get("event_id") or event.get("impact_id") or "unknown")

    projected_prefix  = event.get("projected_camera_prefix")
    candidate_prefixes = event.get("candidate_camera_prefixes") or []
    prefixes = list(dict.fromkeys(p for p in [projected_prefix, *candidate_prefixes] if p))

    sanitized_ts = sanitize(event_ts_str)
    event_label  = f"{rover_id}_{sanitized_ts}_{event_id}"
    event_dir    = os.path.join(work_dir, event_label)
    out_path     = os.path.join(output_dir, f"{event_label}.mp4")

    if skip_existing and os.path.exists(out_path):
        return out_path

    if not prefixes:
        tqdm.write(f"  {event_label}: no camera_prefix on event, skipping")
        return None

    try:
        event_ts = parse_event_ts(event_ts_str)
    except ValueError:
        tqdm.write(f"  {event_label}: unparseable event_ts {event_ts_str!r}, skipping")
        return None

    try:
        blob_names = list_segments_under_prefixes(prefixes)
    except Exception as e:
        tqdm.write(f"  {event_label} ERROR listing GCS prefix(es): {e}")
        return None

    # ── Select only the segments that could plausibly cover our crop window
    window_start = event_ts - timedelta(seconds=half_window + SEGMENT_LOOKBACK_BUFFER)
    window_end   = event_ts + timedelta(seconds=half_window)

    by_camera: dict[str, list[str]] = {c: [] for c in CAMERAS}
    for name in blob_names:
        cam = camera_from_filename(name.split("/")[-1])
        if not cam or cam not in by_camera:
            continue
        ts = parse_cam_ts(name)
        if ts is None or not (window_start <= ts <= window_end):
            continue
        by_camera[cam].append(name)
    for cam in by_camera:
        by_camera[cam].sort()

    if not any(by_camera.values()):
        tqdm.write(f"  {event_label}: no camera segments found near event_ts, skipping")
        return None

    downloaded = download_segments_concurrent(by_camera, event_dir)

    # ── Phase A: per camera, merge + compute collision_in_merged (no trim yet;
    # trimming needs the shared origin computed in Phase B). Exact mirror of
    # build_grid_videos.py / build_clip_dataset.py's (ex.py) process_event.
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
    # event timestamp lands at the SAME offset in every camera's trimmed clip
    # (see build_grid_videos.py's Phase B comment for the full rationale).
    shared_pre = min([half_window] + [t for _, t in merged_info.values()])
    if shared_pre < half_window - 0.05:
        constraining = [c for c, (_, t) in merged_info.items() if abs(t - shared_pre) < 1e-6]
        tqdm.write(f"  {event_label}: lead-in reduced to {shared_pre:.3f}s "
                   f"(from {half_window}s) by camera(s) {constraining}")

    # ── Phase C: per camera, trim using the shared origin
    trimmed: dict[str, str] = {}
    collision_times: dict[str, float] = {}
    for cam, (merged_path, collision_in_merged) in merged_info.items():
        trimmed_path = os.path.join(event_dir, f"{cam}_trimmed.mp4")
        trim_start = collision_in_merged - shared_pre  # always >= 0 by construction
        trim_end   = collision_in_merged + half_window
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
    parser = argparse.ArgumentParser(
        description="Build 2x2 grid videos (front/left/right/rear) from a query result JSON "
                    "that gives per-event GCS camera-prefix directories instead of camera_files",
    )
    parser.add_argument("--events-json", default=str(Path(__file__).parent / "W_QUERY.json"))
    parser.add_argument("--work-dir",    default=str(Path(__file__).parent / "work"))
    parser.add_argument("--output-dir",  default=str(Path(__file__).parent / "grid_videos"))
    parser.add_argument("--half-window", type=float, default=HALF_WINDOW,
                        help="Seconds before/after event_ts to keep -- total clip length is "
                             "~2x this value (default: 5.0s => ~10s clips).")
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
            result = process_event(event, args.work_dir, args.output_dir, skip_existing, args.half_window)
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
                        tqdm.write(f"ERROR rover={event.get('robot_id') or event.get('rover_id')}: {err}")
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
