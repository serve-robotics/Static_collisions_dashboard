#!/usr/bin/env python3
"""
build_clip_dataset.py

Builds a HuggingFace sliding-window video-clip dataset from rover impact
events stored in impact_tp.json (label=1) and impact_fp.json (label=0).

Pipeline per event:
  download .ts segments → merge per camera → trim to ±10s window around
  collision → slide 1.6-second windows at 1-frame (0.1s) stride → extract
  one MP4 clip per camera per window → emit one HF Dataset row per window.

Output structure:
  {work_dir}/
    clips/{source}_{rover_id}_{sanitized_ts}/
      meta.json
      front/  0.000.mp4  0.100.mp4  ...  (1.6s each)
      rear/   ...
      left/   ...
      right/  ...
    dataset/   ← dataset.save_to_disk()
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from google.cloud import storage
from datasets import Dataset
from tqdm import tqdm

# ── binary paths ──────────────────────────────────────────────────────────────
FFMPEG_BIN  = os.environ.get("FFMPEG_BIN")  or shutil.which("ffmpeg")  or "/usr/bin/ffmpeg"
FFPROBE_BIN = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "/usr/bin/ffprobe"

# ── constants ─────────────────────────────────────────────────────────────────
CAMERAS       = ["front", "rear", "left", "right"]
GCS_BUCKET    = "rovers-data"
GCS_PREFIX    = "cameras/"
WINDOW_DUR    = 1.6    # seconds
PRE_COLLISION = 10.0   # seconds before event_ts to start trimmed video
FPS           = 10     # verified via ffprobe on sample rover .ts clip
STRIDE        = 1.0 / FPS  # 0.1 seconds
MAX_WORKERS   = 4      # parallel ffmpeg workers for clip extraction

# ── GCS helpers ───────────────────────────────────────────────────────────────

_storage_client: Optional[storage.Client] = None
_storage_client_lock = threading.Lock()


def get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        with _storage_client_lock:
            if _storage_client is None:  # re-check: another thread may have won the race
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

    # Download to a uniquely-named temp path, then rename atomically on
    # success. Needed now that downloads run concurrently (across cameras
    # within an event, and across events) -- two workers wanting the same
    # segment (e.g. two nearby events sharing raw footage) could otherwise
    # both write to the same path at once, corrupting it or racing a
    # partial file in as "already downloaded". The unique per-attempt
    # suffix (pid + thread id) means concurrent attempts at the same
    # `local_path` never share a temp file with each other.
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
    """Download every segment across every camera for ONE event concurrently
    (network-bound: this is the single biggest per-event speedup, turning what
    was a fully sequential per-segment, per-camera loop into one wave of
    concurrent GCS round-trips). Mirrors the annotation tool's
    _download_all_segments. Returns {cam: [local_path, ...]} in each camera's
    original (sorted) order, skipping any segment whose download failed."""
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

def parse_event_ts(s) -> datetime:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S.%f UTC").replace(
        tzinfo=timezone.utc,
    )


def resolve_annotation(annotations: Optional[list[dict]]) -> tuple[Optional[bool], Optional[datetime]]:
    """
    Reconcile a per-event `annotations` list (as found in the combined
    FT_Dataset_* format) into a single (has_impact, impact_dt) pair.

    Picks the *latest* annotation (by updated_at, falling back to
    created_at) among those that are not skipped and have a non-null
    has_impact. Returns (None, None) if no usable annotation exists.
    """
    usable = [
        a for a in (annotations or [])
        if not a.get("skipped") and a.get("has_impact") is not None
    ]
    if not usable:
        return None, None

    def _sort_key(a: dict) -> str:
        return a.get("updated_at") or a.get("created_at") or ""

    latest = max(usable, key=_sort_key)
    has_impact = bool(latest["has_impact"])

    impact_dt = None
    if has_impact and latest.get("impact_timestamp_utc"):
        impact_dt = datetime.fromisoformat(latest["impact_timestamp_utc"])
        if impact_dt.tzinfo is None:
            impact_dt = impact_dt.replace(tzinfo=timezone.utc)

    return has_impact, impact_dt


def parse_cam_ts(local_path: str) -> Optional[datetime]:
    m = re.search(
        r"(front|rear|left|right)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-(\d{3})\.",
        os.path.basename(local_path),
    )
    if not m:
        return None
    dt = datetime.strptime(m.group(2), "%Y-%m-%d-%H-%M-%S").replace(
        tzinfo=timezone.utc,
    )
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
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


# ── windowing ─────────────────────────────────────────────────────────────────

def get_windows(
    video_path: str,
    collision_time: float,
    video_duration: float,
) -> list[dict]:
    """
    Slice a video into overlapping 2-second windows.
    A window is POSITIVE if it contains the collision timestamp.

    Args:
        collision_time:  exact timestamp of impact in seconds (use -1.0 for FP events)
        video_duration:  total length of video in seconds
    """
    windows = []
    t = 0.0

    while t + WINDOW_DUR <= video_duration:
        window_end  = t + WINDOW_DUR
        is_positive = t <= collision_time <= window_end

        windows.append({
            "video_path": video_path,
            "start_sec":  t,
            "label":      int(is_positive),
        })
        t = round(t + STRIDE, 6)  # round to avoid float drift

    return windows


def get_windows_centered(
    video_path: str,
    ref_time: float,
    video_duration: float,
    label: bool,
) -> list[dict]:
    """
    Generate one 1.6s window per frame-index 2..13 (of 0..15) such that
    `ref_time` always lands strictly inside the clip - never in the first
    or last 2 frames. All returned windows share `label`.
    """
    total_frames = int(round(WINDOW_DUR * FPS))  # 16
    windows = []
    for frame_idx in range(2, total_frames - 2):  # 2..13 inclusive (16-frame clip)
        start = round(ref_time - frame_idx * STRIDE, 6)
        if start < 0 or start + WINDOW_DUR > video_duration:
            continue
        windows.append({
            "video_path": video_path,
            "start_sec":  start,
            "label":      int(label),
        })
    return windows


# ── clip extraction worker (top-level for picklability) ───────────────────────

def extract_window_clip(args: tuple) -> Optional[dict]:
    """Extract a single 2-second window clip. Runs in a subprocess worker."""
    trimmed_path, start_sec, out_path, camera = args
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-i", trimmed_path,
        "-ss", str(start_sec),
        "-t", str(WINDOW_DUR),
        "-y", "-loglevel", "error",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None
    return {"camera": camera, "start_sec": start_sec, "path": out_path}


# ── utilities ─────────────────────────────────────────────────────────────────

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def parse_camera_files(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [x.strip() for x in s.split(",") if x.strip()]


# ── per-event pipeline ────────────────────────────────────────────────────────

def process_event(
    event: dict,
    source: str,
    work_dir: str,
    skip_existing: bool = True,
    window_strategy: str = "sliding",
    download_workers: int = 8,
) -> list[dict]:
    """
    Full pipeline for one event. Returns list of HF dataset row dicts,
    one per window (all cameras combined per window).

    If `event` carries an `annotations` list (combined FT_Dataset_* format),
    the window label/reference timestamp is resolved from the latest usable
    annotation via `resolve_annotation`; otherwise (legacy impact_tp/fp.json
    format) the label falls back to `source` and the reference timestamp is
    `event_ts`, exactly as before.
    """
    rover_id      = str(event.get("rover_id", "unknown"))
    event_ts_str  = str(event.get("event_ts", ""))
    camera_files  = parse_camera_files(event.get("camera_files", []))

    annotations = event.get("annotations")
    if annotations is not None:
        has_impact, impact_dt = resolve_annotation(annotations)
        if has_impact is None:
            tqdm.write("  No usable annotation, skipping event.")
            return []
    else:
        has_impact, impact_dt = None, None

    # Reference collision datetime used for trim/collision-time math below:
    # the annotated impact instant when available, else the top-level event_ts.
    event_ts = impact_dt or parse_event_ts(event_ts_str)

    sanitized_ts = sanitize(event_ts_str)
    event_label  = f"{source}_{rover_id}_{sanitized_ts}"
    event_dir    = os.path.join(work_dir, "clips", event_label)
    os.makedirs(event_dir, exist_ok=True)

    meta_path = os.path.join(event_dir, "meta.json")

    # ── Resume: load from sidecar if already processed ────────────────────
    collision_times: dict[str, float] = {}
    clamped_info: dict[str, bool] = {}

    if skip_existing and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        collision_times = meta.get("collision_times", {})
        clamped_info    = meta.get("clamped", {})
        tqdm.write(f"  Resuming from meta.json ({len(collision_times)} camera(s))")
    else:
        # ── Group GCS paths by camera ──────────────────────────────────────
        by_camera: dict[str, list[str]] = {c: [] for c in CAMERAS}
        for cf in camera_files:
            cam = camera_from_filename(cf.split("/")[-1])
            if cam and cam in by_camera:
                by_camera[cam].append(cf)
        for cam in by_camera:
            by_camera[cam].sort()

        # ── Phase A: per camera, download + merge, compute collision_in_merged.
        # No trimming yet -- trimming needs the shared_pre computed in Phase B
        # below. (Deliberately not skipping cameras whose trimmed video already
        # exists here: recomputing collision_in_merged for every camera on
        # every non-resumed pass is what makes the shared-origin trim in Phase
        # B correct. GCS downloads are still cache-hit via download_from_gcs's
        # content-hash cache, so this only redoes cheap local merge/ffprobe
        # work, not network transfer.)
        merged_info: dict[str, tuple[str, float]] = {}  # cam -> (merged_path, collision_in_merged)

        # Download every segment across every camera concurrently up front --
        # network-bound, so this is the single biggest per-event speedup over
        # the old fully-sequential per-camera, per-segment loop.
        downloaded = download_segments_concurrent(by_camera, event_dir, max_workers=download_workers)

        for cam in CAMERAS:
            if not by_camera[cam]:
                continue

            cam_tmp_dir = os.path.join(event_dir, "tmp", cam)
            local_segments = downloaded.get(cam, [])
            if not local_segments:
                continue

            # Merge
            try:
                merged_path = merge_segments(cam, local_segments, cam_tmp_dir)
            except Exception as e:
                tqdm.write(f"  ERROR [{cam}] merge failed: {e}")
                continue
            if not merged_path:
                continue

            # Compute collision time in merged video
            seg_times = [parse_cam_ts(p) for p in local_segments]
            seg_times = [t for t in seg_times if t is not None]
            if not seg_times:
                tqdm.write(f"  WARNING [{cam}] cannot parse segment timestamps, skipping")
                os.remove(merged_path)
                continue
            min_seg_time = min(seg_times)
            collision_in_merged = (event_ts - min_seg_time).total_seconds()
            merged_info[cam] = (merged_path, collision_in_merged)

        # ── Phase B: shared trim origin across all cameras with footage ────
        # Trimming each camera independently with trim_start = max(0.0,
        # collision_in_merged - PRE_COLLISION) (the old behavior) lets a
        # camera whose footage doesn't reach back PRE_COLLISION seconds
        # before the event "clamp" to trim_start=0 while other cameras don't
        # -- so frame 0 of that camera's trimmed video lands at a different
        # real-world instant than frame 0 of an unclamped camera. Later
        # window extraction applies the *same* start_sec to every camera's
        # independently-trimmed video, silently desyncing them by up to
        # ~1s within a single labeled window (observed directly on a real
        # event: front/left impact at frame 7 of 16, rear/right at frame 15).
        # Using one shared_pre -- the tightest lead-in actually available
        # across the group -- for every camera's trim guarantees
        # collision_in_trimmed comes out identical for all cameras, at the
        # cost of a uniformly shorter lead-in whenever any single camera is
        # the constraining one.
        clamped_info = {
            cam: (collision_in_merged < PRE_COLLISION - 1e-6)
            for cam, (_, collision_in_merged) in merged_info.items()
        }
        shared_pre = min([PRE_COLLISION] + [t for _, t in merged_info.values()])
        if shared_pre < PRE_COLLISION - 0.05:
            constraining = [c for c, (_, t) in merged_info.items() if abs(t - shared_pre) < 1e-6]
            tqdm.write(f"  {event_label}: lead-in reduced to {shared_pre:.3f}s "
                       f"(from {PRE_COLLISION}s) by camera(s) {constraining}")

        # ── Phase C: per camera, trim using the shared origin ──────────────
        for cam, (merged_path, collision_in_merged) in merged_info.items():
            trimmed_path = os.path.join(event_dir, f"{cam}_trimmed.mp4")
            trim_start = collision_in_merged - shared_pre  # always >= 0 by construction
            trim_end   = collision_in_merged + WINDOW_DUR
            collision_in_trimmed = collision_in_merged - trim_start  # == shared_pre, every camera

            try:
                trim_video(merged_path, trimmed_path, trim_start, trim_end)
            except Exception as e:
                tqdm.write(f"  ERROR [{cam}] trim failed: {e}")
                os.remove(merged_path)
                continue

            collision_times[cam] = collision_in_trimmed

            # Clean up intermediate files
            os.remove(merged_path)
            cam_tmp_dir = os.path.join(event_dir, "tmp", cam)
            for f in Path(cam_tmp_dir).iterdir():
                if f.suffix in (".ts",) or f.name.endswith(".concat.txt"):
                    f.unlink(missing_ok=True)

        assert all(abs(t - shared_pre) < 1e-6 for t in collision_times.values()), (
            f"{event_label}: collision_times should all equal shared_pre={shared_pre} "
            f"after shared-origin trim, got {collision_times}"
        )

        # Write sidecar
        with open(meta_path, "w") as f:
            json.dump({
                "collision_times":    collision_times,
                "shared_pre_seconds": shared_pre,
                "clamped":            clamped_info,
                "source":             source,
            }, f)

    # ── Collect trimmed video paths ────────────────────────────────────────
    trimmed_paths: dict[str, str] = {}
    for cam in CAMERAS:
        p = os.path.join(event_dir, f"{cam}_trimmed.mp4")
        if os.path.exists(p):
            trimmed_paths[cam] = p

    if not trimmed_paths:
        tqdm.write("  No trimmed videos available, skipping event.")
        return []

    # ── Get duration from first available camera ───────────────────────────
    # Prefer an unclamped camera as reference. With the shared-origin trim
    # above, collision_times already agree across cameras so this no longer
    # affects correctness -- kept as a diagnostic preference (and parity with
    # the annotation tool's reference-camera selection) in case that guarantee
    # ever regresses upstream.
    unclamped_cams = [c for c in trimmed_paths if not clamped_info.get(c, False)]
    ref_cam = unclamped_cams[0] if unclamped_cams else next(iter(trimmed_paths))
    try:
        trimmed_duration = get_video_duration(trimmed_paths[ref_cam])
    except Exception as e:
        tqdm.write(f"  WARNING duration detection failed: {e}; using PRE+POST fallback")
        trimmed_duration = PRE_COLLISION + WINDOW_DUR

    # Resolve whether this event is positive: prefer the resolved annotation
    # (has_impact), falling back to the legacy source-based gate (tp/fp).
    is_positive = has_impact if has_impact is not None else (source == "tp")

    # ── Generate window list ───────────────────────────────────────────────
    if window_strategy == "centered":
        ref_ts = collision_times.get(ref_cam)
        if ref_ts is None:
            tqdm.write("  No collision time recorded for reference camera, skipping event.")
            return []
        windows = get_windows_centered(trimmed_paths[ref_cam], ref_ts, trimmed_duration, is_positive)
    else:
        ref_collision = collision_times.get(ref_cam, -1.0) if is_positive else -1.0
        windows = get_windows(trimmed_paths[ref_cam], ref_collision, trimmed_duration)
    if not windows:
        return []

    # ── Build extraction task list ─────────────────────────────────────────
    extract_tasks: list[tuple] = []
    for window in windows:
        start_sec  = window["start_sec"]
        clip_name  = f"{start_sec:.3f}.mp4"
        for cam, trimmed_path in trimmed_paths.items():
            cam_clip_dir = os.path.join(event_dir, cam)
            out_path     = os.path.join(cam_clip_dir, clip_name)
            if skip_existing and os.path.exists(out_path):
                continue
            extract_tasks.append((trimmed_path, start_sec, out_path, cam))

    # ── Run clip extraction in parallel ───────────────────────────────────
    clip_map: dict[tuple[float, str], str] = {}

    # Pre-populate with any already-extracted clips (resume case)
    for cam in trimmed_paths:
        cam_clip_dir = os.path.join(event_dir, cam)
        if os.path.isdir(cam_clip_dir):
            for clip_file in Path(cam_clip_dir).glob("*.mp4"):
                try:
                    s = float(clip_file.stem)
                    clip_map[(round(s, 6), cam)] = str(clip_file)
                except ValueError:
                    pass

    if extract_tasks:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(extract_window_clip, task): task for task in extract_tasks}
            with tqdm(total=len(extract_tasks), desc="  clips", unit="clip", leave=False) as clip_bar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        tqdm.write(f"  WARNING clip extraction error: {e}")
                    else:
                        if result:
                            key = (round(result["start_sec"], 6), result["camera"])
                            clip_map[key] = result["path"]
                    clip_bar.update(1)

    # ── Build dataset rows ─────────────────────────────────────────────────
    rows: list[dict] = []
    for window in windows:
        s = round(window["start_sec"], 6)
        rows.append({
            "front_clip": clip_map.get((s, "front"),  ""),
            "rear_clip":  clip_map.get((s, "rear"),   ""),
            "left_clip":  clip_map.get((s, "left"),   ""),
            "right_clip": clip_map.get((s, "right"),  ""),
            "start_sec":  window["start_sec"],
            "label":      window["label"],
            "rover_id":   rover_id,
            "event_ts":   event_ts_str,
            "source":     source,
        })

    # ── Clean up trimmed videos ────────────────────────────────────────────
    # Only when skip_existing is off: the resume branch at the top of this
    # function (`skip_existing and os.path.exists(meta_path)`) assumes these
    # trimmed videos still exist. Deleting them unconditionally after every
    # run -- including the first, non-resumed one -- silently made any later
    # re-run over the same --work-dir return zero rows for already-completed
    # events (meta.json existed, so it resumed; but the trimmed videos it
    # resumed "from" were already gone).
    if not skip_existing:
        for trimmed_path in trimmed_paths.values():
            if os.path.exists(trimmed_path):
                os.remove(trimmed_path)

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build HuggingFace sliding-window clip dataset")
    _root = Path(__file__).parent.parent
    parser.add_argument("--tp-json",      default=str(_root / "data" / "impact_tp.json"))
    parser.add_argument("--fp-json",      default=str(_root / "data" / "impact_fp.json"))
    parser.add_argument("--events-json",  default=None,
                        help="Combined annotated events file (e.g. FT_Dataset_*.json), a single "
                             "list of events each carrying an inline 'source' and 'annotations' "
                             "list. When set, --tp-json/--fp-json are ignored.")
    parser.add_argument("--window-strategy", choices=["sliding", "centered"], default=None,
                        help="'sliding': legacy 1-frame-stride sliding windows across the full "
                             "trimmed clip. 'centered': one window per frame-index 2..13 so the "
                             "reference timestamp always lands strictly inside the clip. "
                             "Default: 'sliding' for --tp-json/--fp-json, 'centered' for "
                             "--events-json.")
    parser.add_argument("--work-dir",     default=str(_root / "data" / "impact_clips"))
    parser.add_argument("--output-dir",   default=None,
                        help="Where to save HF dataset (default: {work_dir}/dataset)")
    parser.add_argument("--max-events",   type=int, default=None)
    parser.add_argument("--workers",      type=int, default=4,
                        help="Concurrent ffmpeg clip-extraction workers PER EVENT (up to 48 "
                             "tasks/event: 12 windows x 4 cameras). Raising this only speeds up "
                             "an already-fast step -- see --event-workers for the actual "
                             "per-event bottleneck (sequential downloads).")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Concurrent GCS segment downloads PER EVENT, across all 4 cameras "
                             "(network-bound I/O -- was fully sequential before).")
    parser.add_argument("--event-workers", type=int, default=4,
                        help="Concurrent EVENTS processed at once (was fully sequential before -- "
                             "the other big lever alongside --download-workers, since network "
                             "round-trip latency dominates per-event wall time).")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    global MAX_WORKERS
    args = parser.parse_args()
    MAX_WORKERS = args.workers

    work_dir   = args.work_dir
    output_dir = args.output_dir or os.path.join(work_dir, "dataset")
    os.makedirs(work_dir, exist_ok=True)

    if args.events_json:
        with open(args.events_json) as f:
            events = json.load(f)
        print(f"Loaded {len(events)} events from {args.events_json}")
        labeled_events: list[tuple[dict, str]] = [(ev, ev.get("source", "tp")) for ev in events]
        window_strategy = args.window_strategy or "centered"
    else:
        with open(args.tp_json) as f:
            tp_events = json.load(f)
        with open(args.fp_json) as f:
            fp_events = json.load(f)

        print(f"Loaded {len(tp_events)} TP events, {len(fp_events)} FP events")

        labeled_events = (
            [(ev, "tp") for ev in tp_events] +
            [(ev, "fp") for ev in fp_events]
        )
        window_strategy = args.window_strategy or "sliding"

    if args.max_events:
        labeled_events = labeled_events[:args.max_events]
        tqdm.write(f"Limiting to {args.max_events} event(s)")

    peak_ffmpeg = args.event_workers * args.workers
    print(f"Concurrency: {args.event_workers} event(s) in flight at once, each with up to "
          f"{args.download_workers} concurrent segment downloads and {args.workers} concurrent "
          f"ffmpeg clip-extraction workers (peak ~{peak_ffmpeg} concurrent ffmpeg processes "
          f"during overlapping clip-extraction bursts across events).")

    all_rows: list[dict] = []
    all_rows_lock = threading.Lock()

    def _process_one(event_source: tuple[dict, str]):
        event, source = event_source
        rover_id = event.get("rover_id", "?")
        try:
            rows = process_event(
                event, source, work_dir,
                skip_existing=args.skip_existing,
                window_strategy=window_strategy,
                download_workers=args.download_workers,
            )
            return rover_id, rows, None
        except Exception as e:
            return rover_id, [], e

    pbar = tqdm(total=len(labeled_events), desc="Events", unit="event")
    with ThreadPoolExecutor(max_workers=args.event_workers) as pool:
        futures = [pool.submit(_process_one, ev_src) for ev_src in labeled_events]
        for future in as_completed(futures):
            rover_id, rows, err = future.result()
            if err is not None:
                tqdm.write(f"ERROR rover={rover_id}: {err}")
            else:
                with all_rows_lock:
                    all_rows.extend(rows)
            pbar.set_postfix(rover=rover_id, windows=len(all_rows))
            pbar.update(1)
    pbar.close()

    print(f"\nTotal rows: {len(all_rows):,}")
    if not all_rows:
        tqdm.write("No rows produced; exiting.")
        return

    print(f"Saving HuggingFace dataset to {output_dir} …")
    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
