# Static Collisions Dashboard

A local Flask dashboard for finding static-collision events, downloading the corresponding rover camera footage from Google Cloud Storage, and reviewing synchronized 2×2 camera-grid videos in a browser.

The dashboard queries BigQuery for candidate impact events, filters out events associated with rover movement or pilot control, then builds a video for each event using the front, left, right, and rear camera feeds.

## Features

- Select a calendar day and query collision events from BigQuery.
- Detect static impacts using rover-state and pilot-session context.
- Discover camera segments under Google Cloud Storage prefixes.
- Download and merge camera segments concurrently.
- Synchronize camera views around the impact timestamp.
- Highlight the impact instant with a red border.
- Stitch camera views into a 2×2 grid:

  ```text
  front | left
  ------+------
  right | rear
  ```

- Process events in bounded batches with **Load more**.
- Stream completed videos into the browser as they become available.
- Skip existing output videos by default.
- Force video regeneration from the dashboard when needed.
- Process exported event JSON files from the command line.

## Architecture

1. The user selects a day in the browser.
2. `app.py` parameterizes and executes `query.txt` against BigQuery.
3. BigQuery returns impact events and candidate camera prefixes.
4. `build_grid_videos_from_prefix.py` locates relevant camera segments in GCS.
5. Camera segments are downloaded, merged, trimmed, synchronized, and stitched with FFmpeg.
6. The Flask application serves generated MP4 files from `grid_videos/`.
7. The browser polls job status and displays completed videos with metadata.

## Repository layout

| Path | Description |
| --- | --- |
| `app.py` | Flask application, BigQuery integration, job management, and HTTP routes. |
| `templates/index.html` | Browser dashboard interface. |
| `query.txt` | BigQuery query for identifying static-collision events and camera prefixes. |
| `build_grid_videos_from_prefix.py` | Video pipeline for events discovered from GCS prefixes. |
| `build_grid_videos.py` | Video pipeline for events with explicit camera file lists. |
| `requirements.txt` | Python dependencies. |
| `work/` | Temporary per-event processing files. |
| `grid_videos/` | Generated 2×2 MP4 videos. |

## Requirements

- Python 3.10+ recommended.
- FFmpeg and FFprobe available on `PATH`.
- Access to the `serve-robotics` Google Cloud project.
- Permission to query the required BigQuery tables.
- Permission to read camera objects from the `rovers-data` bucket.

The video pipeline defaults to `/usr/bin/ffmpeg` and `/usr/bin/ffprobe` when the executables are not found on `PATH`. Override them with `FFMPEG_BIN` and `FFPROBE_BIN` if necessary.

## Installation

```bash
git clone https://github.com/serve-robotics/Static_collisions_dashboard.git
cd Static_collisions_dashboard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure Google Cloud credentials using the method appropriate for your environment. For local development, Application Default Credentials can be configured with:

```bash
gcloud auth application-default login
```

The authenticated account must be able to query BigQuery and read from the `rovers-data` bucket.

## Run the dashboard

Start the Flask server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5050
```

Customize the host and port:

```bash
python app.py --host 0.0.0.0 --port 5050
```

### Dashboard workflow

1. Open the dashboard in a browser.
2. Select a date.
3. Choose a batch size.
4. Click **Run**.
5. Monitor processing progress.
6. Click **Load more** to process the next batch.
7. Enable **Force rebuild** to regenerate existing videos.

Generated videos are saved to `grid_videos/` and served through the `/videos/<filename>` endpoint.

## HTTP API

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Renders the dashboard. |
| `POST` | `/run` | Starts a new day query and processing job. |
| `POST` | `/more` | Processes the next batch. |
| `GET` | `/status` | Returns job state and generated video metadata. |
| `GET` | `/videos/<filename>` | Serves a generated MP4 file. |

Start a job manually:

```bash
curl -X POST http://127.0.0.1:5050/run \
  -H 'Content-Type: application/json' \
  -d '{"day":"2026-07-23","batch_size":20,"force_rebuild":false}'
```

Check progress:

```bash
curl http://127.0.0.1:5050/status
```

## BigQuery query

`query.txt` reads from:

- `serve-robotics.serve_analytics.robot_events`
- `serve-robotics.serve_analytics.stg_rover_state`
- `serve-robotics.serve_analytics.pilot_robot_sessions`

The query identifies impact-detection events and keeps events that:

- have rover-state coverage near the impact;
- were stationary during the surrounding time window;
- had no local-controller authority;
- had no remote-controller authority;
- were not inside an active remote pilot session.

When used by `app.py`, the query’s `start_ts` and `end_ts` declarations are replaced with parameterized BigQuery values for the selected day.

## Command-line processing

For event exports containing explicit `camera_files`, use:

```bash
python build_grid_videos.py \
  --events-json path/to/events.json \
  --output-dir grid_videos \
  --work-dir work \
  --event-workers 4
```

For event exports containing `projected_camera_prefix` and `candidate_camera_prefixes`, use:

```bash
python build_grid_videos_from_prefix.py \
  --events-json path/to/events.json \
  --output-dir grid_videos \
  --work-dir work \
  --event-workers 4
```

Useful options:

- `--max-events N` — stop after producing `N` successful videos.
- `--event-workers N` — control concurrent event processing.
- `--no-skip-existing` — rebuild existing output videos.
- `--half-window SECONDS` — configure the retained time before and after an event in the prefix-based pipeline.

Both processors accept either a JSON array or newline-delimited JSON event export.

Generated files follow this pattern:

```text
{robot_id}_{sanitized_event_timestamp}_{event_id}.mp4
```

## Data and output conventions

- Camera footage is read from the `rovers-data` GCS bucket.
- Camera filenames are expected to identify `front`, `left`, `right`, or `rear`.
- The prefix-based pipeline searches adjacent minute prefixes to handle minute-boundary events.
- The prefix-based dashboard pipeline targets approximately five seconds before and after the event.
- The explicit-file pipeline targets approximately ten seconds before and after the event.
- Missing camera views are replaced with black cells where possible.
- Temporary event directories are removed after processing.

## Troubleshooting

### BigQuery authentication or permission errors

Verify that Application Default Credentials are configured and that the authenticated account can query the analytics tables in the `serve-robotics` project.

### GCS download failures

Confirm access to the `rovers-data` bucket and verify that the event’s camera prefixes contain segments near the event timestamp.

### FFmpeg executable not found

Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are available on `PATH`, or set explicit paths:

```bash
export FFMPEG_BIN=/path/to/ffmpeg
export FFPROBE_BIN=/path/to/ffprobe
```

### No videos are produced

Review terminal output for events skipped because of:

- missing camera segments;
- invalid timestamps;
- failed downloads;
- failed video merges;
- FFmpeg errors.

The dashboard reports attempted, successful, and failed event counts.

## Development notes

The following generated or local files are ignored by Git:

- `.venv/`
- `work/`
- `grid_videos/`
- JSON query exports
- Python cache files
- log files

The application is intended as a local operational dashboard. It does not currently include production deployment configuration, authentication, or persistent job storage.

## License

No license file is currently included in this repository.
