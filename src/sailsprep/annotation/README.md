# annotation

A FastAPI browser tool for reviewing SAILS videos and assigning frame-level
behavior labels (Locomotion and Repetitive Motor Movements).

## Files

```
annotation.py   FastAPI app: video list/status API, annotation load/save endpoints
index.html      single-page frontend served at "/"
```

## Configuration

Paths are set as constants near the top of `annotation.py` and must be
updated for your environment before running:

- `VIDEO_DIR` — directory of source videos, served at `/videos`
- `CSV_FILE` — the video list CSV; its first column is the video path and it
  may include a `status` column and `Locomotion`/`Repetitive_Motor_Movements`
  reference columns
- `ANNOTATION_DIR` — where existing `<video>_actions_corrected.csv`
  annotation files are read from if not already present in `OUTPUT_DIR`
- `OUTPUT_DIR` — where new/edited annotations are saved

## Running

```bash
uvicorn sailsprep.annotation.annotation:app --reload
```

This serves the annotation UI at `http://127.0.0.1:8000`.

## API

- `GET /` — serves `index.html`
- `GET /api/config` — label categories and reference lists
- `GET /api/videos` — video list with status
- `GET /api/video/{video_index}` — info for one video
- `GET /api/annotations/{video_index}` — existing annotation for a video, if any
- `POST /api/save` — save frame-level annotations for a video to
  `OUTPUT_DIR/<video>_actions_corrected.csv`
- `POST /api/update-status` — update a video's status in `CSV_FILE`
- `GET /api/first-unprocessed` — index of the first video with no saved status

## Labels

Two label dictionaries are defined and both are returned by `/api/config`:

- `ACTION_CATEGORIES` (used for the reference values loaded from `CSV_FILE`):
  `Repetitive_Motor_Movements`: Hands flapping, Jumping, Spinning, Rocking;
  `Locomotion`: Crawling, Cruising, Walking, Running, Vehicle
- `ALL_CATEGORIES` (the full annotation label set): `Locomotion`: Crawling,
  Cruising, Walking, Running, Vehicle; `Repetitive_Motor_Movements`: Hands
  flapping, One hand flap, Jumping, Spinning, Rocking

`CSV_FILE` data is cached in memory and only re-read when the file's
modification time changes.
