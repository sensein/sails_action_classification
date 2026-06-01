#!/usr/bin/env python3
"""
Rename H5 bounding-box files to BIDS-style names.

Old: A1H3H9Y3T1_04.04.2020_720p__COI_bboxes.h5
New: sub-A1H3H9Y3T1_ses-01_task-dailyroutine_run-01_desc-processed_COI_bboxes.h5

592 files will be renamed. 29 cannot be matched (UUIDs / unknown IMG numbers /
long Facebook received_ IDs) and will be left untouched with a report saved.

Usage
-----
    # Preview only (safe - nothing is changed):
    python rename_h5_to_bids_final.py --dry-run

    # Actually rename:
    python rename_h5_to_bids_final.py
"""

import re
import csv
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
H5_DIR    = "/orcd/scratch/bcs/001/sensein/sails/Tracking_Output/bboxes_yolo_h5/"
CSV_PATH  = "/home/aparnabg/orcd/scratch/Automatic_Labeling/csv2_with_bids_file_path.csv"
LOG_PATH  = "/home/aparnabg/orcd/scratch/all_project_files/sam3_video/h5_rename_log.txt"       # full log of every action
SKIP_PATH = "/home/aparnabg/orcd/scratch/all_project_files/sam3_video/h5_rename_skipped.txt"   # just the no-match files
DRY_RUN   = "--dry-run" in sys.argv
# ─────────────────────────────────────────────────────────────────────────────

# H5 filename pattern:  <ID>_<stem>__COI_bboxes.h5
# (double underscore separates stem from COI_bboxes)
SPLIT_RE = re.compile(r'^(?P<id>[A-Z0-9]+)_(?P<raw>.+)__COI_bboxes\.h5$')
RES_RE   = re.compile(r'_\d+p$')   # trailing _720p, _1080p, _480p etc.

def parse_h5(fname):
    """Return (subject_id, clean_stem) or (None, None) if unparseable."""
    m = SPLIT_RE.match(fname)
    if not m:
        return None, None
    clean = RES_RE.sub("", m.group("raw"))
    return m.group("id"), clean

def bids_to_new_h5_name(bids_path):
    """
    Convert BidsProcessed path to new H5 filename.
    e.g. .../sub-X_ses-01_task-foo_run-02_desc-processed_beh.mp4
         -> sub-X_ses-01_task-foo_run-02_desc-processed_COI_bboxes.h5
    """
    stem = Path(bids_path).stem   # strips .mp4
    # replace trailing _beh with _COI_bboxes
    new_stem = re.sub(r'_beh$', '_COI_bboxes', stem)
    return new_stem + ".h5"

# ── Load CSV ──────────────────────────────────────────────────────────────────
lookup = {}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        subj = row["ID"].strip()
        stem = Path(row["FileName"].strip()).stem
        bids = row.get("BidsProcessed", "").strip()
        if subj and stem and bids:
            lookup[(subj, stem)] = bids

print(f"CSV loaded: {len(lookup)} (ID, FileName_stem) entries")
print(f"Mode: {'DRY RUN — no files will be changed' if DRY_RUN else '*** LIVE — files WILL be renamed ***'}\n")

# ── Process H5 files ──────────────────────────────────────────────────────────
h5_files = sorted(Path(H5_DIR).glob("*_bboxes.h5"))
print(f"Found {len(h5_files)} H5 files\n")

renamed   = []
skipped   = []
collision = []
bad_parse = []

for h5 in h5_files:
    subj, stem = parse_h5(h5.name)

    if subj is None:
        bad_parse.append(h5.name)
        print(f"[BAD_PARSE] {h5.name}")
        continue

    bids = lookup.get((subj, stem))

    if bids is None:
        skipped.append((h5.name, subj, stem))
        print(f"[NO_MATCH]  {h5.name}")
        continue

    new_name = bids_to_new_h5_name(bids)
    new_path = h5.parent / new_name

    if new_path == h5:
        print(f"[SAME]      {h5.name}  (already correct)")
        renamed.append((h5.name, new_name, "already_correct"))
        continue

    if new_path.exists():
        print(f"[COLLISION] {h5.name}  ->  {new_name}  (target exists!)")
        collision.append((h5.name, new_name))
        continue

    if DRY_RUN:
        print(f"[DRY_RUN]   {h5.name}\n            -> {new_name}")
    else:
        h5.rename(new_path)
        print(f"[RENAMED]   {h5.name}\n            -> {new_name}")

    renamed.append((h5.name, new_name, "renamed"))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Renamed (or would rename) : {len([r for r in renamed if r[2]=='renamed'])}")
print(f"  Already correct           : {len([r for r in renamed if r[2]=='already_correct'])}")
print(f"  No CSV match (skipped)    : {len(skipped)}")
print(f"  Collision (target exists) : {len(collision)}")
print(f"  Bad parse                 : {len(bad_parse)}")
if DRY_RUN:
    print(f"\n*** DRY RUN — nothing was changed ***")

# ── Write logs ────────────────────────────────────────────────────────────────
with open(LOG_PATH, "w") as f:
    f.write(f"H5 RENAME LOG — {'DRY RUN' if DRY_RUN else 'LIVE'}\n\n")
    f.write("--- RENAMED ---\n")
    for old, new, status in renamed:
        f.write(f"{status.upper()}: {old}\n  -> {new}\n")
    f.write("\n--- COLLISIONS ---\n")
    for old, new in collision:
        f.write(f"COLLISION: {old}\n  -> {new}\n")
    f.write("\n--- NO MATCH ---\n")
    for h5_name, subj, stem in skipped:
        f.write(f"NO_MATCH: {h5_name}\n  id={subj}  stem={stem}\n")
    f.write("\n--- BAD PARSE ---\n")
    for f_name in bad_parse:
        f.write(f"BAD_PARSE: {f_name}\n")

with open(SKIP_PATH, "w") as f:
    f.write(f"SKIPPED (no CSV match) — {len(skipped)} files\n\n")
    for h5_name, subj, stem in skipped:
        f.write(f"{h5_name}\n  id={subj}  stem={stem}\n\n")

print(f"\nFull log  -> {LOG_PATH}")
print(f"Skip list -> {SKIP_PATH}")