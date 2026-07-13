"""Shared ID/timestamp parsing helpers (exact duplicates across analysis modules)."""

import re

FPS = 15.0


def extract_pid(path):
    if not isinstance(path, str): return None
    m = re.search(r'(sub-[A-Za-z0-9]+)', path)
    return m.group(1) if m else None


def extract_session(path):
    if not isinstance(path, str): return None
    m = re.search(r'ses-(\d+)', path)
    return int(m.group(1)) if m else None


def parse_timestamps_v1(ts_str, fps=FPS):
    if not isinstance(ts_str, str): return []
    segs = []
    for part in ts_str.split(','):
        m = re.match(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)', part.strip())
        if m:
            s = int(m.group(1))*60 + int(m.group(2))
            e = int(m.group(3))*60 + int(m.group(4))
            if e > s:
                segs.append((int(s*fps), int(e*fps)))
    return segs


def parse_timestamps_v2(ts_str, fps=FPS):
    if not isinstance(ts_str, str): return []
    segs = []
    for part in ts_str.split(','):
        m = re.match(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)', part.strip())
        if m:
            s = int(m.group(1))*60 + int(m.group(2))
            e = int(m.group(3))*60 + int(m.group(4))
            if e > s: segs.append((int(s*fps), int(e*fps)))
    return segs
