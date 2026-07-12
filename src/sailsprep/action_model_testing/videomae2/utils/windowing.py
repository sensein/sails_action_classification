from collections import Counter


def get_window_label(frame_to_label, ann_start, ann_end, na_label="N/A"):
    """Majority label in [ann_start, ann_end). Treats empty/nan as na_label."""
    labels = []
    for f in range(ann_start, ann_end):
        lbl = frame_to_label.get(f, na_label)
        if lbl in ("", "nan", "None"):
            lbl = na_label
        labels.append(lbl)
    if not labels:
        return na_label
    return Counter(labels).most_common(1)[0][0]
