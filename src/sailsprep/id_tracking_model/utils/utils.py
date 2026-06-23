import numpy as np
from numpy.typing import NDArray


def soft_nms(
    dets: NDArray[np.float64],
    iou_thr: float = 0.55,
    score_thr: float = 0.05,
    method: str = 'linear',
    sigma: float = 0.5,
    top_k: int | None = None,
) -> list[int]:
    """
    Soft-NMS (class-agnostic), vectorized-ish.
    dets: np.ndarray [N,5] -> [x1,y1,x2,y2,score] (xyxy, half-open coords)
    iou_thr: IoU used by 'linear' method (Nt in the paper). Ignored by 'gaussian'.
    score_thr: drop boxes whose (decayed) score falls below this.
    method: 'linear' or 'gaussian'
    sigma: gaussian sigma for score decay
    top_k: optionally keep only top_k after decay (None = keep all >= score_thr)
    Returns:
        keep_inds: list of kept indices in the original dets
        dets_out:  dets with updated (decayed) scores, in **kept order**
    """
    if len(dets) == 0:
        return []

    dets = dets.copy()
    x1, y1, x2, y2, scores = [dets[:, i] for i in range(5)]

    # areas without +1 (half-open boxes)
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # IoU with rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        # score decay
        if method == 'linear':
            decay = np.ones_like(iou)
            m = iou > iou_thr
            decay[m] = 1 - iou[m]  # linear
        elif method == 'gaussian':
            decay = np.exp(- (iou * iou) / sigma)  # Nt not used
        else:
            raise ValueError("method must be 'linear' or 'gaussian'")

        scores[order[1:]] *= decay
        dets[order[1:], 4] = scores[order[1:]]

        # filter low-score boxes and re-sort tail
        remain = order[1:][scores[order[1:]] >= score_thr]
        tail = remain[np.argsort(scores[remain])[::-1]]
        order = np.concatenate(([i], tail))

        # drop the selected i from order head
        order = order[1:]

    # final prune & optional top-k
    kept_scores = dets[keep, 4]
    keep = [k for k, s in zip(keep, kept_scores, strict=True) if s >= score_thr]
    if top_k is not None and len(keep) > top_k:
        keep = list(np.array(keep)[np.argsort(dets[keep, 4])[::-1][:top_k]])

    return keep

def oks_iou(
    g: NDArray[np.float32],
    d: NDArray[np.float32],
    a_g: float,
    a_d: NDArray[np.float32],
    sigmas: NDArray[np.float64] | None = None,
    vis_thr: float | None = None,
) -> NDArray[np.float32]:
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious



def oks_nms(
    kpts_db: list[dict],  # type: ignore[type-arg]
    thr: float,
    sigmas: NDArray[np.float64] | None = None,
    vis_thr: float | None = None,
    score_per_joint: bool = False,
) -> NDArray[np.intp]:
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return np.array([], dtype=np.intp)

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep: list[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep_arr: NDArray[np.intp] = np.array(keep)

    return keep_arr