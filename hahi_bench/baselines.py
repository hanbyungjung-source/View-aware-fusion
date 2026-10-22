"""Simple and literature baselines: single full-image inference (several input sizes) and SAHI."""
import time

import numpy as np
import torch
from sahi.postprocess.combine import batched_greedy_nmm, batched_nms
from sahi.slicing import get_slice_bboxes

from .detector import empty_dets


def run_global(det, img, timer, imgsz=640, conf=0.001, max_det=1000):
    t = time.perf_counter()
    b, s, c = det([img], conf=conf, imgsz=imgsz, max_det=max_det)[0]
    timer['global'] += time.perf_counter() - t
    return b, s, c


def run_multiscale(det, img, timer, sizes=(640, 1280), conf=0.001, iou=0.6, max_det=1000, aspect=0.0, vote=0.0):
    """Plain multi-scale test-time inference: full image at several input sizes + class-wise NMS.
    aspect > 0 emulates a dedicated static engine with that width/height ratio for the larger sizes (e.g. 16:9 ->
    1280x736): the image is padded bottom/right to the engine aspect, so other aspect ratios are down-scaled."""
    from torchvision.ops import batched_nms as tv_nms
    h, w = img.shape[:2]
    parts = []
    for sz in sizes:
        t = time.perf_counter()
        x = img
        if aspect > 0 and sz != sizes[0]:
            tw, th = max(w, int(np.ceil(h * aspect))), max(h, int(np.ceil(w / aspect)))
            if (tw, th) != (w, h):
                x = np.full((th, tw, 3), 114, np.uint8)
                x[:h, :w] = img
        b, s, c = det([x], conf=conf, imgsz=sz, max_det=max_det)[0]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        parts.append((b, s, c))
        timer['global' if sz == sizes[0] else 'local'] += time.perf_counter() - t
    t = time.perf_counter()
    B = np.concatenate([p[0] for p in parts])
    S = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    if not len(B):
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    if vote > 0:
        from .thesis_methods import nms_vote
        out = nms_vote(B, S, C, iou, vote)
        timer['merge'] += time.perf_counter() - t
        return out
    keep = tv_nms(torch.from_numpy(B), torch.from_numpy(S), torch.from_numpy(C), iou).numpy()
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], C[keep]


def run_mstile(det, img, timer, size=1280, overlap=0.0, conf=0.001, iou=0.6, max_det=1000, base=640, trunc='none',
               vote=0.0):
    """Multi-scale inference for a detector with a FIXED base x base input (e.g. NPU / static engine):
    the 640 global pass plus the image resized to `size` (long side) and cut into base x base tiles
    (the minimum number of tiles that covers it, spread evenly; `overlap` = minimum overlap ratio).
    trunc='drop' / vote>0 apply the same view-aware fusion as HAHI (drop tile boxes touching an inner tile border,
    which the global pass covers; NMS with box voting)."""
    from torchvision.ops import batched_nms as tv_nms
    h, w = img.shape[:2]
    t = time.perf_counter()
    parts = [det([img], conf=conf, imgsz=base, max_det=max_det)[0]]
    timer['global'] += time.perf_counter() - t
    t = time.perf_counter()
    T = base * max(w, h) / size                    # tile side in native pixels
    step = T * (1 - overlap)

    def starts(L):
        if L <= T:
            return [0]
        n = int(np.ceil((L - T) / step)) + 1
        return [int(round(v)) for v in np.linspace(0, L - T, n)]
    tiles = [(x, y, int(min(w, x + T)), int(min(h, y + T))) for y in starts(h) for x in starts(w)]
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
    timer['build'] += time.perf_counter() - t
    t = time.perf_counter()
    res = det(crops, conf=conf, imgsz=base, max_det=max_det)
    timer['local'] += time.perf_counter() - t
    t = time.perf_counter()
    for (x1, y1, x2, y2), (b, s, c) in zip(tiles, res):
        if trunc == 'drop' and len(b):
            m = 5.0 * max(x2 - x1, y2 - y1) / base          # 5 px at detector input, in native pixels
            cut = ((b[:, 0] <= m) & (x1 > 0)) | ((b[:, 1] <= m) & (y1 > 0)) | \
                ((b[:, 2] >= x2 - x1 - m) & (x2 < w)) | ((b[:, 3] >= y2 - y1 - m) & (y2 < h))
            b, s, c = b[~cut], s[~cut], c[~cut]
        parts.append((b + np.array([x1, y1, x1, y1], np.float32), s, c))
    B = np.concatenate([p[0] for p in parts])
    S = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    if not len(B):
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    if vote > 0:
        from .thesis_methods import nms_vote
        out = nms_vote(B, S, C, iou, vote)
        timer['merge'] += time.perf_counter() - t
        return out
    keep = tv_nms(torch.from_numpy(B), torch.from_numpy(S), torch.from_numpy(C), iou).numpy()
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], C[keep]


def _ios(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def sahi_postprocess(B, S, C, kind='GREEDYNMM', metric='IOS', thr=0.5):
    """SAHI postprocess (class-aware), using SAHI's own matching functions; merge = union box, max score."""
    if len(B) == 0:
        return empty_dets()
    t = torch.from_numpy(np.concatenate([B, S[:, None], C[:, None].astype(np.float32)], 1).astype(np.float32))
    if kind == 'NMS':
        if metric == 'IOU':
            # Same greedy class-wise IoU NMS as sahi.postprocess.combine.batched_nms, but vectorised (SAHI's
            # shapely loop takes minutes on DOTA-size images with ~40k candidate boxes).
            from torchvision.ops import batched_nms as tv_nms
            keep = tv_nms(t[:, :4], t[:, 4], t[:, 5].long(), thr).numpy()
            return B[keep], S[keep], C[keep]
        keep = batched_nms(t, match_metric=metric, match_threshold=thr)
        keep = np.asarray(keep, dtype=np.int64).reshape(-1)
        return B[keep], S[keep], C[keep]
    k2m = batched_greedy_nmm(t, match_metric=metric, match_threshold=thr)
    match = _ios if metric == 'IOS' else _iou
    ob, os_, oc = [], [], []
    for keep, merges in k2m.items():
        box, score, cls = B[keep].astype(np.float64).tolist(), float(S[keep]), int(C[keep])
        for m in merges:
            mb = B[m].tolist()
            if match(box, mb) > thr:
                box = [min(box[0], mb[0]), min(box[1], mb[1]), max(box[2], mb[2]), max(box[3], mb[3])]
                if float(S[m]) >= score:
                    cls = int(C[m])
                score = max(score, float(S[m]))
        ob.append(box)
        os_.append(score)
        oc.append(cls)
    return np.asarray(ob, np.float32), np.asarray(os_, np.float32), np.asarray(oc, np.int64)


def run_sahi(det, img, timer, slice_size=512, overlap=0.2, standard_pred=True, post='GREEDYNMM',
             metric='IOS', thr=0.5, conf=0.001, max_det=1000, batch=16, trunc='none', vote=0.0):
    """SAHI (Akyon et al., ICIP 2022) sliced inference with the library's default settings.

    The library predicts slices one by one; here slices are batched for speed, which does not change outputs.
    trunc='drop' / vote>0: view-aware fusion (drop slice boxes touching an inner slice border, which the standard
    full-image prediction covers; NMS with box voting at IoU `vote`), as a plug-in on top of the tuned SAHI.
    """
    h, w = img.shape[:2]
    t = time.perf_counter()
    sl = get_slice_bboxes(h, w, slice_height=slice_size, slice_width=slice_size, auto_slice_resolution=False,
                          overlap_height_ratio=overlap, overlap_width_ratio=overlap)
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in sl]
    timer['build'] += time.perf_counter() - t
    t = time.perf_counter()
    res = det(crops, conf=conf, batch=batch, max_det=max_det)
    timer['local'] += time.perf_counter() - t
    parts = []
    for (x1, y1, x2, y2), (b, s, c) in zip(sl, res):
        if len(b) and trunc == 'drop' and standard_pred and len(sl) > 1:
            m = 5.0 * max(x2 - x1, y2 - y1) / 640
            cut = ((b[:, 0] <= m) & (x1 > 0)) | ((b[:, 1] <= m) & (y1 > 0)) | \
                ((b[:, 2] >= x2 - x1 - m) & (x2 < w)) | ((b[:, 3] >= y2 - y1 - m) & (y2 < h))
            b, s, c = b[~cut], s[~cut], c[~cut]
        if len(b):
            b = b + np.array([x1, y1, x1, y1], np.float32)
            parts.append((b, s, c))
    if standard_pred and len(sl) > 1:
        t = time.perf_counter()
        parts.append(det([img], conf=conf, max_det=max_det)[0])
        timer['global'] += time.perf_counter() - t
    t = time.perf_counter()
    parts = [p for p in parts if len(p[0])]
    if not parts:
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    B = np.concatenate([p[0] for p in parts])
    B[:, [0, 2]] = B[:, [0, 2]].clip(0, w)
    B[:, [1, 3]] = B[:, [1, 3]].clip(0, h)
    S = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    if vote > 0:
        from .thesis_methods import nms_vote
        out = nms_vote(B, S, C, thr, vote)
    else:
        out = sahi_postprocess(B, S, C, post, metric, thr)
    timer['merge'] += time.perf_counter() - t
    return out
