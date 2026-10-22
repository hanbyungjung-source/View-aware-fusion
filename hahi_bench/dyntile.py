"""Dynamic Tiling (Nguyen, Tulabandhula and Nguyen, arXiv:2309.11069, 2023), re-implemented from the paper text
(no official code is available).

Paper settings reproduced (Sections 3 and 4.3 of the paper):
  * every image is resized to 1920x1080 and cut into a 3x2 grid of non-overlapping tiles (640x540);
  * a tile prediction within x% of W (vertical shared boundary) or of H (horizontal shared boundary) is Near-Edge,
    within the threshold of two shared boundaries Near-Corner, otherwise Central; x = 2 %;
  * dynamic patches: for a common edge E(i,j) the patch has two sides parallel to E, its extent along E is set by the
    flagged predictions and its maximum size is one tile (W/N x H/M) with E as the axis of symmetry; when flagged
    predictions exist only on one side, the patch is expanded to cover 50 % of the neighbouring tile; for a common
    corner the patch is one tile centred on the corner;
  * flagged predictions are discarded and replaced by the dynamic-patch predictions that are near or cross E(i,j);
  * Tile Minimizer: dynamic patches are packed into composite images of tile size (fewer forward passes);
  * FI (full-image inference) with the large-small filter: theta = alpha * W * H; full-image predictions are kept
    only if larger than theta, tile / dynamic-patch predictions only if smaller than theta;
  * the predictions are combined with class-wise NMS.
Not specified in the paper (documented assumptions, selected on VisDrone val with the same rule as every baseline):
  alpha, the NMS IoU threshold, the score above which a flagged prediction creates a dynamic patch (plan_conf), the
  composite packing rule (shelf packing, tallest first), and how a patch that is longer than one tile along E is
  handled (split into equal segments). The detector input size is 640, so tiles and
  composites are seen at native resolution as in the paper; the paper also fine-tunes the detector on tiles, which we
  do not do (the same detector is used by every method in this study).

Study options: dynamic=False keeps the flagged predictions (uniform tiles + FI); filt=False disables the large-small
filter; trunc='drop' applies our truncation-aware filtering instead of dynamic patches; vote > 0 replaces NMS by NMS
with box voting.
"""
import time

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from .detector import empty_dets


def _shelf_pack(sizes, cw, ch):
    """Pack (w, h) rectangles (each <= cw x ch) into as few cw x ch canvases as possible (tallest first)."""
    order = sorted(range(len(sizes)), key=lambda k: -sizes[k][1])
    canvases, place = [], {}
    for k in order:
        w, h = sizes[k]
        done = False
        for ci, cv in enumerate(canvases):
            for shelf in cv['shelves']:
                if h <= shelf['h'] and shelf['x'] + w <= cw:
                    place[k] = (ci, shelf['x'], shelf['y'])
                    shelf['x'] += w
                    done = True
                    break
            if done:
                break
            if cv['y'] + h <= ch:
                cv['shelves'].append({'y': cv['y'], 'h': h, 'x': w})
                place[k] = (ci, 0, cv['y'])
                cv['y'] += h
                done = True
                break
        if not done:
            canvases.append({'shelves': [{'y': 0, 'h': h, 'x': w}], 'y': h})
            place[k] = (len(canvases) - 1, 0, 0)
    return len(canvases), place


def run_dyntile(det, img, timer, base_w=1920, base_h=1080, nx=3, ny=2, edge=0.02, alpha=0.001, fi=True,
                dynamic=True, minimizer=True, filt=True, iou=0.6, conf=0.001, max_det=1000, imgsz=640,
                trunc='none', vote=0.0, plan_conf=0.25, dbg=None):
    """plan_conf: only flagged predictions with at least this score create dynamic patches (all flagged predictions
    are discarded). The paper does not state the score used for planning; 0.25 is the Ultralytics default
    prediction threshold, while the evaluation keeps every output down to `conf`."""
    h, w = img.shape[:2]
    W, H = base_w, base_h
    t = time.perf_counter()
    x = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR) if (w, h) != (W, H) else img
    tw, th = W / nx, H / ny
    tiles = [(int(round(c * tw)), int(round(r * th)), int(round((c + 1) * tw)), int(round((r + 1) * th)), c, r)
             for r in range(ny) for c in range(nx)]
    crops = [x[y1:y2, x1:x2] for x1, y1, x2, y2, _, _ in tiles]
    timer['build'] += time.perf_counter() - t
    parts_full = None
    if fi:
        t = time.perf_counter()
        parts_full = det([x], conf=conf, imgsz=imgsz, max_det=max_det)[0]
        timer['global'] += time.perf_counter() - t
    t = time.perf_counter()
    res = det(crops, conf=conf, imgsz=imgsz, max_det=max_det)
    timer['local'] += time.perf_counter() - t

    t = time.perf_counter()
    dx, dy = edge * W, edge * H
    keepB, keepS, keepC = [], [], []
    groups = {}                     # boundary key -> list of (box, side)
    for (x1, y1, x2, y2, c, r), (b, s, cl) in zip(tiles, res):
        if not len(b):
            continue
        b = b.astype(np.float64) + np.array([x1, y1, x1, y1], np.float64)
        if trunc == 'drop':
            m = 5.0 * max(x2 - x1, y2 - y1) / imgsz          # 5 px at the detector input
            cut = ((b[:, 0] - x1 <= m) & (c > 0)) | ((b[:, 1] - y1 <= m) & (r > 0)) | \
                ((x2 - b[:, 2] <= m) & (c < nx - 1)) | ((y2 - b[:, 3] <= m) & (r < ny - 1))
            keepB.append(b[~cut]), keepS.append(s[~cut]), keepC.append(cl[~cut])
            continue
        dl, dr = b[:, 0] - x1, x2 - b[:, 2]
        dt, db = b[:, 1] - y1, y2 - b[:, 3]
        nl, nr = (c > 0) & (dl <= dx), (c < nx - 1) & (dr <= dx)
        nt, nb = (r > 0) & (dt <= dy), (r < ny - 1) & (db <= dy)
        vert, hori = nl | nr, nt | nb
        flagged = vert | hori
        if not dynamic:
            flagged[:] = False
        keepB.append(b[~flagged]), keepS.append(s[~flagged]), keepC.append(cl[~flagged])
        for k in np.flatnonzero(flagged & (s >= plan_conf)):
            left = nl[k] and (not nr[k] or dl[k] <= dr[k])
            top = nt[k] and (not nb[k] or dt[k] <= db[k])
            vx = c if left else c + 1
            vy = r if top else r + 1
            if vert[k] and hori[k]:
                key = ('c', vx, vy)
            elif vert[k]:
                key = ('v', vx, r)
            else:
                key = ('h', c, vy)
            groups.setdefault(key, []).append((b[k], 0 if (left if vert[k] else top) else 1))
    patches = []                    # (x1, y1, x2, y2, key)
    for key, items in groups.items():
        bb = np.array([it[0] for it in items])
        sides = {it[1] for it in items}
        if key[0] == 'c':
            cx, cy = key[1] * tw, key[2] * th
            patches.append((cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2, key))
            continue
        if key[0] == 'v':
            e = key[1] * tw
            hw = tw / 2 if len(sides) == 1 else min(tw / 2, max((e - bb[:, 0]).max(), (bb[:, 2] - e).max()) + dx)
            a0, a1 = max(0.0, bb[:, 1].min() - dy), min(float(H), bb[:, 3].max() + dy)
            n = int(np.ceil((a1 - a0) / th - 1e-9))
            for k in range(max(n, 1)):
                s0, s1 = a0 + (a1 - a0) * k / max(n, 1), a0 + (a1 - a0) * (k + 1) / max(n, 1)
                patches.append((e - hw, s0, e + hw, s1, key))
        else:
            e = key[2] * th
            hh = th / 2 if len(sides) == 1 else min(th / 2, max((e - bb[:, 1]).max(), (bb[:, 3] - e).max()) + dy)
            a0, a1 = max(0.0, bb[:, 0].min() - dx), min(float(W), bb[:, 2].max() + dx)
            n = int(np.ceil((a1 - a0) / tw - 1e-9))
            for k in range(max(n, 1)):
                s0, s1 = a0 + (a1 - a0) * k / max(n, 1), a0 + (a1 - a0) * (k + 1) / max(n, 1)
                patches.append((s0, e - hh, s1, e + hh, key))
    rects = []
    for p in patches:
        x1, y1 = int(np.floor(max(0.0, p[0]))), int(np.floor(max(0.0, p[1])))
        x2, y2 = int(np.ceil(min(float(W), p[2]))), int(np.ceil(min(float(H), p[3])))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            rects.append((x1, y1, min(x2, x1 + int(np.ceil(tw))), min(y2, y1 + int(np.ceil(th))), p[4]))
    if dbg is not None:
        dbg.update({'rects': rects, 'groups': groups, 'base': (W, H), 'tiles': tiles})
    timer['plan'] += time.perf_counter() - t

    if rects:
        t = time.perf_counter()
        cw, ch = int(np.ceil(tw)), int(np.ceil(th))
        sizes = [(r[2] - r[0], r[3] - r[1]) for r in rects]
        if minimizer:
            n_can, place = _shelf_pack(sizes, cw, ch)
        else:
            n_can, place = len(rects), {k: (k, 0, 0) for k in range(len(rects))}
        canv = [np.full((ch, cw, 3), 114, np.uint8) for _ in range(n_can)]
        for k, (x1, y1, x2, y2, _) in enumerate(rects):
            ci, ox, oy = place[k]
            canv[ci][oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = x[y1:y2, x1:x2]
        timer['build'] += time.perf_counter() - t
        t = time.perf_counter()
        cres = det(canv, conf=conf, imgsz=imgsz, max_det=max_det)
        timer['local'] += time.perf_counter() - t
        t = time.perf_counter()
        for k, (x1, y1, x2, y2, key) in enumerate(rects):
            ci, ox, oy = place[k]
            b, s, cl = cres[ci]
            if not len(b):
                continue
            bcx, bcy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            own = (bcx >= ox) & (bcx < ox + (x2 - x1)) & (bcy >= oy) & (bcy < oy + (y2 - y1))
            bb = b[own].astype(np.float64) + np.array([x1 - ox, y1 - oy, x1 - ox, y1 - oy], np.float64)
            ss, cc = s[own], cl[own]
            if key[0] in ('v', 'c'):
                ex = key[1] * tw
                nv = (bb[:, 0] <= ex + dx) & (bb[:, 2] >= ex - dx)
            else:
                nv = np.zeros(len(bb), bool)
            if key[0] in ('h', 'c'):
                ey = key[2] * th
                nh = (bb[:, 1] <= ey + dy) & (bb[:, 3] >= ey - dy)
            else:
                nh = np.zeros(len(bb), bool)
            m = nv | nh
            keepB.append(bb[m]), keepS.append(ss[m]), keepC.append(cc[m])
        timer['merge'] += time.perf_counter() - t

    t = time.perf_counter()
    B = np.concatenate(keepB) if keepB else np.zeros((0, 4))
    S = np.concatenate(keepS) if keepS else np.zeros(0, np.float32)
    C = np.concatenate(keepC) if keepC else np.zeros(0, np.int64)
    theta = alpha * W * H
    if fi and filt:
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        m = area < theta
        B, S, C = B[m], S[m], C[m]
    if fi:
        gb, gs, gc = parts_full
        gb = gb.astype(np.float64)
        if filt:
            m = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) > theta
            gb, gs, gc = gb[m], gs[m], gc[m]
        B, S, C = np.concatenate([B, gb]), np.concatenate([S, gs]), np.concatenate([C, gc])
    if not len(B):
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    B = B * np.array([w / W, h / H, w / W, h / H])
    B[:, [0, 2]] = B[:, [0, 2]].clip(0, w)
    B[:, [1, 3]] = B[:, [1, 3]].clip(0, h)
    B, S, C = B.astype(np.float32), S.astype(np.float32), C.astype(np.int64)
    if vote > 0:
        from .thesis_methods import nms_vote
        out = nms_vote(B, S, C, iou, vote)
        timer['merge'] += time.perf_counter() - t
        return out
    keep = batched_nms(torch.from_numpy(B), torch.from_numpy(S), torch.from_numpy(C), iou).numpy()
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], C[keep]
