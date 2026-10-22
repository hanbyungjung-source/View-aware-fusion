"""Faithful port of the thesis implementation (detect.ipynb, cell 21) with switchable components.

Every default reproduces the original code path exactly ("thesis-code" preset). Extra switches
expose (a) the variants described in the thesis text but absent in the code (logistic upscaling,
(1-c)*A weights), and (b) per-component ablations.
"""
import time
from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np

from .detector import concat_dets, empty_dets


@dataclass(frozen=True)
class HahiCfg:
    routing: str = 'sbsi'            # none | sbsi | dahi_grid | dahi_grid_once
    packing: bool = True             # pack the residual ROIs on canvases
    sce: bool = True                 # Spatial Context Extension
    upscale: str = 'step'            # step (code) | logistic (thesis Eq.) | none
    upscale_ratio: float = 1.5       # step ratio / R_max
    upscale_max: int = 200           # step threshold on max crop side
    tau_scale: float = 128.0         # logistic midpoint (sqrt area)
    gamma: float = 2.5               # logistic steepness (not given in thesis; fitted to Fig. 3-5)
    weight: str = 'conf_sqrt_area'   # conf_sqrt_area (code) | conf_area (thesis) | uniform | conf | sqrt_area
    snap: bool = True                # weighted centre-of-mass snapping
    route_ratio: float = 0.30        # cluster must hold >= ratio * #small objects
    window: int = 512
    sigma_div: float = 4.0           # sigma = window / sigma_div
    canvas: int = 960
    canvas_margin: int = 2
    infer_imgsz: int = 640           # detector input size for crops/canvases (code: model default 640)
    global_imgsz: int = 640          # detector input size of the global pass (code: 640)
    merge_pad: int = 16
    crop_pad_large: int = 80
    crop_pad_small: int = 16
    crop_pad_thresh: int = 200
    conf_filter: float = 0.1         # ROI candidates from global pass
    conf_global: float = 0.3         # global detections accepted as final (code)
    gcc: float = 1.10                # global confidence calibration factor
    decay: float = 0.8               # boundary confidence decay
    nms_local: float = 0.4
    nms_final: float = 0.45
    dahi_step: int = 320
    out_conf: float = 0.3            # final score floor (thesis: 0.3; standard eval: 0.001)
    emit_low_global: bool = False    # also output global dets below conf_global (standard eval)
    max_det: int = 300
    batch: int = 16
    roi_source: str = 'det'          # det (method) | gt (diagnostic oracle: ROI candidates = ground-truth boxes)
    fast_merge: bool = False         # vectorised merge stage; bit-identical output to the thesis-code loops
    nms_impl: str = 'cv2'            # cv2 (thesis code: integer xywh, per class) | tv (torchvision batched_nms, float)
    packing_mode: str = 'shelf'      # shelf (thesis code) | sra (scale-response allocation, see build_canvases_sra)
    sra_lambda: float = 4.0          # price of one canvas (640x640) in expected-recall units
    sra_overlap: int = 32            # native-pixel overlap when an oversized region is split into tiles
    sra_curve: str = 'runs/vd_val/v8s/standard/diag_scale.json'   # detector scale-response profile (val)
    sra_mags: tuple = (0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0)
    trunc: str = 'decay'             # decay (thesis code: x decay for any crop-border box) | drop (see _merge_fast)
    vote: float = 0.0                # > 0: single-stage NMS with box voting at this IoU (needs fast_merge)
    def as_dict(self):
        return asdict(self)


THESIS = HahiCfg()


def standard(cfg: HahiCfg) -> HahiCfg:
    """Low score floor so the whole PR curve is evaluated (standard COCO/VisDrone practice)."""
    return replace(cfg, out_conf=0.001, emit_low_global=True, max_det=1000)


def size_cat(w, h):
    a = w * h
    return 'small' if a < 32 ** 2 else ('medium' if a < 96 ** 2 else 'large')


def cv_nms_per_class(boxes, scores, classes, score_thr, iou_thr):
    """Per-class cv2.dnn.NMSBoxes on integer xywh, exactly as in the thesis code."""
    keep = []
    classes = np.asarray(classes)
    for c in sorted(set(classes.tolist())):
        idx = np.where(classes == c)[0]
        cv_boxes = [[int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2] - boxes[j][0]), int(boxes[j][3] - boxes[j][1])]
                    for j in idx]
        sel = cv2.dnn.NMSBoxes(cv_boxes, [float(scores[j]) for j in idx], score_thr, iou_thr)
        if len(sel):
            keep.extend(idx[np.asarray(sel).flatten()].tolist())
    return keep


def cv_nms_np(B, S, C, score_thr, iou_thr):
    """cv_nms_per_class on arrays: same integer truncation (in the dtype of B), same cv2 NMS, same output order."""
    if not len(B):
        return np.zeros(0, np.int64)
    xywh = np.stack([np.trunc(B[:, 0]), np.trunc(B[:, 1]), np.trunc(B[:, 2] - B[:, 0]),
                     np.trunc(B[:, 3] - B[:, 1])], 1).astype(np.int32)
    S32 = S.astype(np.float32)
    keep = []
    for c in np.unique(C):
        idx = np.flatnonzero(C == c)
        sel = cv2.dnn.NMSBoxes(xywh[idx], S32[idx], score_thr, iou_thr)
        if len(sel):
            keep.append(idx[np.asarray(sel).flatten()])
    return np.concatenate(keep) if keep else np.zeros(0, np.int64)


def _nms(cfg, B, S, C, iou_thr):
    if cfg.nms_impl == 'cv2':
        return cv_nms_np(B, S, C, cfg.out_conf, iou_thr)
    import torch
    from torchvision.ops import batched_nms
    m = np.flatnonzero(S > cfg.out_conf)       # cv2.dnn.NMSBoxes keeps score > threshold
    if not len(m):
        return m
    k = batched_nms(torch.from_numpy(B[m].astype(np.float32)), torch.from_numpy(S[m].astype(np.float32)),
                    torch.from_numpy(C[m]), iou_thr).numpy()
    return m[k]


def wbf_single(B, S, C, iou_thr=0.6, conf_type='avg'):
    """Weighted Boxes Fusion (Solovyev et al., Image Vis. Comput. 2021) for one prediction list, following the
    reference implementation (ensemble_boxes.weighted_boxes_fusion with a single model): boxes are visited in
    descending score order and appended to the first cluster whose current fused box has IoU > iou_thr (same class);
    the fused box is the score-weighted mean of the cluster; score = mean ('avg') or max ('max') of the cluster."""
    out_b, out_s, out_c = [], [], []
    for c in np.unique(C):
        i = np.flatnonzero(C == c)
        i = i[np.argsort(-S[i], kind='stable')]
        fb = np.zeros((0, 4), np.float64)
        wsum, bsum, ssum, smax, cnt = [], [], [], [], []
        for j in i:
            b, s = B[j].astype(np.float64), float(S[j])
            if len(fb):
                ix1 = np.maximum(fb[:, 0], b[0]); iy1 = np.maximum(fb[:, 1], b[1])
                ix2 = np.minimum(fb[:, 2], b[2]); iy2 = np.minimum(fb[:, 3], b[3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                ua = (fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                iou = inter / np.maximum(ua, 1e-9)
                k = int(np.argmax(iou))
                if iou[k] > iou_thr:
                    bsum[k] += s * b; wsum[k] += s; ssum[k] += s; smax[k] = max(smax[k], s); cnt[k] += 1
                    fb[k] = bsum[k] / wsum[k]
                    continue
            fb = np.vstack([fb, b[None]])
            bsum.append(s * b); wsum.append(s); ssum.append(s); smax.append(s); cnt.append(1)
        if len(fb):
            out_b.append(fb)
            out_s.append(np.asarray(ssum) / np.asarray(cnt) if conf_type == 'avg' else np.asarray(smax))
            out_c.append(np.full(len(fb), c))
    if not out_b:
        return B[:0], S[:0], C[:0]
    return (np.concatenate(out_b).astype(B.dtype), np.concatenate(out_s).astype(S.dtype),
            np.concatenate(out_c).astype(C.dtype))


def nms_vote(B, S, C, thr, vote_thr=0.0):
    """Class-wise NMS; with vote_thr > 0 each kept box is replaced by the score-weighted mean of all same-class
    candidate boxes with IoU >= vote_thr (box voting, Gidaris & Komodakis 2015). Returns (boxes, scores, classes).

    Study switches (environment, default off; used only for the overlap / overhead analysis in REPORT section 21):
      HAHI_FUSION = wbf_avg | wbf_max | nmm  -> replace the fusion rule (WBF, or SAHI's GREEDYNMM with IoS 0.5)
      HAHI_VOTE_DEVICE = cuda                -> run NMS + voting on the GPU"""
    import os
    import torch
    from torchvision.ops import batched_nms, box_iou
    if not len(B):
        return B, S, C
    mode = os.environ.get('HAHI_FUSION', '')
    if vote_thr > 0 and mode.startswith('wbf'):
        return wbf_single(B, S, C, vote_thr, mode.split('_')[1])
    if vote_thr > 0 and mode == 'nmm':
        from .baselines import sahi_postprocess
        return sahi_postprocess(B, S, C, 'GREEDYNMM', 'IOS', 0.5)
    dev = os.environ.get('HAHI_VOTE_DEVICE', 'cpu')
    tb, ts, tc = (torch.from_numpy(B.astype(np.float32)).to(dev), torch.from_numpy(S.astype(np.float32)).to(dev),
                  torch.from_numpy(C).to(dev))
    keep = batched_nms(tb, ts, tc, thr)
    if vote_thr <= 0:
        k = keep.cpu().numpy()
        return B[k], S[k], C[k]
    off = tc.to(tb.dtype)[:, None] * (tb.max() + 1)
    M = torch.nan_to_num(box_iou(tb[keep] + off[keep], tb + off))
    M[torch.arange(len(keep), device=M.device), keep] = 1.0   # a kept box always votes for itself (also zero-area)
    W = (M >= vote_thr).to(tb.dtype) * ts[None]
    nb = (W @ tb) / W.sum(1, keepdim=True)
    k = keep.cpu().numpy()
    return nb.cpu().numpy().astype(B.dtype), S[k], C[k]



def _merge_fast(cfg, gb, gs, gc, regions, res, n_dense, infos, wh=None):
    """Vectorised version of the merge stage of run_hahi (float64 arithmetic in the same order as the loops).
    trunc='drop': a local box that touches a crop border lying INSIDE the image (not an image border) is treated as
    a possibly truncated object and discarded instead of score-decayed; the object is left to the other views."""
    LB, LS, LC = [], [], []
    W, H = wh if wh is not None else (np.inf, np.inf)
    drop = cfg.trunc == 'drop'
    for (x1, y1, x2, y2), (b, s, c) in zip(regions, res[:n_dense]):
        if not len(b):
            continue
        bb, ss = b.astype(np.float64), s.astype(np.float64)
        if drop:
            t = ((bb[:, 0] <= 5) & (x1 > 0)) | ((bb[:, 1] <= 5) & (y1 > 0)) | \
                ((bb[:, 2] >= x2 - x1 - 5) & (x2 < W)) | ((bb[:, 3] >= y2 - y1 - 5) & (y2 < H))
            bb, ss, c = bb[~t], ss[~t], c[~t]
            LS.append(ss)
        else:
            edge = (bb[:, 0] <= 5) | (bb[:, 1] <= 5) | (bb[:, 2] >= x2 - x1 - 5) | (bb[:, 3] >= y2 - y1 - 5)
            LS.append(np.where(edge, ss * cfg.decay, ss))
        LB.append(bb + np.array([x1, y1, x1, y1], np.float64))
        LC.append(c)
    if infos:
        icx1, icy1, icx2, icy2, iox, ioy, isc, ick = (np.array([i[k] for i in infos]) for k in
                                                       ('cx1', 'cy1', 'cx2', 'cy2', 'ox', 'oy', 'scale', 'c_idx'))
    for k, (b, s, c) in enumerate(res[n_dense:]):
        if not len(b):
            continue
        bb, ss = b.astype(np.float64), s.astype(np.float64)
        bcx, bcy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        owner = np.full(len(bb), -1)
        for j in np.flatnonzero(ick == k):          # first matching ROI in list order, as in the loop
            m = (owner < 0) & (icx1[j] <= bcx) & (bcx <= icx2[j]) & (icy1[j] <= bcy) & (bcy <= icy2[j])
            owner[m] = j
        sel = np.flatnonzero(owner >= 0)
        if not len(sel):
            continue
        o, bb, ss = owner[sel], bb[sel], ss[sel]
        cc = c[sel]
        if drop:
            ow, oh = (icx2[o] - icx1[o]) / isc[o], (icy2[o] - icy1[o]) / isc[o]
            t = ((bb[:, 0] <= icx1[o] + 3) & (iox[o] > 0)) | ((bb[:, 1] <= icy1[o] + 3) & (ioy[o] > 0)) | \
                ((bb[:, 2] >= icx2[o] - 3) & (iox[o] + ow < W - 1)) | ((bb[:, 3] >= icy2[o] - 3) & (ioy[o] + oh < H - 1))
            o, bb, ss, cc = o[~t], bb[~t], ss[~t], cc[~t]
            LS.append(ss)
        else:
            edge = (bb[:, 0] <= icx1[o] + 3) | (bb[:, 1] <= icy1[o] + 3) | (bb[:, 2] >= icx2[o] - 3) | \
                (bb[:, 3] >= icy2[o] - 3)
            LS.append(np.where(edge, ss * cfg.decay, ss))
        LB.append(np.stack([(bb[:, 0] - icx1[o]) / isc[o] + iox[o], (bb[:, 1] - icy1[o]) / isc[o] + ioy[o],
                            (bb[:, 2] - icx1[o]) / isc[o] + iox[o], (bb[:, 3] - icy1[o]) / isc[o] + ioy[o]], 1))
        LC.append(cc)
    gs64 = gs.astype(np.float64)
    gm = (gs64 >= cfg.conf_global) | cfg.emit_low_global
    gsc = np.where(gs64 >= cfg.conf_global, np.minimum(1.0, gs64 * cfg.gcc), gs64)[gm]
    parts_b, parts_s, parts_c = [gb[gm].astype(np.float32)], [gsc.astype(np.float32)], [gc[gm].astype(np.int64)]
    if cfg.vote > 0:
        # single-stage fusion over all views (global + local) with box voting
        if LB:
            parts_b.append(np.concatenate(LB).astype(np.float32))
            parts_s.append(np.concatenate(LS).astype(np.float32))
            parts_c.append(np.concatenate(LC).astype(np.int64))
        B, S, Cc = np.concatenate(parts_b), np.concatenate(parts_s), np.concatenate(parts_c)
        m = S > cfg.out_conf
        if not m.any():
            return empty_dets()
        return nms_vote(B[m], S[m], Cc[m], cfg.nms_final, cfg.vote)
    if LB:
        lb, ls, lc = np.concatenate(LB), np.concatenate(LS), np.concatenate(LC).astype(np.int64)
        keep = _nms(cfg, lb, ls, lc, cfg.nms_local)
        parts_b.append(lb[keep].astype(np.float32))
        parts_s.append(ls[keep].astype(np.float32))
        parts_c.append(lc[keep])
    B, S, Cc = np.concatenate(parts_b), np.concatenate(parts_s), np.concatenate(parts_c)
    if not len(B):
        return empty_dets()
    keep = _nms(cfg, B, S, Cc, cfg.nms_final)
    return B[keep], S[keep], Cc[keep]


def merge_clusters(boxes, w, h, cfg):
    if not len(boxes):
        return []

    def pad(b, p):
        return [max(0, b[0] - p), max(0, b[1] - p), min(w, b[2] + p), min(h, b[3] + p)]

    def overlap(b1, b2):
        p1, p2 = pad(b1, cfg.merge_pad), pad(b2, cfg.merge_pad)
        return (min(p1[2], p2[2]) > max(p1[0], p2[0])) and (min(p1[3], p2[3]) > max(p1[1], p2[1]))

    curr = [list(b) for b in boxes]
    while True:
        merged, flags = [], [False] * len(curr)
        for i in range(len(curr)):
            if flags[i]:
                continue
            b = curr[i]
            for j in range(i + 1, len(curr)):
                if not flags[j] and overlap(b, curr[j]):
                    b = [min(b[0], curr[j][0]), min(b[1], curr[j][1]), max(b[2], curr[j][2]), max(b[3], curr[j][3])]
                    flags[j] = True
            merged.append(b)
        if len(merged) == len(curr):
            break
        curr = merged
    out = []
    for b in curr:
        bw, bh = b[2] - b[0], b[3] - b[1]
        p = cfg.crop_pad_large if max(bw, bh) < cfg.crop_pad_thresh else cfg.crop_pad_small
        out.append(pad(b, p))
    return out


def upscale_ratio(cw, ch, cfg):
    if cfg.upscale == 'step':
        return cfg.upscale_ratio if max(cw, ch) <= cfg.upscale_max else 1.0
    if cfg.upscale == 'logistic':
        return 1.0 + (cfg.upscale_ratio - 1.0) / (1.0 + (np.sqrt(cw * ch) / cfg.tau_scale) ** cfg.gamma)
    return 1.0


def sbsi_regions(remaining, w, h, cfg):
    """Weighted Gaussian anchor search + weighted centre-of-mass snapping (thesis code)."""
    info = []
    for bx1, by1, bx2, by2, conf in remaining:
        bw, bh = bx2 - bx1, by2 - by1
        if size_cat(bw, bh) != 'small':
            continue
        if cfg.weight == 'conf_sqrt_area':
            wt = (1.0 - conf) * np.sqrt(bw * bh)
        elif cfg.weight == 'conf_area':
            wt = (1.0 - conf) * (bw * bh)
        elif cfg.weight == 'conf':
            wt = 1.0 - conf
        elif cfg.weight == 'sqrt_area':
            wt = np.sqrt(bw * bh)
        else:
            wt = 1.0
        info.append([(bx1 + bx2) / 2, (by1 + by2) / 2, wt])
    regions = []
    n_small = len(info)
    thr = max(1, int(n_small * cfg.route_ratio))
    if n_small == 0:
        return regions, remaining
    pts = np.array([[i[0], i[1]] for i in info])
    weights = np.array([i[2] for i in info])
    nw = weights / (np.mean(weights) + 1e-6)
    sigma = cfg.window / cfg.sigma_div
    W = cfg.window
    while len(pts) > 0:
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        K = np.exp(-d2 / (2 * sigma ** 2))
        dens = K @ nw
        best = np.argmax(dens)
        if cfg.snap:
            infl = nw * K[best]
            s = np.sum(infl)
            cx, cy = (np.sum(infl * pts[:, 0]) / s, np.sum(infl * pts[:, 1]) / s) if s > 0 else pts[best]
        else:
            cx, cy = pts[best]
        dx1 = int(max(0, cx - W / 2))
        dy1 = int(max(0, cy - W / 2))
        dx2 = int(min(w, dx1 + W))
        dy2 = int(min(h, dy1 + W))
        if dx2 - dx1 < W:
            dx1 = max(0, dx2 - W)
        if dy2 - dy1 < W:
            dy1 = max(0, dy2 - W)
        inside = (pts[:, 0] >= dx1) & (pts[:, 0] <= dx2) & (pts[:, 1] >= dy1) & (pts[:, 1] <= dy2)
        if np.sum(inside) >= thr:
            regions.append((dx1, dy1, dx2, dy2))
            pts, nw = pts[~inside], nw[~inside]
            remaining = [rb for rb in remaining if not (rb[0] >= dx1 and rb[1] >= dy1 and rb[2] <= dx2 and rb[3] <= dy2)]
        else:
            break
    return regions, remaining


def dahi_grid_regions(remaining, w, h, cfg, once):
    """The "DAHI" baseline as implemented in the thesis code (exhaustive grid, count of contained boxes)."""
    regions = []
    W, S = cfg.window, cfg.dahi_step
    while len(remaining) > 0:
        best_count, best = -1, None
        for y in range(0, h - W + 1, S):
            for x in range(0, w - W + 1, S):
                cnt = sum(1 for rb in remaining if rb[0] >= x and rb[1] >= y and rb[2] <= x + W and rb[3] <= y + W)
                if cnt > best_count:
                    best_count, best = cnt, (x, y, x + W, y + W)
        if best and best_count >= 1:
            regions.append(best)
            dx1, dy1, dx2, dy2 = best
            remaining = [rb for rb in remaining if not (rb[0] >= dx1 and rb[1] >= dy1 and rb[2] <= dx2 and rb[3] <= dy2)]
        else:
            break
        if once:
            break
    return regions, remaining


def build_canvases(img, remaining, cfg):
    h, w = img.shape[:2]
    C = cfg.canvas
    clustered = merge_clusters(remaining, w, h, cfg)
    crops = []
    for cb in clustered:
        cx1, cy1, cx2, cy2 = map(int, cb[:4])
        cw, ch = cx2 - cx1, cy2 - cy1
        s = upscale_ratio(cw, ch, cfg)
        cwc, chc = min(int(cw * s), C), min(int(ch * s), C)
        if cwc > 0 and chc > 0:
            crop = img[cy1:cy1 + ch, cx1:cx1 + cw]
            if s > 1.0:
                crop = cv2.resize(crop, (cwc, chc), interpolation=cv2.INTER_CUBIC)
            else:
                crop = crop[:chc, :cwc]
                chc, cwc = crop.shape[:2]
            crops.append({'crop': crop, 'ox': cx1, 'oy': cy1, 'cw': cwc, 'ch': chc, 'scale': s})
    crops.sort(key=lambda x: x['ch'], reverse=True)
    canvases, infos = [], []
    cur = np.full((C, C, 3), 114, np.uint8)
    cx = cy = mh = 0
    for it in crops:
        if cx + it['cw'] > C:
            cx = 0
            cy += mh + cfg.canvas_margin
            mh = 0
        if cy + it['ch'] > C:
            canvases.append(cur)
            cur = np.full((C, C, 3), 114, np.uint8)
            cx = cy = mh = 0
        cur[cy:cy + it['ch'], cx:cx + it['cw']] = it['crop']
        infos.append({'c_idx': len(canvases), 'cx1': cx, 'cy1': cy, 'cx2': cx + it['cw'], 'cy2': cy + it['ch'],
                      'ox': it['ox'], 'oy': it['oy'], 'scale': it['scale']})
        cx += it['cw'] + cfg.canvas_margin
        mh = max(mh, it['ch'])
    if mh > 0 or cx > 0:
        canvases.append(cur)
    if cfg.sce:
        _sce(img, canvases, infos, cfg)
        used = sorted(set(i['c_idx'] for i in infos))
        canvases = [canvases[k] for k in used]
        remap = {old: new for new, old in enumerate(used)}
        for i in infos:
            i['c_idx'] = remap[i['c_idx']]
    return canvases, infos


_CURVES = {}


def scale_response(path):
    """Piecewise-linear (in log2 input size) recall curve R(px) measured on the validation set (diag_scale.py)."""
    if path not in _CURVES:
        import json
        import os
        p = path if os.path.isabs(path) else os.path.join(os.path.dirname(__file__), path)
        rows = json.load(open(p))['scale_response']
        xs, ys = [], []
        for r in rows:
            a, b = (float(v) for v in r['in_px'].split('-'))
            b = min(b, 2 * a) if b > 1e6 else b
            xs.append(np.log2(np.sqrt(max(a, 2.0) * b)))
            ys.append(r['recall'])
        _CURVES[path] = (np.asarray(xs), np.asarray(ys))
    xs, ys = _CURVES[path]
    return lambda px: np.interp(np.log2(np.maximum(px, 1e-3)), xs, ys)


def build_canvases_sra(img, remaining, cfg):
    """Scale-response allocation (SRA): choose a magnification per residual cluster by maximising
    sum_i conf_i * [R(size_i * m) - R(size_i * m_global)] - lambda * (canvas area / C^2), where R is the detector's
    measured recall-vs-input-size curve; oversized regions are split into tiles (no truncation) and pieces are
    shelf-packed into C x C canvases that are fed to the detector without further resizing."""
    h, w = img.shape[:2]
    C, M = cfg.canvas, cfg.canvas_margin
    R = scale_response(cfg.sra_curve)
    m_glob = cfg.global_imgsz / max(h, w)
    boxes = np.asarray([b[:4] for b in remaining], np.float64)
    conf = np.asarray([b[4] for b in remaining], np.float64)
    sig = np.sqrt(np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0))
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    base = R(sig * m_glob)
    pieces = []
    for x1, y1, x2, y2 in merge_clusters(remaining, w, h, cfg):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mem = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        if not mem.any() or x2 <= x1 or y2 <= y1:
            continue
        W, H = x2 - x1, y2 - y1
        ov = cfg.sra_overlap
        ntile = lambda L, m: max(1, int(np.ceil((L - ov) / max(C / m - ov, 1.0)))) if L * m > C else 1
        best, best_m = 0.0, None
        for m in cfg.sra_mags:
            if m <= m_glob * 1.05:
                continue
            gain = float(np.sum(conf[mem] * (R(sig[mem] * m) - base[mem])))
            nx, ny = ntile(W, m), ntile(H, m)
            cw = (W + (nx - 1) * ov) * m
            ch = (H + (ny - 1) * ov) * m
            v = gain - cfg.sra_lambda * cw * ch / C ** 2
            if v > best:
                best, best_m = v, m
        if best_m is None:
            continue
        m = best_m
        nx, ny = ntile(W, m), ntile(H, m)
        tw = (W + (nx - 1) * ov) / nx
        th = (H + (ny - 1) * ov) / ny
        for iy in range(ny):
            for ix in range(nx):
                px1 = int(round(x1 + ix * (tw - ov)))
                py1 = int(round(y1 + iy * (th - ov)))
                px2, py2 = min(x2, int(round(px1 + tw))), min(y2, int(round(py1 + th)))
                pw, ph = max(1, int(round((px2 - px1) * m))), max(1, int(round((py2 - py1) * m)))
                pw, ph = min(C, pw), min(C, ph)
                crop = img[py1:py2, px1:px2]
                if crop.size == 0:
                    continue
                if (pw, ph) != (crop.shape[1], crop.shape[0]):
                    crop = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_CUBIC if m > 1 else cv2.INTER_AREA)
                pieces.append({'crop': crop, 'ox': px1, 'oy': py1, 'cw': pw, 'ch': ph, 'scale': m})
    pieces.sort(key=lambda x: x['ch'], reverse=True)
    canvases, infos = [], []
    cur = np.full((C, C, 3), 114, np.uint8)
    cxp = cyp = mh = 0
    for it in pieces:
        if cxp + it['cw'] > C:
            cxp, cyp, mh = 0, cyp + mh + M, 0
        if cyp + it['ch'] > C:
            canvases.append(cur)
            cur = np.full((C, C, 3), 114, np.uint8)
            cxp = cyp = mh = 0
        cur[cyp:cyp + it['ch'], cxp:cxp + it['cw']] = it['crop']
        infos.append({'c_idx': len(canvases), 'cx1': cxp, 'cy1': cyp, 'cx2': cxp + it['cw'], 'cy2': cyp + it['ch'],
                      'ox': it['ox'], 'oy': it['oy'], 'scale': it['scale']})
        cxp += it['cw'] + M
        mh = max(mh, it['ch'])
    if mh > 0 or cxp > 0:
        canvases.append(cur)
    if cfg.sce and infos:
        _sce(img, canvases, infos, cfg)
    return canvases, infos


def _dist(gap, a1, a2):
    half = gap // 2
    if a1 < half:
        return a1, min(gap - a1, a2)
    if a2 < half:
        return min(gap - a2, a1), a2
    return half, gap - half


def _sce(img, canvases, infos, cfg):
    h, w = img.shape[:2]
    C, M = cfg.canvas, cfg.canvas_margin
    for c_idx, canvas in enumerate(canvases):
        c_infos = [i for i in infos if i['c_idx'] == c_idx]
        if not c_infos:
            continue
        rows = sorted(set(i['cy1'] for i in c_infos))
        for ri, cy1r in enumerate(rows):
            row = sorted([i for i in c_infos if i['cy1'] == cy1r], key=lambda x: x['cx1'])
            next_cy1 = rows[ri + 1] if ri + 1 < len(rows) else C
            for j, info in enumerate(row):
                cx1, cy1, cx2, cy2 = info['cx1'], info['cy1'], info['cx2'], info['cy2']
                ox, oy, s = info['ox'], info['oy'], info['scale']
                iw, ih = cx2 - cx1, cy2 - cy1
                ow, oh = max(1, int(iw / s)), max(1, int(ih / s))
                next_cx1 = row[j + 1]['cx1'] if j + 1 < len(row) else C
                aw = next_cx1 - cx1 - (M if j + 1 < len(row) else 0)
                ah = next_cy1 - cy1 - (M if ri + 1 < len(rows) else 0)
                gw, gh = max(0, aw - iw), max(0, ah - ih)
                if gw > 0 or gh > 0:
                    el, er = _dist(int(gw / s), ox, w - (ox + ow))
                    et, eb = _dist(int(gh / s), oy, h - (oy + oh))
                    nox, noy = ox - el, oy - et
                    nw, nh = ow + el + er, oh + et + eb
                    if nw > 0 and nh > 0:
                        ext = img[noy:noy + nh, nox:nox + nw]
                        aw2 = int(nw * s) if s != 1.0 else nw
                        ah2 = int(nh * s) if s != 1.0 else nh
                        if ext.size > 0:
                            if s != 1.0:
                                ext = cv2.resize(ext, (aw2, ah2), interpolation=cv2.INTER_CUBIC)
                            hc, wc = ext.shape[:2]
                            ye, xe = min(cy1 + hc, C), min(cx1 + wc, C)
                            canvas[cy1:ye, cx1:xe] = ext[:ye - cy1, :xe - cx1]
                            info['ox'], info['oy'] = nox, noy
                            info['cx2'], info['cy2'] = xe, ye


def run_uc(det, img, cfg, timer):
    h, w = img.shape[:2]
    ch, cw = h // 2, w // 2
    crops, offs = [img], [(0, 0)]
    for y in (0, ch):
        for x in (0, cw):
            crops.append(img[y:y + ch, x:x + cw])
            offs.append((x, y))
    t = time.perf_counter()
    res = det(crops, conf=cfg.out_conf, batch=5, max_det=cfg.max_det)
    timer['local'] += time.perf_counter() - t
    t = time.perf_counter()
    boxes, scores, classes = [], [], []
    for k, ((ox, oy), (b, s, c)) in enumerate(zip(offs, res)):
        if len(b) and k > 0 and cfg.trunc == 'drop':
            m = 5.0 * max(cw, ch) / 640                   # 5 px at detector input, in native pixels
            cut = ((b[:, 0] <= m) & (ox > 0)) | ((b[:, 1] <= m) & (oy > 0)) | \
                ((b[:, 2] >= cw - m) & (ox + cw < w)) | ((b[:, 3] >= ch - m) & (oy + ch < h))
            b, s, c = b[~cut], s[~cut], c[~cut]
        if len(b):
            boxes.append(b + np.array([ox, oy, ox, oy], np.float32))
            scores.append(s)
            classes.append(c)
    if not boxes:
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    B, S, Cc = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    if cfg.vote > 0:
        m = S > cfg.out_conf
        out = nms_vote(B[m], S[m], Cc[m], cfg.nms_local, cfg.vote)
        timer['merge'] += time.perf_counter() - t
        return out
    keep = cv_nms_per_class(B, S, Cc, cfg.out_conf, cfg.nms_local)
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], Cc[keep]


def run_hahi(det, img, cfg: HahiCfg, timer, rec=None):
    h, w = img.shape[:2]
    t = time.perf_counter()
    gb, gs, gc = det([img], conf=min(cfg.conf_filter, cfg.out_conf), imgsz=cfg.global_imgsz, max_det=cfg.max_det)[0]
    timer['global'] += time.perf_counter() - t

    t = time.perf_counter()
    g_boxes, g_scores, g_cls = [], [], []
    roi = []
    for b, s, c in zip(gb.tolist(), gs.tolist(), gc.tolist()):
        if s >= cfg.conf_global:
            g_boxes.append(b)
            g_scores.append(min(1.0, s * cfg.gcc))
            g_cls.append(c)
        elif cfg.emit_low_global:
            g_boxes.append(b)
            g_scores.append(s)
            g_cls.append(c)
        if s >= cfg.conf_filter:
            roi.append(b + [s])
    if cfg.roi_source == 'gt':
        roi = [b + [0.5] for b in rec.boxes.tolist()]
    remaining = roi
    regions = []
    if cfg.routing == 'sbsi':
        regions, remaining = sbsi_regions(remaining, w, h, cfg)
    elif cfg.routing in ('dahi_grid', 'dahi_grid_once'):
        regions, remaining = dahi_grid_regions(remaining, w, h, cfg, once=cfg.routing == 'dahi_grid_once')
    timer['plan'] += time.perf_counter() - t

    t = time.perf_counter()
    infer = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
    canvases, infos = [], []
    if cfg.packing and remaining:
        canvases, infos = (build_canvases_sra if cfg.packing_mode == 'sra' else build_canvases)(img, remaining, cfg)
    n_dense = len(infer)
    infer.extend(canvases)
    timer['build'] += time.perf_counter() - t

    lb, ls, lc = [], [], []
    if infer:
        t = time.perf_counter()
        res = det(infer, conf=cfg.out_conf, imgsz=cfg.infer_imgsz, batch=cfg.batch, max_det=cfg.max_det)
        timer['local'] += time.perf_counter() - t
    else:
        res = []
    if cfg.fast_merge:
        t = time.perf_counter()
        out = _merge_fast(cfg, gb, gs, gc, regions, res, n_dense, infos, wh=(w, h))
        timer['merge'] += time.perf_counter() - t
        return out
    if infer:
        t = time.perf_counter()
        for (x1, y1, x2, y2), (b, s, c) in zip(regions, res[:n_dense]):
            cwd, chd = x2 - x1, y2 - y1
            for bb, ss, cc in zip(b.tolist(), s.tolist(), c.tolist()):
                if bb[0] <= 5 or bb[1] <= 5 or bb[2] >= cwd - 5 or bb[3] >= chd - 5:
                    ss *= cfg.decay
                lb.append([bb[0] + x1, bb[1] + y1, bb[2] + x1, bb[3] + y1])
                ls.append(ss)
                lc.append(cc)
        for k, (b, s, c) in enumerate(res[n_dense:]):
            for bb, ss, cc in zip(b.tolist(), s.tolist(), c.tolist()):
                bcx, bcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                for info in infos:
                    if info['c_idx'] == k and info['cx1'] <= bcx <= info['cx2'] and info['cy1'] <= bcy <= info['cy2']:
                        if (bb[0] <= info['cx1'] + 3 or bb[1] <= info['cy1'] + 3 or bb[2] >= info['cx2'] - 3
                                or bb[3] >= info['cy2'] - 3):
                            ss *= cfg.decay
                        sc = info['scale']
                        lb.append([(bb[0] - info['cx1']) / sc + info['ox'], (bb[1] - info['cy1']) / sc + info['oy'],
                                   (bb[2] - info['cx1']) / sc + info['ox'], (bb[3] - info['cy1']) / sc + info['oy']])
                        ls.append(ss)
                        lc.append(cc)
                        break
    else:
        t = time.perf_counter()

    keep = cv_nms_per_class(lb, ls, lc, cfg.out_conf, cfg.nms_local) if lb else []
    B = g_boxes + [lb[k] for k in keep]
    S = g_scores + [ls[k] for k in keep]
    Cc = g_cls + [lc[k] for k in keep]
    if not B:
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    B, S, Cc = np.asarray(B, np.float32), np.asarray(S, np.float32), np.asarray(Cc, np.int64)
    keep = cv_nms_per_class(B, S, Cc, cfg.out_conf, cfg.nms_final)
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], Cc[keep]


# Presets matching the six thesis rows.
THESIS_PRESETS = {
    'UC2x2': ('uc', THESIS),
    'DAHI-thesis': ('hahi', replace(THESIS, routing='dahi_grid', packing=False, sce=False)),
    'Packing-thesis': ('hahi', replace(THESIS, routing='none', sce=False)),
    'DAHI+Packing-thesis': ('hahi', replace(THESIS, routing='dahi_grid_once', sce=False)),
    'SBSI+Packing': ('hahi', replace(THESIS, routing='sbsi', sce=False)),
    'HAHI': ('hahi', THESIS),
}

_ = concat_dets  # re-exported for other modules
