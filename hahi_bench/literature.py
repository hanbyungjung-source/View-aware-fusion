"""Literature baselines re-implemented from the original papers (no official code exists for either).

DAHI  - Suarez-Ramirez, Santana-Cedres, Monzon, Pattern Recognition 171 (2026) 112228.
        Global pass (beta) -> n random regions -> RDE (Eq. 1-2) -> DACS (Alg. 1, IoS tau) ->
        crop inference (C_TH) -> CMA-NMS (Alg. 2). Defaults = paper Table 2 values for VisDrone.
PAN   - Fan et al., "Speed up object detection on gigapixel-level images with patch arrangement", CVPR 2022.
        The learned PFM/PPM (pointer networks trained with REINFORCE) are replaced by the paper's own
        "fine-grained + packing" configuration (Table 2, AP50 0.717 vs. 0.715 for full PAN): coarse detections
        -> mean-shift clustering -> fine patches -> greedy bottom-up packing, max capacity C=4 per canvas.
        Unspecified details (canvas size, bandwidth, merge NMS) are documented assumptions tuned on val.
"""
import time
import zlib
from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np
import torch
from sklearn.cluster import MeanShift
from torchvision.ops import batched_nms

from .detector import empty_dets


# ----------------------------------------------------------------------------------------------- DAHI
@dataclass(frozen=True)
class DahiCfg:
    beta: float = 0.001          # global-pass threshold used for density estimation
    c_th: float = 0.01           # acceptance threshold (global + crops)
    n_regions: int = 1000
    size_min: int = 480          # VisDrone region width range (= training crop range)
    size_max: int = 960
    aspect: float = 16 / 9       # VisDrone (w:h), DAHI-base
    lam: float = 0.001           # density regulariser
    tau: float = 0.25            # DACS IoS threshold
    n_roi: int = 0               # 0 = unconstrained (paper default)
    c_dth: float = 0.0           # 0 = off (paper default)
    nms_t: float = 0.65          # CMA-NMS IoU threshold
    theta: int = 10              # margin (px)
    t_mt: float = 0.9            # IoS threshold for margin boxes
    imgsz: int = 640             # crop input size (detector native size)
    batch: int = 16
    seed: int = 0
    max_det: int = 1000
    trunc: str = 'none'          # plug-in: 'drop' = discard crop boxes touching an inner crop border (view-aware fusion)
    vote: float = 0.0            # plug-in: > 0 replaces CMA-NMS by NMS(nms_t) with box voting at this IoU


def _inter(a, b):
    """Pairwise intersection areas between (N,4) and (M,4) xyxy arrays."""
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return iw * ih


def _area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def dahi_regions(boxes, w, h, cfg, rng):
    n = cfg.n_regions
    rw = rng.uniform(cfg.size_min, cfg.size_max, n)
    rh = rw / cfg.aspect
    rw, rh = np.minimum(rw, w), np.minimum(rh, h)
    x1 = rng.uniform(0, 1, n) * (w - rw)
    y1 = rng.uniform(0, 1, n) * (h - rh)
    R = np.stack([x1, y1, x1 + rw, y1 + rh], 1)
    if len(boxes) == 0:
        return R[:0], np.zeros(0)
    frac = _inter(R, boxes) / np.maximum(_area(boxes), 1e-9)[None, :]
    omega = np.where(frac >= 0.5, frac, 0.0)
    dens = omega.sum(1) / (_area(R) * cfg.lam)
    return R, dens


def dacs(R, dens, cfg, boxes=None):
    order = np.argsort(-dens, kind='stable')
    order = order[dens[order] > 0]
    alive = np.ones(len(R), bool)
    roi = []
    covered = np.zeros(len(boxes) if boxes is not None else 0, bool)
    for k in order:
        if not alive[k]:
            continue
        if cfg.n_roi and len(roi) >= cfg.n_roi:
            break
        if cfg.c_dth and boxes is not None and len(boxes) and covered.mean() >= cfg.c_dth:
            break
        roi.append(R[k])
        alive[k] = False
        idx = np.where(alive)[0]
        if len(idx):
            inter = _inter(R[k:k + 1], R[idx])[0]
            ios = inter / np.maximum(np.minimum(_area(R[k:k + 1]), _area(R[idx])), 1e-9)
            alive[idx[ios > cfg.tau]] = False
        if cfg.c_dth and boxes is not None and len(boxes):
            covered |= (_inter(R[k:k + 1], boxes)[0] / np.maximum(_area(boxes), 1e-9)) >= 0.5
    return np.asarray(roi).reshape(-1, 4)


def cma_nms(B, S, C, origin, crops, cfg):
    """Crop-Margin-Aware NMS (Alg. 2), class-wise. origin=-1 marks global (whole-image) boxes."""
    keep_all = []
    margin = np.zeros(len(B), bool)
    loc = origin >= 0
    if loc.any():
        cr = crops[origin[loc]]
        b = B[loc]
        t = cfg.theta
        margin[loc] = ((b[:, 0] < cr[:, 0] + t) | (b[:, 0] > cr[:, 2] - t) | (b[:, 2] < cr[:, 0] + t) |
                       (b[:, 2] > cr[:, 2] - t) | (b[:, 1] < cr[:, 1] + t) | (b[:, 1] > cr[:, 3] - t) |
                       (b[:, 3] < cr[:, 1] + t) | (b[:, 3] > cr[:, 3] - t))
    areas = _area(B)
    for c in np.unique(C):
        idx = np.where(C == c)[0]
        idx = idx[np.argsort(-S[idx], kind='stable')]
        alive = np.ones(len(idx), bool)
        for p in range(len(idx)):
            if not alive[p]:
                continue
            s = idx[p]
            keep_all.append(s)
            rest = np.where(alive)[0]
            rest = rest[rest > p]
            if not len(rest):
                continue
            ri = idx[rest]
            inter = _inter(B[s:s + 1], B[ri])[0]
            iou = inter / np.maximum(areas[s] + areas[ri] - inter, 1e-9)
            ios = inter / np.maximum(np.minimum(areas[s], areas[ri]), 1e-9)
            use_ios = (origin[ri] != origin[s]) & margin[ri]
            sup = np.where(use_ios, ios > cfg.t_mt, iou > cfg.nms_t)
            alive[rest[sup]] = False
    keep_all = np.asarray(keep_all, np.int64)
    return keep_all


def run_dahi(det, img, cfg: DahiCfg, timer):
    h, w = img.shape[:2]
    t = time.perf_counter()
    gb, gs, gc = det([img], conf=cfg.beta, max_det=cfg.max_det)[0]
    timer['global'] += time.perf_counter() - t

    t = time.perf_counter()
    seed = zlib.crc32(np.ascontiguousarray(img[::64, ::64]).tobytes()) ^ cfg.seed
    rng = np.random.default_rng(seed)
    R, dens = dahi_regions(gb, w, h, cfg, rng)
    roi = dacs(R, dens, cfg, gb) if len(R) else np.zeros((0, 4))
    roi = np.round(roi).astype(int)
    timer['plan'] += time.perf_counter() - t

    parts_b, parts_s, parts_c, parts_o = [], [], [], []
    keepg = gs > cfg.c_th
    parts_b.append(gb[keepg]); parts_s.append(gs[keepg]); parts_c.append(gc[keepg])
    parts_o.append(np.full(int(keepg.sum()), -1))
    if len(roi):
        t = time.perf_counter()
        crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in roi]
        timer['build'] += time.perf_counter() - t
        t = time.perf_counter()
        res = det(crops, conf=cfg.c_th, imgsz=cfg.imgsz, batch=cfg.batch, max_det=cfg.max_det)
        timer['local'] += time.perf_counter() - t
        for k, ((x1, y1, x2, y2), (b, s, c)) in enumerate(zip(roi, res)):
            if cfg.trunc == 'drop' and len(b):
                m = 5.0 * max(x2 - x1, y2 - y1) / cfg.imgsz
                cut = ((b[:, 0] <= m) & (x1 > 0)) | ((b[:, 1] <= m) & (y1 > 0)) | \
                    ((b[:, 2] >= x2 - x1 - m) & (x2 < w)) | ((b[:, 3] >= y2 - y1 - m) & (y2 < h))
                b, s, c = b[~cut], s[~cut], c[~cut]
            parts_b.append(b + np.array([x1, y1, x1, y1], np.float32))
            parts_s.append(s); parts_c.append(c); parts_o.append(np.full(len(b), k))
    t = time.perf_counter()
    B = np.concatenate(parts_b).astype(np.float32)
    S = np.concatenate(parts_s).astype(np.float32)
    Cc = np.concatenate(parts_c).astype(np.int64)
    O = np.concatenate(parts_o).astype(np.int64)
    if not len(B):
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    if cfg.vote > 0:
        from .thesis_methods import nms_vote
        out = nms_vote(B, S, Cc, cfg.nms_t, cfg.vote)
        timer['merge'] += time.perf_counter() - t
        return out
    keep = cma_nms(B, S, Cc, O, roi.astype(np.float32) if len(roi) else np.zeros((1, 4), np.float32), cfg)
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], Cc[keep]


# ----------------------------------------------------------------------------------------------- PAN
@dataclass(frozen=True)
class PanCfg:
    conf_coarse: float = 0.1     # coarse detections used for patch generation (not given in paper)
    bandwidth: float = 128.0     # mean-shift bandwidth in px (not given in paper)
    pad: int = 16                # patch margin around clustered boxes
    canvas: int = 640            # canvas = detector input (not given in paper)
    capacity: int = 4            # max patches per canvas (paper default C=4)
    imgsz: int = 640
    fuse_global: bool = True     # also keep coarse/global detections
    nms_iou: float = 0.5
    conf: float = 0.001
    batch: int = 16
    max_det: int = 1000
    trunc: str = 'none'          # plug-in: 'drop' = discard canvas boxes touching an inner patch border
    vote: float = 0.0            # plug-in: > 0 = NMS(nms_iou) with box voting at this IoU


def pan_patches(boxes, w, h, cfg):
    if len(boxes) == 0:
        return np.zeros((0, 4))
    ctr = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    if len(boxes) == 1:
        labels = np.zeros(1, int)
    else:
        labels = MeanShift(bandwidth=cfg.bandwidth, bin_seeding=True, cluster_all=True).fit(ctr).labels_
    P = []
    for l in np.unique(labels):
        bb = boxes[labels == l]
        P.append([max(0, bb[:, 0].min() - cfg.pad), max(0, bb[:, 1].min() - cfg.pad),
                  min(w, bb[:, 2].max() + cfg.pad), min(h, bb[:, 3].max() + cfg.pad)])
    return np.floor(np.asarray(P)).astype(int)


def pan_pack(img, patches, cfg):
    """Greedy bottom-up shelf packing; a new canvas is opened when space or capacity C is exhausted."""
    C = cfg.canvas
    items = []
    for x1, y1, x2, y2 in patches:
        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            continue
        s = min(1.0, C / pw, C / ph)
        crop = img[y1:y2, x1:x2]
        if s < 1.0:
            crop = cv2.resize(crop, (max(1, int(pw * s)), max(1, int(ph * s))), interpolation=cv2.INTER_AREA)
        items.append((crop, x1, y1, s))
    items.sort(key=lambda it: it[0].shape[0], reverse=True)
    canvases, infos = [], []
    cur, n_in = None, 0
    x = 0
    y_bottom = C            # current shelf bottom edge (packing from the bottom of the canvas upwards)
    shelf_h = 0
    for crop, ox, oy, s in items:
        ch, cw = crop.shape[:2]
        if cur is None or n_in >= cfg.capacity:
            cur, n_in, x, y_bottom, shelf_h = np.full((C, C, 3), 114, np.uint8), 0, 0, C, 0
            canvases.append(cur)
        if x + cw > C:
            y_bottom -= shelf_h
            x, shelf_h = 0, 0
        if y_bottom - ch < 0:
            cur, n_in, x, y_bottom, shelf_h = np.full((C, C, 3), 114, np.uint8), 0, 0, C, 0
            canvases.append(cur)
        y0 = y_bottom - ch
        cur[y0:y_bottom, x:x + cw] = crop
        infos.append({'c': len(canvases) - 1, 'x1': x, 'y1': y0, 'x2': x + cw, 'y2': y_bottom, 'ox': ox, 'oy': oy, 's': s})
        x += cw
        shelf_h = max(shelf_h, ch)
        n_in += 1
    return canvases, infos


def run_pan(det, img, cfg: PanCfg, timer):
    h, w = img.shape[:2]
    t = time.perf_counter()
    gb, gs, gc = det([img], conf=min(cfg.conf, cfg.conf_coarse), max_det=cfg.max_det)[0]
    timer['global'] += time.perf_counter() - t
    t = time.perf_counter()
    patches = pan_patches(gb[gs >= cfg.conf_coarse], w, h, cfg)
    timer['plan'] += time.perf_counter() - t
    t = time.perf_counter()
    canvases, infos = pan_pack(img, patches, cfg)
    timer['build'] += time.perf_counter() - t
    pb, ps, pc = [], [], []
    if cfg.fuse_global:
        k = gs >= cfg.conf
        pb.append(gb[k]); ps.append(gs[k]); pc.append(gc[k])
    if canvases:
        t = time.perf_counter()
        res = det(canvases, conf=cfg.conf, imgsz=cfg.imgsz, batch=cfg.batch, max_det=cfg.max_det)
        timer['local'] += time.perf_counter() - t
        t = time.perf_counter()
        for ci, (b, s, c) in enumerate(res):
            if not len(b):
                continue
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            for inf in infos:
                if inf['c'] != ci:
                    continue
                m = (cx >= inf['x1']) & (cx < inf['x2']) & (cy >= inf['y1']) & (cy < inf['y2'])
                if cfg.trunc == 'drop' and m.any():
                    pw, ph = (inf['x2'] - inf['x1']) / inf['s'], (inf['y2'] - inf['y1']) / inf['s']
                    cut = ((b[:, 0] <= inf['x1'] + 5) & (inf['ox'] > 0)) | ((b[:, 1] <= inf['y1'] + 5) & (inf['oy'] > 0)) | \
                        ((b[:, 2] >= inf['x2'] - 5) & (inf['ox'] + pw < w - 1)) | \
                        ((b[:, 3] >= inf['y2'] - 5) & (inf['oy'] + ph < h - 1))
                    m &= ~cut
                if not m.any():
                    continue
                bb = b[m].copy()
                bb[:, [0, 2]] = (np.clip(bb[:, [0, 2]], inf['x1'], inf['x2']) - inf['x1']) / inf['s'] + inf['ox']
                bb[:, [1, 3]] = (np.clip(bb[:, [1, 3]], inf['y1'], inf['y2']) - inf['y1']) / inf['s'] + inf['oy']
                pb.append(bb); ps.append(s[m]); pc.append(c[m])
        timer['merge'] += time.perf_counter() - t
    t = time.perf_counter()
    pb = [x for x in pb if len(x)]
    if not pb:
        timer['merge'] += time.perf_counter() - t
        return empty_dets()
    B = np.concatenate(pb).astype(np.float32)
    S = np.concatenate([x for x in ps if len(x)]).astype(np.float32)
    Cc = np.concatenate([x for x in pc if len(x)]).astype(np.int64)
    if cfg.vote > 0:
        from .thesis_methods import nms_vote
        out = nms_vote(B, S, Cc, cfg.nms_iou, cfg.vote)
        timer['merge'] += time.perf_counter() - t
        return out
    keep = batched_nms(torch.from_numpy(B), torch.from_numpy(S), torch.from_numpy(Cc), cfg.nms_iou).numpy()
    timer['merge'] += time.perf_counter() - t
    return B[keep], S[keep], Cc[keep]


def build(name, arg, overrides, conf):
    if name == 'DAHI':
        cfg = replace(DahiCfg(), **overrides)
        return (lambda det, img, timer, rec=None: run_dahi(det, img, cfg, timer)), asdict(cfg)
    if name == 'PAN':
        cfg = replace(PanCfg(conf=conf), **overrides)
        return (lambda det, img, timer, rec=None: run_pan(det, img, cfg, timer)), asdict(cfg)
    raise ValueError(name)
