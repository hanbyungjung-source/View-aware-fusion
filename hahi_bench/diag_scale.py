"""Diagnostics for improving HAHI-S: where does it lose to multi-scale inference?

(1) Detector scale-response: recall of each GT object as a function of its size *in the detector input*
    (pooled over global passes at several input sizes), at a deployment score threshold.
(2) Per-object magnification actually given by HAHI-S (dense window / packed canvas / global only), recovered by
    replaying the HAHI-S planning on the stored 640 global-pass detections (identical to HAHI-S's own global pass).
(3) Recall and AP by native object size for HAHI-S vs MS and single-scale passes.

  python -m hahi_bench.diag_scale --data vd_val
"""
import argparse
import contextlib
import io
import json
import os
import pickle
from dataclasses import replace

import cv2
import numpy as np

from . import thesis_methods as TM
from .data import load_dataset
from .metrics import _ignored_fraction, build_coco
from .run_eval import RUN_ROOT

S = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv'
HAHIS = f'HAHI_{S}_'
CFG_S = replace(TM.standard(TM.THESIS), weight='uniform', snap=False, upscale='none', gcc=1.0, nms_local=0.6,
                nms_final=0.6, fast_merge=True, nms_impl='tv')
SIZE_BINS = [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 1e9]          # sqrt(area) in native pixels
IN_BINS = [0, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 192, 1e9]   # sqrt(area) at detector input


def load_dets(data, tag, run):
    with open(os.path.join(RUN_ROOT, data, tag, 'standard', run, 'dets.pkl'), 'rb') as f:
        return pickle.load(f)


def valid_gt(r):
    if not len(r.boxes):
        return r.boxes, r.cls
    k = _ignored_fraction(r.boxes, r.ignore, r.w, r.h) <= 0.5 if len(r.ignore) else np.ones(len(r.boxes), bool)
    return r.boxes[k], r.cls[k]


def iou(a, b):
    ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix2, iy2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def matched(gb, gc, db, ds, dc, thr=0.3, iou_thr=0.5):
    """Greedy (score-ordered) class-wise matching; returns bool per GT."""
    hit = np.zeros(len(gb), bool)
    keep = ds >= thr
    db, ds, dc = db[keep], ds[keep], dc[keep]
    for c in np.unique(gc):
        gi, di = np.flatnonzero(gc == c), np.flatnonzero(dc == c)
        if not len(di):
            continue
        di = di[np.argsort(-ds[di])]
        M = iou(db[di], gb[gi])
        used = np.zeros(len(gi), bool)
        for k in range(len(di)):
            m = np.where(used, -1, M[k])
            j = int(np.argmax(m))
            if m[j] >= iou_thr:
                used[j] = True
        hit[gi[used]] = True
    return hit


def plan_views(img_shape, gdets, cfg, img=None):
    """Replay HAHI planning; returns list of (x1, y1, x2, y2, magnification, kind)."""
    h, w = img_shape
    gb, gs, _ = gdets
    roi = [b + [s] for b, s in zip(gb.tolist(), gs.tolist()) if s >= cfg.conf_filter]
    regions, remaining = TM.sbsi_regions(roi, w, h, cfg)
    views = [(x1, y1, x2, y2, cfg.infer_imgsz / max(x2 - x1, y2 - y1), 'window') for x1, y1, x2, y2 in regions]
    if cfg.packing and remaining:
        if img is None:
            img = np.zeros((h, w, 3), np.uint8)
        _, infos = TM.build_canvases(img, remaining, cfg)
        k = cfg.infer_imgsz / cfg.canvas
        for i in infos:
            s = i['scale']
            views.append((i['ox'], i['oy'], i['ox'] + (i['cx2'] - i['cx1']) / s, i['oy'] + (i['cy2'] - i['cy1']) / s,
                          s * k, 'packed'))
    return views


def size_ap(recs, dets, n_cls, bins):
    """COCO AP@[.5:.95] per native-size bin (areaRng on sqrt-area bins)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    idx = list(range(len(recs)))
    gt, dt = build_coco(recs, dets, idx, n_cls=n_cls)
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        E = COCOeval(cg, cg.loadRes(dt), 'bbox')
        E.params.imgIds = idx
        E.params.maxDets = [1, 100, 500]
        E.params.areaRng = [[bins[i] ** 2, bins[i + 1] ** 2] for i in range(len(bins) - 1)]
        E.params.areaRngLbl = [f'{bins[i]:g}-{bins[i + 1]:g}' for i in range(len(bins) - 1)]
        E.evaluate()
        E.accumulate()
    P = E.eval['precision'][:, :, :, :, -1]
    out = []
    for a in range(len(bins) - 1):
        q = P[:, :, :, a]
        out.append(float(q[q > -1].mean()) if (q > -1).any() else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='vd_val')
    ap.add_argument('--tag', default='v8s')
    ap.add_argument('--thr', type=float, default=0.3)
    args = ap.parse_args()
    recs, names = load_dataset(args.data)
    gts = [valid_gt(r) for r in recs]
    sizes = np.concatenate([np.sqrt((g[0][:, 2] - g[0][:, 0]) * (g[0][:, 3] - g[0][:, 1])) for g in gts])
    res = {}

    # (1) scale response from single global passes
    pts_in, pts_hit = [], []
    for S_ in [640, 960, 1280, 1536]:
        run = f'global@{S_}'
        if not os.path.exists(os.path.join(RUN_ROOT, args.data, args.tag, 'standard', run, 'dets.pkl')):
            continue
        D = load_dets(args.data, args.tag, run)
        for i, r in enumerate(recs):
            gb, gc = gts[i]
            if not len(gb):
                continue
            s = S_ / max(r.w, r.h)
            pts_in.append(np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])) * s)
            pts_hit.append(matched(gb, gc, *D[i], thr=args.thr))
    pin, phit = np.concatenate(pts_in), np.concatenate(pts_hit)
    curve = []
    for a, b in zip(IN_BINS[:-1], IN_BINS[1:]):
        m = (pin >= a) & (pin < b)
        curve.append({'in_px': f'{a:g}-{b:g}', 'n': int(m.sum()), 'recall': float(phit[m].mean()) if m.any() else None})
    res['scale_response'] = curve
    print('scale response (recall@IoU.5, score>=%.2f) vs object size at detector input:' % args.thr)
    for c in curve:
        print(f"  {c['in_px']:>9} px  n={c['n']:6d}  recall={c['recall'] if c['recall'] is None else round(c['recall'], 3)}")

    # (2) magnification given by HAHI-S
    G = load_dets(args.data, args.tag, 'global@640')
    mags, kinds = [], []
    n_trunc, n_clusters = 0, 0
    for i, r in enumerate(recs):
        gb, _ = gts[i]
        img = cv2.imread(r.path)
        views = plan_views((r.h, r.w), G[i], CFG_S, img)
        gmag = 640 / max(r.w, r.h)
        best = np.full(len(gb), gmag)
        kind = np.array(['global'] * len(gb), dtype=object)
        for x1, y1, x2, y2, m, kd in views:
            inside = (gb[:, 0] >= x1 - 1) & (gb[:, 1] >= y1 - 1) & (gb[:, 2] <= x2 + 1) & (gb[:, 3] <= y2 + 1)
            upd = inside & (m > best)
            best[upd] = m
            kind[upd] = kd
            n_clusters += kd == 'packed'
        mags.append(best)
        kinds.append(kind)
    mags, kinds = np.concatenate(mags), np.concatenate(kinds)
    table = []
    for a, b in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        m = (sizes >= a) & (sizes < b)
        if not m.any():
            continue
        row = {'native_px': f'{a:g}-{b:g}', 'n': int(m.sum()),
               'frac_window': float(np.mean(kinds[m] == 'window')), 'frac_packed': float(np.mean(kinds[m] == 'packed')),
               'frac_global_only': float(np.mean(kinds[m] == 'global')),
               'median_mag': float(np.median(mags[m])), 'median_input_px_hahis': float(np.median(sizes[m] * mags[m])),
               'median_input_px_ms1280': float(np.median(sizes[m] * np.concatenate(
                   [np.full(len(gts[i][0]), 1280 / max(r.w, r.h)) for i, r in enumerate(recs)])[m]))}
        table.append(row)
    res['hahis_magnification'] = table
    print('\nHAHI-S view coverage by native object size:')
    for t in table:
        print(f"  {t['native_px']:>8}  n={t['n']:6d}  window {t['frac_window']:.2f}  packed {t['frac_packed']:.2f}  "
              f"global-only {t['frac_global_only']:.2f}  median mag {t['median_mag']:.2f}  "
              f"input px HAHI-S {t['median_input_px_hahis']:.1f} vs MS-1280 {t['median_input_px_ms1280']:.1f}")

    # (3) recall and AP per native size bin
    runs = ['global@640', 'global@1280', 'MS@640x1280', 'HAHI', HAHIS, f'HAHI_{S},route_ratio=0.1_']
    res['by_size'] = {}
    print('\nrecall@IoU.5 (score>=%.2f) / AP@[.5:.95] by native size:' % args.thr)
    print('  ' + ' '.join(f'{f"{a:g}-{b:g}":>13}' for a, b in zip(SIZE_BINS[:-1], SIZE_BINS[1:])))
    for run in runs:
        if not os.path.exists(os.path.join(RUN_ROOT, args.data, args.tag, 'standard', run, 'dets.pkl')):
            continue
        D = load_dets(args.data, args.tag, run)
        hit = np.concatenate([matched(gts[i][0], gts[i][1], *D[i], thr=args.thr) if len(gts[i][0]) else np.zeros(0, bool)
                              for i in range(len(recs))])
        aps = size_ap(recs, D, len(names), SIZE_BINS)
        rec = [float(hit[(sizes >= a) & (sizes < b)].mean()) if ((sizes >= a) & (sizes < b)).any() else float('nan')
               for a, b in zip(SIZE_BINS[:-1], SIZE_BINS[1:])]
        res['by_size'][run] = {'recall': rec, 'ap': aps}
        print(f'  {run[:40]:<40}')
        print('    recall ' + ' '.join(f'{v:13.3f}' for v in rec))
        print('    AP     ' + ' '.join(f'{v:13.4f}' for v in aps))
    with open(os.path.join(RUN_ROOT, args.data, args.tag, 'standard', 'diag_scale.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
