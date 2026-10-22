"""Per-view (pre-fusion) detections for HAHI-S and MS, and offline fusion experiments.

collect: runs the same views as the online method but stores every view's detections separately, with
  view kind (0 global / 1 dense window / 2 packed canvas / 3 MS high-res pass), magnification, and whether the box
  touches a view border that is NOT an image border (i.e. the object may be truncated by the crop).
fuse: re-creates the online result (sanity check) and tests alternative fusion rules offline.

  python -m hahi_bench.diag_views collect --data vd_val
  python -m hahi_bench.diag_views fuse --data vd_val
"""
import argparse
import os
import pickle
from dataclasses import replace

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

from . import thesis_methods as TM
from .data import load_dataset
from .detector import Detector
from .metrics import coco_eval
from .run_eval import RUN_ROOT

WEIGHTS = {'vd_val': 'hahi_bench/train_runs/v8s_official/weights/best.pt',
           'vd_testdev': 'hahi_bench/train_runs/v8s_official/weights/best.pt',
           'vid_holdout': 'hahi_bench/train_runs/v8s_official/weights/best.pt'}
CFG_S = replace(TM.standard(TM.THESIS), weight='uniform', snap=False, upscale='none', gcc=1.0, nms_local=0.6,
                nms_final=0.6, fast_merge=True, nms_impl='tv')


def _pack(parts):
    keys = ['b', 's', 'c', 'kind', 'mag', 'edge', 'vid']
    if not parts:
        return {k: np.zeros((0, 4) if k == 'b' else 0) for k in keys}
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def hahi_views(det, img, cfg):
    h, w = img.shape[:2]
    gb, gs, gc = det([img], conf=cfg.out_conf, imgsz=cfg.global_imgsz, max_det=cfg.max_det)[0]
    n = len(gb)
    parts = [{'b': gb.astype(np.float64), 's': gs.astype(np.float64), 'c': gc, 'kind': np.zeros(n, int),
              'mag': np.full(n, cfg.global_imgsz / max(h, w)), 'edge': np.zeros(n, bool), 'vid': np.zeros(n, int)}]
    roi = [b + [s] for b, s in zip(gb.tolist(), gs.tolist()) if s >= cfg.conf_filter]
    regions, remaining = TM.sbsi_regions(roi, w, h, cfg)
    infer = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
    canvases, infos = [], []
    if cfg.packing and remaining:
        canvases, infos = (TM.build_canvases_sra if cfg.packing_mode == 'sra' else TM.build_canvases)(img, remaining, cfg)
    nd = len(infer)
    infer += canvases
    res = det(infer, conf=cfg.out_conf, imgsz=cfg.infer_imgsz, max_det=cfg.max_det) if infer else []
    vid = 1
    for (x1, y1, x2, y2), (b, s, c) in zip(regions, res[:nd]):
        bb = b.astype(np.float64)
        cw, ch = x2 - x1, y2 - y1
        e = ((bb[:, 0] <= 5) & (x1 > 0)) | ((bb[:, 1] <= 5) & (y1 > 0)) | \
            ((bb[:, 2] >= cw - 5) & (x2 < w)) | ((bb[:, 3] >= ch - 5) & (y2 < h))
        e_any = (bb[:, 0] <= 5) | (bb[:, 1] <= 5) | (bb[:, 2] >= cw - 5) | (bb[:, 3] >= ch - 5)
        parts.append({'b': bb + [x1, y1, x1, y1], 's': s.astype(np.float64), 'c': c, 'kind': np.ones(len(b), int),
                      'mag': np.full(len(b), cfg.infer_imgsz / max(cw, ch)), 'edge': e, 'vid': np.full(len(b), vid),
                      'edge_any': e_any})
        vid += 1
    for k, (b, s, c) in enumerate(res[nd:]):
        bb = b.astype(np.float64)
        bcx, bcy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        owner = np.full(len(bb), -1)
        cand = [j for j, i in enumerate(infos) if i['c_idx'] == k]
        for j in cand:
            i = infos[j]
            m = (owner < 0) & (i['cx1'] <= bcx) & (bcx <= i['cx2']) & (i['cy1'] <= bcy) & (bcy <= i['cy2'])
            owner[m] = j
        for j in cand:
            sel = np.flatnonzero(owner == j)
            if not len(sel):
                continue
            i = infos[j]
            q, sc = bb[sel], i['scale']
            ow, oh = (i['cx2'] - i['cx1']) / sc, (i['cy2'] - i['cy1']) / sc
            e_any = (q[:, 0] <= i['cx1'] + 3) | (q[:, 1] <= i['cy1'] + 3) | (q[:, 2] >= i['cx2'] - 3) | \
                (q[:, 3] >= i['cy2'] - 3)
            e = ((q[:, 0] <= i['cx1'] + 3) & (i['ox'] > 0)) | ((q[:, 1] <= i['cy1'] + 3) & (i['oy'] > 0)) | \
                ((q[:, 2] >= i['cx2'] - 3) & (i['ox'] + ow < w - 1)) | ((q[:, 3] >= i['cy2'] - 3) & (i['oy'] + oh < h - 1))
            ob = np.stack([(q[:, 0] - i['cx1']) / sc + i['ox'], (q[:, 1] - i['cy1']) / sc + i['oy'],
                           (q[:, 2] - i['cx1']) / sc + i['ox'], (q[:, 3] - i['cy1']) / sc + i['oy']], 1)
            parts.append({'b': ob, 's': s[sel].astype(np.float64), 'c': c[sel], 'kind': np.full(len(sel), 2),
                          'mag': np.full(len(sel), sc * cfg.infer_imgsz / cfg.canvas), 'edge': e,
                          'vid': np.full(len(sel), vid), 'edge_any': e_any})
            vid += 1
    for p in parts:
        p.setdefault('edge_any', p['edge'])
    keys = ['b', 's', 'c', 'kind', 'mag', 'edge', 'vid', 'edge_any']
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def ms_views(det, img, sizes=(640, 1280), aspect=1.7778):
    h, w = img.shape[:2]
    parts = []
    for j, sz in enumerate(sizes):
        x = img
        if aspect > 0 and j > 0:
            tw, th = max(w, int(np.ceil(h * aspect))), max(h, int(np.ceil(w / aspect)))
            if (tw, th) != (w, h):
                x = np.full((th, tw, 3), 114, np.uint8)
                x[:h, :w] = img
        b, s, c = det([x], conf=0.001, imgsz=sz, max_det=1000)[0]
        b = b.astype(np.float64)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        n = len(b)
        parts.append({'b': b, 's': s.astype(np.float64), 'c': c, 'kind': np.full(n, 0 if j == 0 else 3),
                      'mag': np.full(n, sz / max(x.shape[:2])), 'edge': np.zeros(n, bool), 'vid': np.full(n, j),
                      'edge_any': np.zeros(n, bool)})
    keys = ['b', 's', 'c', 'kind', 'mag', 'edge', 'vid', 'edge_any']
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def collect(args):
    recs, _ = load_dataset(args.data)
    det = Detector(WEIGHTS[args.data], max_det=1000)
    det.warmup()
    out_dir = os.path.join(RUN_ROOT, args.data, 'v8s', 'views')
    os.makedirs(out_dir, exist_ok=True)
    jobs = {'HAHI-S': lambda img: hahi_views(det, img, CFG_S),
            'MS-1280x736': lambda img: ms_views(det, img, (640, 1280), 1.7778)}
    for name, fn in jobs.items():
        path = os.path.join(out_dir, f'{name}.pkl')
        if os.path.exists(path):
            continue
        V = {i: fn(cv2.imread(r.path)) for i, r in enumerate(recs)}
        with open(path, 'wb') as f:
            pickle.dump(V, f)
        print('collected', name, flush=True)


def _nms(b, s, c, thr):
    if not len(b):
        return np.zeros(0, int)
    return batched_nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), torch.from_numpy(c).long(),
                       thr).numpy()


def fuse_online(v, decay=0.8, local_nms=0.6, final_nms=0.6):
    """Exact re-creation of HAHI-S fusion: decay all border-touching local boxes, local NMS, then final NMS."""
    loc = v['kind'] > 0
    s = np.where(loc & v['edge_any'], v['s'] * decay, v['s'])
    li = np.flatnonzero(loc)
    keep_l = li[_nms(v['b'][li], s[li], v['c'][li], local_nms)] if len(li) else li
    idx = np.concatenate([np.flatnonzero(~loc), keep_l])
    k = idx[_nms(v['b'][idx], s[idx], v['c'][idx], final_nms)]
    return v['b'][k], s[k], v['c'][k]


def fuse_rule(v, rule, R=None, decay=0.8, thr=0.6, gamma=1.0, low=0.0):
    """Alternative single-stage fusions over all views."""
    loc = v['kind'] > 0
    s = v['s'].copy()
    keep = np.ones(len(s), bool)
    if 'decay_any' in rule:
        s = np.where(loc & v['edge_any'], s * decay, s)
    if 'drop_trunc' in rule:
        keep &= ~(loc & v['edge'])
    if 'decay_trunc' in rule:
        s = np.where(loc & v['edge'], s * decay, s)
    if 'rel' in rule:
        px = np.sqrt(np.maximum((v['b'][:, 2] - v['b'][:, 0]) * (v['b'][:, 3] - v['b'][:, 1]), 1)) * v['mag']
        r = R(px) / R(np.array([80.0]))[0]
        rank = s * np.clip(r, low, None) ** gamma
    else:
        rank = s
    idx = np.flatnonzero(keep & (s >= 0.001))
    k = idx[_nms(v['b'][idx], rank[idx], v['c'][idx], thr)]
    return v['b'][k], s[k], v['c'][k]


def fuse_vote(v, drop_trunc=True, thr=0.6, vote_thr=0.6, score='max'):
    """NMS with score-weighted box voting over all views (Gidaris & Komodakis style): each kept box is replaced by
    the score-weighted mean of the boxes it suppresses (IoU >= vote_thr), so different views of the same object
    are averaged instead of discarded."""
    loc = v['kind'] > 0
    keep = (v['s'] >= 0.001) & ~(drop_trunc & loc & v['edge'])
    B, S, C = v['b'][keep], v['s'][keep], v['c'][keep]
    outb, outs, outc = [], [], []
    for c in np.unique(C):
        i = np.flatnonzero(C == c)
        b, s = B[i], S[i]
        order = np.argsort(-s)
        b, s = b[order], s[order]
        M = np.nan_to_num(box_iou(torch.from_numpy(b), torch.from_numpy(b)).numpy())
        alive = np.ones(len(b), bool)
        for k in range(len(b)):
            if not alive[k]:
                continue
            sup = alive & (M[k] >= thr)
            vot = alive & (M[k] >= vote_thr)
            sup[k] = vot[k] = True
            w = s[vot]
            outb.append((b[vot] * w[:, None]).sum(0) / w.sum())
            outs.append(s[k] if score == 'max' else float(np.mean(s[vot])))
            outc.append(c)
            alive &= ~sup
    if not outb:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.asarray(outb), np.asarray(outs), np.asarray(outc)


def fuse(args):
    recs, names = load_dataset(args.data)
    d = os.path.join(RUN_ROOT, args.data, 'v8s', 'views')
    R = TM.scale_response(CFG_S.sra_curve)
    idx = list(range(len(recs)))
    for name in ['HAHI-S', 'MS-1280x736']:
        V = pickle.load(open(os.path.join(d, f'{name}.pkl'), 'rb'))
        if args.vote:
            tests = [('drop truncated (reference)', lambda v: fuse_rule(v, 'drop_trunc')),
                     ('drop truncated + box voting 0.6', lambda v: fuse_vote(v, vote_thr=0.6)),
                     ('drop truncated + box voting 0.75', lambda v: fuse_vote(v, vote_thr=0.75)),
                     ('drop truncated + box voting 0.5', lambda v: fuse_vote(v, vote_thr=0.5))]
        else:
            tests = [('online fusion (as method)', lambda v: fuse_online(v))] if name == 'HAHI-S' else \
                [('plain NMS 0.6 (as method)', lambda v: fuse_rule(v, ''))]
            tests += [
                ('single NMS, decay border boxes', lambda v: fuse_rule(v, 'decay_any')),
                ('drop truncated (non-image-border) boxes', lambda v: fuse_rule(v, 'drop_trunc')),
                ('decay truncated only', lambda v: fuse_rule(v, 'decay_trunc')),
                ('reliability-ranked NMS', lambda v: fuse_rule(v, 'rel', R)),
                ('drop truncated + reliability-ranked NMS', lambda v: fuse_rule(v, 'drop_trunc,rel', R)),
                ('drop truncated + rel (gamma 2)', lambda v: fuse_rule(v, 'drop_trunc,rel', R, gamma=2.0)),
            ]
        for label, fn in tests:
            D = {i: tuple(np.asarray(x, dtype) for x, dtype in zip(fn(V[i]), (np.float32, np.float32, np.int64)))
                 for i in idx}
            m, _ = coco_eval(recs, D, idx, n_cls=len(names))
            print(f"{name:<12} {label:<44} mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} "
                  f"APs {m['APs']:.4f} APm {m['APm']:.4f} APl {m['APl']:.4f}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['collect', 'fuse'])
    ap.add_argument('--data', default='vd_val')
    ap.add_argument('--vote', action='store_true', help='box-voting fusion tests')
    a = ap.parse_args()
    {'collect': collect, 'fuse': fuse}[a.cmd](a)
