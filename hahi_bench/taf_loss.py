"""Recall lost by truncation-aware filtering (TAF): objects whose only correct detection was a removed border box.

Uses the per-view detections of HAHI-S (diag_views collect). For every ground-truth object we compare
  * the candidates the fusion receives after TAF (global view + local boxes not touching an interior crop border),
  * all candidates before TAF.
An object is "lost to TAF" if it is not matched after fusion, no post-TAF candidate matches it (IoU >= 0.5, same class),
but a removed border box does. The AP effect is measured by adding back, for each such object, the best removed box
with its own score (an optimistic restore: a real rule would not know which border boxes are correct).

  python -m hahi_bench.taf_loss --data vd_val
"""
import argparse
import json
import os
import pickle

import numpy as np

from .data import load_dataset
from .headroom import SIZE_EDGES, SIZE_NAMES, _iou, full_of, fuse_online, greedy_match, pool_of
from .metrics import _ignored_fraction, coco_eval
from .run_eval import RUN_ROOT

WIDTH_EDGES = [0, 1400, 1920, np.inf]
WIDTH_NAMES = ['<=1400 px', '1401-1920 px', '>1920 px']


def analyse(r, v, thr):
    gb, gc = r.boxes.astype(np.float64), r.cls.astype(np.int64)
    fb, fs, fc = (np.asarray(x) for x in fuse_online(v, 'HAHI-S'))
    fb, fs, fc = fb.astype(np.float64), fs.astype(np.float64), fc.astype(np.int64)
    pb, ps, pc, _ = pool_of(v, 'HAHI-S')
    ab, as_, ac, _ = full_of(v)
    k = ps >= thr
    pb, pc = pb[k], pc[k]
    k = as_ >= thr
    ab, as_, ac = ab[k], as_[k], ac[k]
    if len(r.ignore):
        kg = _ignored_fraction(gb.astype(np.float32), r.ignore, r.w, r.h) <= 0.5
        gb, gc = gb[kg], gc[kg]
        kd = _ignored_fraction(fb.astype(np.float32), r.ignore, r.w, r.h) <= 0.5
        fb, fs, fc = fb[kd], fs[kd], fc[kd]

    def best(b, c):
        if not len(b) or not len(gb):
            return np.zeros(len(gb)), None
        I = _iou(b, gb)
        I[c[:, None] != gc[None]] = 0.0
        return I.max(0), I
    pool_best, _ = best(pb, pc)
    all_best, IA = best(ab, ac)
    _, gm = greedy_match(fb, fs, fc, gb, gc)
    lost = (gm < 0) & (pool_best < 0.5) & (all_best >= 0.5)
    add = [int(IA[:, j].argmax()) for j in np.flatnonzero(lost)]
    restored = (np.concatenate([fb, ab[add]]) if add else fb, np.concatenate([fs, as_[add]]) if add else fs,
                np.concatenate([fc, ac[add]]) if add else fc)
    size = np.sqrt(np.maximum((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]), 0))
    return lost, size, restored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='vd_val')
    ap.add_argument('--thr', type=float, default=0.001, help='score floor for counting candidates')
    a = ap.parse_args()
    recs, names = load_dataset(a.data)
    d = os.path.join(RUN_ROOT, a.data, 'v8s', 'views')
    V = pickle.load(open(os.path.join(d, 'HAHI-S.pkl'), 'rb'))
    idx = sorted(V)
    n_gt = np.zeros(len(SIZE_NAMES), int)
    n_lost = np.zeros(len(SIZE_NAMES), int)
    w_gt = np.zeros(len(WIDTH_NAMES), int)
    w_lost = np.zeros(len(WIDTH_NAMES), int)
    base, rest = {}, {}
    for i in idx:
        r = recs[i]
        lost, size, restored = analyse(r, V[i], a.thr)
        n_gt += np.histogram(size, SIZE_EDGES)[0]
        n_lost += np.histogram(size[lost], SIZE_EDGES)[0]
        wb = int(np.searchsorted(WIDTH_EDGES, r.w, side='left')) - 1
        w_gt[wb] += len(size)
        w_lost[wb] += int(lost.sum())
        base[i] = tuple(np.asarray(x) for x in fuse_online(V[i], 'HAHI-S'))
        rest[i] = tuple(x.astype(t) for x, t in zip(restored, (np.float32, np.float32, np.int64)))
    m0, _ = coco_eval(recs, base, idx, n_cls=len(names))
    m1, _ = coco_eval(recs, rest, idx, n_cls=len(names))
    out = {'data': a.data, 'thr': a.thr, 'n_images': len(idx), 'n_gt': int(n_gt.sum()), 'n_lost': int(n_lost.sum()),
           'size': {s: [int(l), int(g)] for s, l, g in zip(SIZE_NAMES, n_lost, n_gt)},
           'width': {s: [int(l), int(g)] for s, l, g in zip(WIDTH_NAMES, w_lost, w_gt)},
           'AP_fused': m0['mAP'], 'AP_restored': m1['mAP'], 'AP50_fused': m0['AP50'], 'AP50_restored': m1['AP50'],
           'APs_fused': m0['APs'], 'APs_restored': m1['APs']}
    print(json.dumps(out, indent=1), flush=True)
    path = os.path.join(d, 'taf_loss.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[f'thr{a.thr:g}'] = out
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
