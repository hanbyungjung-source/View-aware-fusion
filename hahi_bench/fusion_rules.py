"""Realisable "which view to trust" rules for box voting (REPORT section 22).

headroom.py shows that choosing, per object, the best view's box would add about +0.02 mAP (loc_view oracle). This
script tests training-free voting weights that try to realise part of it, on the stored per-view detections
(diag_views collect). Every rule keeps the same NMS (IoU 0.6), the same kept boxes and the same scores; only the
coordinates of each kept box change (score-weighted mean of its cluster with a different weight per candidate).

  python -m hahi_bench.fusion_rules --data vd_val
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

from . import thesis_methods as TM
from .data import load_dataset
from .metrics import coco_eval
from .run_eval import RUN_ROOT

R_CURVE = TM.scale_response(TM.HahiCfg().sra_curve)


def candidates(v, method):
    keep = v['s'] >= 0.001
    if method == 'HAHI-S':
        keep &= ~((v['kind'] > 0) & v['edge'])
    return {k: v[k][keep] for k in ('b', 's', 'c', 'vid', 'mag')}


def vote(v, method, rule, thr=0.6):
    p = candidates(v, method)
    B, S, C, V, G = p['b'].astype(np.float64), p['s'].astype(np.float64), p['c'].astype(np.int64), p['vid'], p['mag']
    if not len(B):
        return B.astype(np.float32), S.astype(np.float32), C
    tb, ts, tc = torch.from_numpy(B), torch.from_numpy(S), torch.from_numpy(C)
    keep = batched_nms(tb.float(), ts.float(), tc, 0.6)
    off = tc.double()[:, None] * (tb.max() + 1)
    M = torch.nan_to_num(box_iou(tb[keep] + off[keep], tb + off))
    M[torch.arange(len(keep)), keep] = 1.0
    mask = M >= thr
    side = np.sqrt(np.maximum((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]), 1.0))
    if rule == 'score':
        w = S
    elif rule.startswith('score^'):
        w = S ** float(rule[6:])
    elif rule.startswith('score*mag^'):
        w = S * G ** float(rule[10:])
    elif rule == 'score*R':
        w = S * R_CURVE(side * G)
    elif rule == 'score*R^2':
        w = S * R_CURVE(side * G) ** 2
    elif rule in ('topview', 'topview*R'):
        w = S if rule == 'topview' else S * R_CURVE(side * G)
        Sm = torch.where(mask, ts[None].expand_as(M), torch.full_like(M, -1.0))
        Wm = torch.zeros_like(M)
        tw = torch.from_numpy(w)
        for vv in np.unique(V):
            cols = torch.from_numpy(np.flatnonzero(V == vv))
            sub = Sm[:, cols]
            mx, am = sub.max(1)
            rows = torch.nonzero(mx >= 0).flatten()
            Wm[rows, cols[am[rows]]] = tw[cols[am[rows]]]
        nb = (Wm @ tb) / Wm.sum(1, keepdim=True)
        k = keep.numpy()
        return nb.numpy().astype(np.float32), S[k].astype(np.float32), C[k]
    else:
        raise ValueError(rule)
    W = mask.double() * torch.from_numpy(w)[None]
    nb = (W @ tb) / W.sum(1, keepdim=True)
    k = keep.numpy()
    return nb.numpy().astype(np.float32), S[k].astype(np.float32), C[k]


RULES = ['score', 'score^2', 'score^4', 'score*mag^1', 'score*mag^2', 'score*R', 'score*R^2', 'topview', 'topview*R']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='vd_val')
    ap.add_argument('--rules', default=','.join(RULES))
    ap.add_argument('--methods', default='HAHI-S,MS-1280x736')
    a = ap.parse_args()
    recs, names = load_dataset(a.data)
    d = os.path.join(RUN_ROOT, a.data, 'v8s', 'views')
    idx = list(range(len(recs)))
    out_path = os.path.join(d, 'fusion_rules.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for method in a.methods.split(','):
        V = pickle.load(open(os.path.join(d, f'{method}.pkl'), 'rb'))
        for rule in a.rules.split(','):
            D = {i: vote(V[i], method, rule) for i in idx}
            m, _ = coco_eval(recs, D, idx, n_cls=len(names))
            res.setdefault(method, {})[rule] = m
            print(f"{method:<12} {rule:<12} mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} "
                  f"APs {m['APs']:.4f} APm {m['APm']:.4f} APl {m['APl']:.4f}", flush=True)
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
