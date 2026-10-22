"""Where is the remaining error, and how much of it can fusion still fix? (REPORT section 22)

Uses the per-view (pre-fusion) detections stored by `diag_views collect`. For each method the online fusion is
re-created (TAF + box voting for HAHI-S, box voting for MS), the fused result is matched to GT (IoU 0.5, COCO greedy
order), and TIDE-style oracles (Bolya et al., ECCV 2020) are applied one at a time. "Pool" = every candidate box of
every view that the fusion receives (after TAF, score >= 0.001).

  dupe       remove false positives that duplicate an already-matched GT
  loc_pool   replace each detection by the box of its own cluster (same class, IoU >= 0.5 with it, any view) that best
             overlaps its GT -> the best box a better box-fusion rule could have produced from these candidates
             (optimistic: best-of-N over many low-score candidates)
  loc_view   as loc_pool, but only the top-scored candidate of each view in the cluster may be chosen
             -> "which view to trust" ceiling
  cls_pool   a class-confused detection takes its GT's class if some view proposed that class for an overlapping box
             (IoU >= 0.5) -> ceiling of cross-view class consensus
  miss_pool  add, for every GT missed after fusion, the best same-class pool candidate (IoU >= 0.5) with its own score
             -> recall that the fusion threw away
  taf_lost   add, for every GT that only TAF-removed border boxes found (IoU >= 0.5), the best such box
             -> recall lost to truncation-aware filtering (HAHI-S only)
  fusion     dupe + loc_pool + miss_pool together -> ceiling of any fusion rule over these candidates
  cls / bkg  remove class-confusion / background false positives -> scoring errors of the detector (reference)
  miss_det   remove GT that no view detected before TAF (IoU >= 0.5) -> recall the detector never produced (reference)

  python -m hahi_bench.headroom --data vd_val
"""
import argparse
import json
import os
import pickle
from dataclasses import replace

import numpy as np
import torch
from torchvision.ops import box_iou

from . import thesis_methods as TM
from .data import load_dataset
from .metrics import _ignored_fraction, coco_eval
from .run_eval import RUN_ROOT

SIZE_EDGES = [0, 16, 32, 96, np.inf]
SIZE_NAMES = ['<16 px', '16-32 px', '32-96 px', '>=96 px']


def _iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    return np.nan_to_num(box_iou(torch.from_numpy(a).double(), torch.from_numpy(b).double()).numpy())


def pool_of(v, method):
    """(boxes, scores, classes, view ids) of every candidate the fusion receives."""
    keep = v['s'] >= 0.001
    if method == 'HAHI-S':
        keep &= ~((v['kind'] > 0) & v['edge'])
    return (v['b'][keep].astype(np.float64), v['s'][keep].astype(np.float64), v['c'][keep].astype(np.int64),
            v['vid'][keep].astype(np.int64))


def full_of(v):
    """Every candidate of every view before TAF (score >= 0.001): defines what the detector found at all."""
    keep = v['s'] >= 0.001
    return (v['b'][keep].astype(np.float64), v['s'][keep].astype(np.float64), v['c'][keep].astype(np.int64),
            v['vid'][keep].astype(np.int64))


def fuse_online(v, method):
    """Fusion as in the online method: TAF (HAHI-S only) + NMS 0.6 + box voting 0.6."""
    return TM.nms_vote(*pool_of(v, method)[:3], 0.6, 0.6)


def fuse_plain(v, method):
    """Fusion without the plug-in: thesis-code merge for HAHI-S, plain NMS 0.6 for MS."""
    if method == 'HAHI-S':
        from .diag_views import fuse_online as hahi_merge
        return tuple(np.asarray(x) for x in hahi_merge(v))
    B, S, C = v['b'], v['s'], v['c'].astype(np.int64)
    k = S >= 0.001
    return TM.nms_vote(B[k], S[k], C[k], 0.6, 0.0)


def greedy_match(db, ds, dc, gb, gc, thr=0.5):
    """COCO matching at one IoU threshold: detections in descending score order take the best unmatched GT."""
    I = _iou(db, gb)
    I[dc[:, None] != gc[None]] = 0.0
    dm, gm = np.full(len(db), -1), np.full(len(gb), -1)
    if not len(gb):
        return dm, gm
    for k in np.argsort(-ds, kind='stable'):
        cand = np.where(gm < 0, I[k], 0.0)
        j = int(cand.argmax())
        if cand[j] >= thr:
            dm[k], gm[j] = j, k
    return dm, gm


def analyse_image(r, det, pool, full=None):
    """Returns per-image oracle detections / GT masks and coverage statistics. `full` = candidates before TAF."""
    gb_all, gc_all = r.boxes.astype(np.float64), r.cls.astype(np.int64)
    kg = np.ones(len(gb_all), bool)
    db, ds, dc = (np.asarray(x) for x in det)
    db, ds, dc = db.astype(np.float64), ds.astype(np.float64), dc.astype(np.int64)
    pb, ps, pc, pv = pool
    fb, fs, fc, _ = full if full is not None else pool
    if len(r.ignore):
        kg = _ignored_fraction(gb_all.astype(np.float32), r.ignore, r.w, r.h) <= 0.5
        kd = _ignored_fraction(db.astype(np.float32), r.ignore, r.w, r.h) <= 0.5
        db, ds, dc = db[kd], ds[kd], dc[kd]
        kp = _ignored_fraction(pb.astype(np.float32), r.ignore, r.w, r.h) <= 0.5
        pb, ps, pc, pv = pb[kp], ps[kp], pc[kp], pv[kp]
        kf = _ignored_fraction(fb.astype(np.float32), r.ignore, r.w, r.h) <= 0.5
        fb, fs, fc = fb[kf], fs[kf], fc[kf]
    gidx = np.flatnonzero(kg)
    gb, gc = gb_all[kg], gc_all[kg]

    dm, gm = greedy_match(db, ds, dc, gb, gc)
    Iall = _iou(db, gb)
    same = np.where(dc[:, None] == gc[None], Iall, 0.0)
    other = np.where(dc[:, None] != gc[None], Iall, 0.0)
    typ = np.full(len(db), 'tp', dtype=object)
    for k in np.flatnonzero(dm < 0):
        sm = same[k].max() if len(gb) else 0.0
        om = other[k].max() if len(gb) else 0.0
        typ[k] = 'dupe' if sm >= 0.5 else 'cls' if om >= 0.5 else 'loc' if sm >= 0.1 else 'bkg'

    PG = _iou(pb, gb)
    PG[pc[:, None] != gc[None]] = 0.0
    pool_best = PG.max(0) if len(pb) else np.zeros(len(gb))
    FG = _iou(fb, gb)
    FG[fc[:, None] != gc[None]] = 0.0
    full_best = FG.max(0) if len(fb) else np.zeros(len(gb))
    fused_best = same.max(0) if len(db) else np.zeros(len(gb))

    # loc_pool: each TP / loc-FP takes the best box of its own cluster for its target GT
    DPa = _iou(db, pb)
    DP = np.where(dc[:, None] == pc[None], DPa, 0.0)
    loc_b, locv_b = db.copy(), db.copy()
    targeted = set()
    n_cl, n_view = [], []
    for k in range(len(db)):
        if typ[k] == 'tp':
            g = dm[k]
        elif typ[k] == 'loc':
            g = int(same[k].argmax())
            if gm[g] >= 0 or g in targeted:
                continue
            targeted.add(g)
        else:
            continue
        cl = np.flatnonzero(DP[k] >= 0.5)
        if not len(cl):
            continue
        best = cl[int(PG[cl, g].argmax())]
        if PG[best, g] > same[k, g]:
            loc_b[k] = pb[best]
        # loc_view: only the top-scored candidate of each view may be chosen ("which view to trust")
        order = cl[np.argsort(-ps[cl], kind='stable')]
        _, first = np.unique(pv[order], return_index=True)
        top = order[first]
        bv = top[int(PG[top, g].argmax())]
        if PG[bv, g] > same[k, g]:
            locv_b[k] = pb[bv]
        if typ[k] == 'tp':
            n_cl.append(len(cl))
            n_view.append(len(top))

    # cls_pool: a class-confused detection takes its GT's class if some view proposed that class for the same box
    cls_c = dc.copy()
    for k in np.flatnonzero(typ == 'cls'):
        g = int(other[k].argmax())
        if gm[g] < 0 and ((DPa[k] >= 0.5) & (pc == gc[g])).any():
            cls_c[k] = gc[g]

    # miss_pool: GT missed after fusion but present in the pool
    add = []
    for j in np.flatnonzero(gm < 0):
        if pool_best[j] >= 0.5:
            add.append((j, int(PG[:, j].argmax())))
    add_nt = [p for j, p in add if j not in targeted]

    def cat(b, s, c, extra):
        if not extra:
            return b, s, c
        e = np.asarray(extra)
        return np.concatenate([b, pb[e]]), np.concatenate([s, ps[e]]), np.concatenate([c, pc[e]])

    keep_nd = typ != 'dupe'
    # taf_lost: GT that only TAF-removed border boxes found (IoU >= 0.5); the oracle restores the best such box
    taf_lost = (gm < 0) & (pool_best < 0.5) & (full_best >= 0.5)
    add_taf = [int(FG[:, j].argmax()) for j in np.flatnonzero(taf_lost)]
    taf_b = np.concatenate([db, fb[add_taf]]) if add_taf else db
    taf_s = np.concatenate([ds, fs[add_taf]]) if add_taf else ds
    taf_c = np.concatenate([dc, fc[add_taf]]) if add_taf else dc
    out = {
        'base': (db, ds, dc),
        'dupe': (db[keep_nd], ds[keep_nd], dc[keep_nd]),
        'loc_pool': (loc_b, ds, dc),
        'loc_view': (locv_b, ds, dc),
        'cls_pool': (db, ds, cls_c),
        'miss_pool': cat(db, ds, dc, [p for _, p in add]),
        'taf_lost': (taf_b, taf_s, taf_c),
        'fusion': cat(loc_b[keep_nd], ds[keep_nd], dc[keep_nd], add_nt),
        'cls': (db[typ != 'cls'], ds[typ != 'cls'], dc[typ != 'cls']),
        'bkg': (db[typ != 'bkg'], ds[typ != 'bkg'], dc[typ != 'bkg']),
    }
    out = {k: (b.astype(np.float32), s.astype(np.float32), c.astype(np.int64)) for k, (b, s, c) in out.items()}
    # detector misses are defined on all views before TAF, so that objects removed by TAF are not counted here
    det_miss = (gm < 0) & (full_best < 0.5)
    keep_gt = np.ones(len(gb_all), bool)
    keep_gt[gidx[det_miss]] = False
    size = np.sqrt(np.maximum((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]), 0))
    stats = {'n_gt': len(gb), 'fused50': int((gm >= 0).sum()), 'pool50': int((pool_best >= 0.5).sum()),
             'fused75': int((fused_best >= 0.75).sum()), 'pool75': int((pool_best >= 0.75).sum()),
             'det_miss': int(det_miss.sum()), 'fusion_miss': int(((gm < 0) & (pool_best >= 0.5)).sum()),
             'taf_lost': int(taf_lost.sum()), 'taf_lost_size': np.histogram(size[taf_lost], SIZE_EDGES)[0].tolist(),
             'det_miss_size': np.histogram(size[det_miss], SIZE_EDGES)[0].tolist(),
             'gt_size': np.histogram(size, SIZE_EDGES)[0].tolist(),
             'tp_cluster_sum': int(sum(n_cl)), 'tp_views_sum': int(sum(n_view)), 'tp_with_cluster': len(n_cl),
             'types': {t: int((typ == t).sum()) for t in ['tp', 'dupe', 'cls', 'loc', 'bkg']}}
    return out, keep_gt, stats


def run(args):
    recs, names = load_dataset(args.data)
    d = os.path.join(RUN_ROOT, args.data, 'v8s', 'views')
    idx = list(range(len(recs)))
    res = {}
    for method in ['HAHI-S', 'MS-1280x736']:
        V = pickle.load(open(os.path.join(d, f'{method}.pkl'), 'rb'))
        plain = {i: tuple(np.asarray(x) for x in fuse_plain(V[i], method)) for i in idx}
        m_plain, _ = coco_eval(recs, plain, idx, n_cls=len(names))
        O, R2, agg = {}, [], None
        for i in idx:
            out, keep_gt, st = analyse_image(recs[i], fuse_online(V[i], method), pool_of(V[i], method), full_of(V[i]))
            for k, v in out.items():
                O.setdefault(k, {})[i] = v
            R2.append(replace(recs[i], boxes=recs[i].boxes[keep_gt], cls=recs[i].cls[keep_gt]))
            if agg is None:
                agg = st
            else:
                for k, v in st.items():
                    if isinstance(v, dict):
                        for t in v:
                            agg[k][t] += v[t]
                    elif isinstance(v, list):
                        agg[k] = [a + b for a, b in zip(agg[k], v)]
                    else:
                        agg[k] += v
        rows = {'plain': m_plain}
        for k in ['base', 'dupe', 'loc_pool', 'loc_view', 'cls_pool', 'miss_pool', 'taf_lost', 'fusion', 'cls', 'bkg']:
            rows[k], _ = coco_eval(recs, O[k], idx, n_cls=len(names))
        rows['miss_det'], _ = coco_eval(R2, O['base'], idx, n_cls=len(names))
        res[method] = {'metrics': rows, 'stats': agg}
        for k, m in rows.items():
            print(f"{method:<12} {k:<10} mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} "
                  f"APs {m['APs']:.4f}", flush=True)
        print(method, json.dumps(agg), flush=True)
    with open(os.path.join(d, 'headroom.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='vd_val')
    run(ap.parse_args())
