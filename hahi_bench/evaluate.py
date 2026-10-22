"""Evaluate stored runs: COCO metrics (ALL/HR/LR) + cost statistics, cached per run directory.

Example: python -m hahi_bench.evaluate --data vd_val --tag v8s --protocol standard
"""
import argparse
import json
import os
import pickle

import numpy as np

from .data import load_dataset
from .metrics import BootstrapTable, coco_eval
from .run_eval import RUN_ROOT

SUBSETS = ('ALL', 'HR', 'LR', 'FHD')


def subset_indices(recs, idx, name):
    if name == 'ALL':
        return list(idx)
    if name == 'FHD':          # FHD-class frames (1916x1078 and up); thesis HR threshold excludes 1916x1078
        return [i for i in idx if recs[i].w >= 1900]
    if name in ('W169', 'W43'):  # frames of one aspect ratio (a deployed drone camera has a single fixed aspect)
        target = 16 / 9 if name == 'W169' else 4 / 3
        return [i for i in idx if abs(recs[i].w / recs[i].h - target) < 0.03]
    if name in ('LE1920', 'GT1920'):  # frame width up to / above 1920 px (exploratory, VID hold-out)
        return [i for i in idx if (recs[i].w <= 1920) == (name == 'LE1920')]
    hr = name == 'HR'
    return [i for i in idx if recs[i].is_hr == hr]


def cost_stats(per, idx_set):
    rows = [p for p in per if p['i'] in idx_set]
    if not rows:
        return {}
    a = lambda k: np.array([r[k] for r in rows])
    tot = a('total') * 1000
    return {'n_img': len(rows), 'total_ms': float(tot.mean()), 'total_p50': float(np.median(tot)),
            'total_p95': float(np.percentile(tot, 95)), 'infer_ms': float(a('infer').mean() * 1000),
            'global_ms': float(a('global').mean() * 1000), 'plan_ms': float(a('plan').mean() * 1000),
            'build_ms': float(a('build').mean() * 1000), 'local_ms': float(a('local').mean() * 1000),
            'merge_ms': float(a('merge').mean() * 1000), 'n_inf': float(a('n_inf').mean()),
            'gflops': float(a('gflops').mean()), 'n_det': float(a('n_det').mean())}


def evaluate_run(run_dir, recs, score_thr=0.0, use_ignore=True, force=False, n_cls=10):
    key = f'metrics_thr{score_thr:g}_ign{int(use_ignore)}.json'
    cache = os.path.join(run_dir, key)
    if os.path.exists(cache) and not force:
        with open(cache) as f:
            return json.load(f)
    with open(os.path.join(run_dir, 'dets.pkl'), 'rb') as f:
        dets = pickle.load(f)
    with open(os.path.join(run_dir, 'per_image.json')) as f:
        per = json.load(f)
    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)
    idx = sorted(dets)
    out = {'spec': meta['spec'], 'peak_alloc_mb': meta['peak_alloc_mb'], 'peak_reserved_mb': meta['peak_reserved_mb']}
    # One COCOeval over all images; subset APs are re-accumulated from the per-image matches with 0/1 image
    # weights, which is mathematically identical to running COCOeval on the subset (verified for ALL below).
    m_all, E = coco_eval(recs, dets, idx, score_thr=score_thr, use_ignore=use_ignore, n_cls=n_cls, keep_eval=True)
    if E is None:
        return out
    tabs = {a: BootstrapTable(E, a) for a in ('all', 'small', 'medium', 'large')}
    assert abs(tabs['all'].ap() - m_all['mAP']) < 1e-6 and abs(tabs['small'].ap() - m_all['APs']) < 1e-6
    pos = tabs['all'].img_pos
    for sub in SUBSETS:
        si = subset_indices(recs, idx, sub)
        if not si:
            continue
        if sub == 'ALL':
            m = m_all
        else:
            w = np.zeros(len(pos))
            w[[pos[i] for i in si]] = 1.0
            m = {'mAP': tabs['all'].ap(w), 'AP50': tabs['all'].ap(w, 0), 'AP75': tabs['all'].ap(w, 5),
                 'APs': tabs['small'].ap(w), 'APm': tabs['medium'].ap(w), 'APl': tabs['large'].ap(w)}
        out[sub] = {**m, **cost_stats(per, set(si))}
    with open(cache, 'w') as f:
        json.dump(out, f, indent=1)
    return out


def list_runs(data, tag, protocol):
    root = os.path.join(RUN_ROOT, data, tag, protocol)
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.exists(os.path.join(root, d, 'meta.json'))]


def print_table(results, subsets=SUBSETS):
    hdr = f"{'Method':<46}{'Sub':<4}{'N':>5} {'mAP':>7}{'AP50':>7}{'APs':>7}{'APm':>7}{'APl':>7} " \
          f"{'#inf':>5}{'GFLOP':>7}{'inf ms':>8}{'tot ms':>8}{'VRAM':>7}"
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        for k, sub in enumerate(subsets):
            if sub not in r:
                continue
            s = r[sub]
            name = r['spec'] if k == 0 else ''
            vram = f"{r['peak_alloc_mb']:7.1f}" if k == 0 else ''
            print(f"{name:<46}{sub:<4}{s['n_img']:>5} {s['mAP']:.4f} {s['AP50']:.4f} {s['APs']:.4f} {s['APm']:.4f} "
                  f"{s['APl']:.4f} {s['n_inf']:5.2f}{s['gflops']:7.1f}{s['infer_ms']:8.1f}{s['total_ms']:8.1f}{vram}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='standard')
    ap.add_argument('--thr', type=float, default=0.0)
    ap.add_argument('--no-ignore', action='store_true')
    ap.add_argument('--subsets', default='ALL,HR,LR')
    ap.add_argument('--filter', default='')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    recs, names = load_dataset(args.data)
    use_ignore = not args.no_ignore
    res = []
    for rd in list_runs(args.data, args.tag, args.protocol):
        if args.filter and args.filter not in os.path.basename(rd):
            continue
        res.append(evaluate_run(rd, recs, args.thr, use_ignore, args.force, n_cls=len(names)))
    print_table(res, args.subsets.split(','))


if __name__ == '__main__':
    main()
