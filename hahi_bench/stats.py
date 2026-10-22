"""Paired image-level bootstrap of AP differences (reference minus other).

Each run is re-accumulated once per shared image resample (cached as boot_reps_*.npy, parallel over runs);
paired differences for any number of method pairs are then formed from those caches.

Example:
  python -m hahi_bench.stats --data vd_testdev --tag v8s --refs "HAHI" --others "MS@640x1280;global@1280" --n 1000
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .data import load_dataset
from .evaluate import list_runs, subset_indices
from .metrics import BootstrapTable, coco_eval


def tables_for(run_dir, recs, subset, score_thr=0.0, n_cls=10):
    """(all-area table, small-area table), cached on disk next to the run."""
    cache = os.path.join(run_dir, f'boot_tables_{subset}_thr{score_thr:g}.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    with open(os.path.join(run_dir, 'dets.pkl'), 'rb') as f:
        dets = pickle.load(f)
    idx = subset_indices(recs, sorted(dets), subset)
    m, E = coco_eval(recs, dets, idx, score_thr=score_thr, keep_eval=True, n_cls=n_cls)
    t_all, t_small = BootstrapTable(E, 'all'), BootstrapTable(E, 'small')
    assert abs(t_all.ap() - m['mAP']) < 1e-6, (t_all.ap(), m['mAP'])
    assert abs(t_all.ap(iou_idx=0) - m['AP50']) < 1e-6, (t_all.ap(iou_idx=0), m['AP50'])
    assert abs(t_small.ap() - m['APs']) < 1e-6, (t_small.ap(), m['APs'])
    with open(cache, 'wb') as f:
        pickle.dump((t_all, t_small), f)
    return t_all, t_small


def _build(args):
    run_dir, data, subset, thr = args
    recs, names = load_dataset(data)
    tables_for(run_dir, recs, subset, thr, len(names))
    return run_dir


def _reps(args):
    """Per-replicate (mAP, AP50, APs) of one run; row 0 is the full sample. All runs share the same image
    resamples (same seed), so paired differences are formed afterwards without re-evaluating either method.
    cluster=True resamples whole sequences (record name prefix before '/') instead of images, for video data
    whose consecutive frames are correlated."""
    run_dir, data, subset, thr, n, seed, cluster = args
    cache = os.path.join(run_dir, f'boot_reps_{subset}_thr{thr:g}_n{n}_s{seed}{"_seq" if cluster else ""}.npy')
    if os.path.exists(cache):
        return run_dir, np.load(cache)
    recs, names = load_dataset(data)
    ta, ts = tables_for(run_dir, recs, subset, thr, len(names))
    n_img = len(ta.img_pos)
    rng = np.random.default_rng(seed)
    if cluster:
        pos2img = sorted(ta.img_pos, key=ta.img_pos.get)
        groups = [recs[i].name.split('/')[0] for i in pos2img]
        uniq = sorted(set(groups))
        gidx = np.asarray([uniq.index(g) for g in groups])
    out = np.empty((n + 1, 3))
    for k in range(n + 1):
        if k == 0:
            w = None
        elif cluster:
            cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            w = cnt[gidx].astype(np.float64)
        else:
            w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
        a, s = ta.per_cat_tp(w), ts.per_cat_tp(w)
        out[k] = a.mean(), a[:, 0].mean(), s.mean()
    np.save(cache, out)
    return run_dir, out


def _summary(ref, other, n):
    d = ref - other
    res = {}
    for j, key in enumerate(['mAP', 'AP50', 'APs']):
        lo, hi = np.percentile(d[1:, j], [2.5, 97.5])
        k = int(np.sum(d[1:, j] <= 0))
        # p_boot: one-sided bootstrap p-value with the usual add-one correction, (k + 1) / (B + 1)
        res[key] = {'delta': float(d[0, j]), 'ci_lo': float(lo), 'ci_hi': float(hi),
                    'p_le_0': float(np.mean(d[1:, j] <= 0)), 'k_le_0': k, 'p_boot': (k + 1) / (n + 1), 'n': n}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='standard')
    ap.add_argument('--refs', required=True, help='semicolon-separated run dir names used as references')
    ap.add_argument('--others', default='', help='semicolon-separated run dir names (default: all)')
    ap.add_argument('--pairs', default='', help='optional explicit "ref|other;ref|other" list')
    ap.add_argument('--subset', default='ALL')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--thr', type=float, default=0.0)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cluster', action='store_true', help='resample sequences (video data) instead of images')
    args = ap.parse_args()
    runs = {os.path.basename(r): r for r in list_runs(args.data, args.tag, args.protocol)}
    refs = [r for r in args.refs.split(';') if r]
    others = [o for o in args.others.split(';') if o] or list(runs)
    pairs = [tuple(p.split('|')) for p in args.pairs.split(';') if p] or \
        [(r, o) for r in refs for o in others if o != r]
    root = os.path.dirname(next(iter(runs.values())))
    nsuf = '' if args.n == 1000 else f'_n{args.n}'   # the 1,000-replicate results keep their original file names
    out_path = lambda ref: os.path.join(root, f'bootstrap_{args.subset}_thr{args.thr:g}{nsuf}_ref-{ref}.json')
    load = lambda ref: json.load(open(out_path(ref))) if os.path.exists(out_path(ref)) else {}
    needed = sorted({x for p in pairs for x in p})
    print(f'{len(pairs)} pairs, {len(needed)} runs', flush=True)
    reps = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_reps, (runs[x], args.data, args.subset, args.thr, args.n, args.seed, args.cluster))
                for x in needed]
        for fut in as_completed(futs):
            run_dir, r = fut.result()
            reps[os.path.basename(run_dir)] = r
            print(f'  replicates done: {os.path.basename(run_dir)}', flush=True)
    f = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
    by_ref = {}
    for ref, other in pairs:
        r = _summary(reps[ref], reps[other], args.n)
        by_ref.setdefault(ref, {})[other] = r
        print(f"{ref:<34} vs {other:<34} dmAP {f(r['mAP'])} P<=0 {r['mAP']['p_le_0']:.3f} | dAP50 {f(r['AP50'])} | "
              f"dAPs {f(r['APs'])} P<=0 {r['APs']['p_le_0']:.3f}", flush=True)
    for ref, out in by_ref.items():
        old = load(ref)
        old.update(out)
        with open(out_path(ref), 'w') as fh:
            json.dump(old, fh, indent=1)


if __name__ == '__main__':
    main()
