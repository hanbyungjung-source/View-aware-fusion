"""Simulation of per-image adaptive inference (cascades) from stored detections.

Every method below starts with the same 640 px global pass, so a cascade "run the 640 pass on every image and the
expensive method only on images whose 640-pass statistics exceed a threshold" produces exactly the stored
detections of the cheap method on non-gated images and of the expensive method on gated images. The mixture is
evaluated exactly by swapping per-image COCOeval results (evalImgs) and re-running pycocotools' accumulate().

Example:
  python -m hahi_bench.gating --data vd_val --tag v8s
  python -m hahi_bench.gating --data vd_testdev --tag v8s --apply runs/vd_val/v8s/standard/gating_choice.json
"""
import argparse
import contextlib
import copy
import io
import json
import os
import pickle

import numpy as np

from .data import load_dataset
from .metrics import coco_eval
from .run_eval import RUN_ROOT

CHEAP = 'global@640'
# expensive option -> extra cost model: 'self' = stored per-image cost already includes the 640 pass,
# 'plus640' = the option does not reuse the 640 pass, so its cost is added on top of the cheap pass.
OPTIONS = {
    'MS@640x1280': 'self',
    'MS@640x1536': 'self',
    'MS@640x1280x1536': 'self',
    'HAHI': 'self',
    'HAHI_gcc=1.0,route_ratio=0.1_': 'self',
    'HAHI_global_imgsz=1280,gcc=1.0_': 'plus640',
}
REF_POINTS = ['global@640', 'global@1280', 'MS@640x1280', 'MS@640x1536', 'MS@640x1280x1536', 'HAHI',
              'HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI_global_imgsz=1280,gcc=1.0_']


def _load(data, tag, name):
    d = os.path.join(RUN_ROOT, data, tag, 'standard', name)
    if not os.path.exists(os.path.join(d, 'dets.pkl')):
        return None
    with open(os.path.join(d, 'dets.pkl'), 'rb') as f:
        dets = pickle.load(f)
    per = {p['i']: p for p in json.load(open(os.path.join(d, 'per_image.json')))}
    return dets, per


def features(recs, dets640):
    """Per-image routing features computed from the 640 pass only (no ground truth)."""
    F = {k: [] for k in ['n_small', 'n_small_unc', 'n_det', 'small_mass', 'pixels']}
    for i in range(len(recs)):
        b, s, _ = dets640[i]
        a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)
        small = a < 32 ** 2
        F['n_small'].append(np.sum(small & (s >= 0.1)))
        F['n_small_unc'].append(np.sum(small & (s >= 0.1) & (s < 0.5)))
        F['n_det'].append(np.sum(s >= 0.1))
        F['small_mass'].append(float(np.sum(s[small & (s >= 0.05)])))
        F['pixels'].append(recs[i].w * recs[i].h)
    F = {k: np.asarray(v, np.float64) for k, v in F.items()}
    # oracle feature (upper bound for any predictor of "how many small objects"): GT small-object count
    F['ORACLE_gt_small'] = np.asarray([np.sum((r.boxes[:, 2] - r.boxes[:, 0]) * (r.boxes[:, 3] - r.boxes[:, 1])
                                              < 32 ** 2) if len(r.boxes) else 0 for r in recs], np.float64)
    return F


class Mixer:
    """Holds evaluated COCOeval objects of several runs and evaluates per-image mixtures exactly."""

    def __init__(self, recs, n_cls, runs):
        self.E = {}
        idx = list(range(len(recs)))
        for name, (dets, _) in runs.items():
            _, E = coco_eval(recs, dets, idx, keep_eval=True, n_cls=n_cls)
            self.E[name] = E
        E0 = next(iter(self.E.values()))
        self.n_img = len(E0.params.imgIds)
        self.n_area = len(E0.params.areaRng)
        self.n_cat = len(E0.params.catIds)

    def evaluate(self, choice, names):
        """choice: int array (n_img,) indexing into names."""
        base = self.E[names[0]]
        E = copy.copy(base)
        ev = list(base.evalImgs)
        for j, name in enumerate(names[1:], start=1):
            src = self.E[name].evalImgs
            for i in np.flatnonzero(choice == j):
                for k in range(self.n_cat):
                    for a in range(self.n_area):
                        p = k * self.n_area * self.n_img + a * self.n_img + i
                        ev[p] = src[p]
        E.evalImgs = ev
        with contextlib.redirect_stdout(io.StringIO()):
            E.accumulate()
        P = E.eval['precision']
        out = {}
        for key, a in [('mAP', 0), ('APs', 1)]:
            q = P[:, :, :, a, -1]
            out[key] = float(q[q > -1].mean())
        q = P[0, :, :, 0, -1]
        out['AP50'] = float(q[q > -1].mean())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--fracs', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    ap.add_argument('--apply', default='', help='json with thresholds chosen on val (feature values)')
    ap.add_argument('--options', default='', help='semicolon-separated subset of OPTIONS to simulate')
    args = ap.parse_args()
    recs, names = load_dataset(args.data)
    runs = {n: r for n in [CHEAP, 'global@1280'] + list(OPTIONS) if (r := _load(args.data, args.tag, n))}
    mixer = Mixer(recs, len(names), runs)
    F = features(recs, runs[CHEAP][0])
    n = len(recs)
    cost = {name: np.asarray([per[i]['gflops'] for i in range(n)]) for name, (_, per) in runs.items()}
    out_dir = os.path.join(RUN_ROOT, args.data, args.tag, 'standard')
    res = {'points': {}, 'curves': {}}
    for name in REF_POINTS:
        if name in runs:
            m = mixer.evaluate(np.zeros(n, int), [name])
            res['points'][name] = {**m, 'gflops': float(cost[name].mean())}
            print(f'{name:<36} mAP {m["mAP"]:.4f} APs {m["APs"]:.4f} GFLOPs {cost[name].mean():6.1f}', flush=True)
    if args.apply:
        rules = json.load(open(args.apply))
        res['applied'] = []
        for r in rules:
            opt, feat, thr = r['option'], r['feature'], r['threshold']
            g = F[feat] >= thr
            extra = cost[opt] + (cost[CHEAP] if OPTIONS[opt] == 'plus640' else 0)
            m = mixer.evaluate(g.astype(int), [CHEAP, opt])
            gf = float(np.where(g, extra, cost[CHEAP]).mean())
            res['applied'].append({**r, **m, 'gflops': gf, 'frac': float(g.mean())})
            print(f'APPLY {r.get("label", ""):<28} {opt:<34} {feat:<12} thr {thr:8.1f} frac {g.mean():.2f} '
                  f'mAP {m["mAP"]:.4f} APs {m["APs"]:.4f} GFLOPs {gf:6.1f}', flush=True)
        with open(os.path.join(out_dir, 'gating_applied.json'), 'w') as f:
            json.dump(res, f, indent=1)
        return
    fracs = [float(x) for x in args.fracs.split(',')]
    path = os.path.join(out_dir, 'gating_curves.json')
    if os.path.exists(path):
        old = json.load(open(path))
        res['curves'] = old.get('curves', {})
    for opt, mode in OPTIONS.items():
        if opt not in runs or (args.options and opt not in args.options.split(';')):
            continue
        extra = cost[opt] + (cost[CHEAP] if mode == 'plus640' else 0)
        for feat, v in F.items():
            if f'{opt}|{feat}' in res['curves']:
                continue
            order = np.argsort(-v, kind='mergesort')
            curve = []
            for fr in fracs:
                k = int(round(fr * n))
                g = np.zeros(n, bool)
                g[order[:k]] = True
                thr = float(v[order[k - 1]]) if k > 0 else float('inf')
                m = mixer.evaluate(g.astype(int), [CHEAP, opt])
                gf = float(np.where(g, extra, cost[CHEAP]).mean())
                curve.append({'frac': fr, 'threshold': thr, **m, 'gflops': gf})
            res['curves'][f'{opt}|{feat}'] = curve
            print(f'{opt:<34} {feat:<16} ' + ' '.join(f'{c["mAP"]:.4f}@{c["gflops"]:.0f}' for c in curve), flush=True)
            with open(path, 'w') as f:
                json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
