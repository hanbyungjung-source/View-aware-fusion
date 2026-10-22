"""Anti-'benchmark tuning' checks: equal-or-larger tuning budget for the multi-scale (MS) baseline, and transfer of
frozen configurations to other detectors.

MS = global 640 pass + one or two high-resolution passes, fused by class-wise NMS (optionally with box voting).
High-resolution passes are evaluated as
  a = 0      dynamic input (GPU; letterbox of the original image)
  a = 1.7778 static 16:9 engine (image padded to 16:9, so 4:3 images are down-scaled)
  a = 1.0    static square engine
Per-pass raw detections are stored once, so any (engine set, fusion) configuration is evaluated offline exactly as
the online run_multiscale would produce it (verified in `check`).

  python -m hahi_bench.fairness collect --data vd_val --tag v8s
  python -m hahi_bench.fairness check --data vd_val --tag v8s
  python -m hahi_bench.fairness grid --data vd_val --tag v8s
  python -m hahi_bench.fairness materialise --data vd_testdev --tag v8s --select runs/vd_val/v8s/ms_grid.json
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .data import load_dataset
from .detector import Detector, letterbox_shape
from .metrics import coco_eval
from .run_eval import RUN_ROOT
from .thesis_methods import nms_vote

SIZES = (1152, 1280, 1408, 1536, 1664)
ASPECTS = (0.0, 1.7778, 1.0)
FUSIONS = [(0.5, 0.0), (0.6, 0.0), (0.7, 0.0), (0.6, 0.6), (0.6, 0.75), (0.7, 0.7)]
WEIGHTS = {'v8s': 'hahi_bench/train_runs/v8s_official/weights/best.pt',
           'v8n': 'hahi_bench/train_runs/v8n_official/weights/best.pt',
           'y11s': 'hahi_bench/train_runs/y11s_official/weights/best.pt'}
G640 = (640, 0.0)


def engine(S, a):
    """Static engine input (w, h) for a high-resolution pass (None for dynamic input)."""
    if a == 0.0:
        return None
    if a == 1.0:
        return S, S
    return S, int(np.ceil(S / a / 32) * 32)


def pad_to(img, a):
    h, w = img.shape[:2]
    tw, th = max(w, int(np.ceil(h * a))), max(h, int(np.ceil(w / a)))
    if (tw, th) == (w, h):
        return img
    x = np.full((th, tw, 3), 114, np.uint8)
    x[:h, :w] = img
    return x


def pass_area(key, w, h):
    """Detector input pixels of one pass / 640^2 (actual compute)."""
    S, a = key
    e = engine(S, a)
    if e is not None:
        return e[0] * e[1] / 640 ** 2
    lh, lw = letterbox_shape(h, w, S, auto=True)
    return lh * lw / 640 ** 2


def static_units(passes):
    """Cost on a fixed-input accelerator in 640x640 units (global pass = one 640x640 engine call); None if any pass
    needs dynamic input."""
    u = 0.0
    for S, a in passes:
        if (S, a) == G640:
            u += 1.0
            continue
        e = engine(S, a)
        if e is None:
            return None
        u += e[0] * e[1] / 640 ** 2
    return u


def raw_path(data, tag, key):
    return os.path.join(RUN_ROOT, data, tag, 'ms_raw', f'{key[0]}_{key[1]:g}.pkl')


def collect(args):
    recs, _ = load_dataset(args.data)
    keys = [G640] + [(S, a) for S in SIZES for a in ASPECTS]
    todo = [k for k in keys if not os.path.exists(raw_path(args.data, args.tag, k))]
    if not todo:
        return
    os.makedirs(os.path.dirname(raw_path(args.data, args.tag, G640)), exist_ok=True)
    det = Detector(WEIGHTS[args.tag], max_det=1000)
    det.warmup()
    out = {k: {} for k in todo}
    for i, r in enumerate(recs):
        img = cv2.imread(r.path)
        h, w = img.shape[:2]
        padded = {a: pad_to(img, a) for a in ASPECTS if a > 0}
        for (S, a) in todo:
            x = img if a == 0.0 else padded[a]
            b, s, c = det([x], conf=0.001, imgsz=S, max_det=1000)[0]
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
            out[(S, a)][i] = (b, s, c)
        if i % 200 == 0:
            print(f'{args.data}/{args.tag}: {i}/{len(recs)}', flush=True)
    for k, v in out.items():
        with open(raw_path(args.data, args.tag, k), 'wb') as f:
            pickle.dump(v, f)
    print('collected', args.data, args.tag, len(todo), 'passes', flush=True)


_RAW = {}


def _raw(data, tag, key):
    k = (data, tag, key)
    if k not in _RAW:
        with open(raw_path(data, tag, key), 'rb') as f:
            _RAW[k] = pickle.load(f)
    return _RAW[k]


def fuse(data, tag, passes, nms, vote, n):
    raws = [_raw(data, tag, tuple(p)) for p in passes]
    D = {}
    for i in range(n):
        B = np.concatenate([r[i][0] for r in raws])
        S = np.concatenate([r[i][1] for r in raws])
        C = np.concatenate([r[i][2] for r in raws])
        if len(B):
            D[i] = nms_vote(B, S, C, nms, vote)
        else:
            D[i] = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    return D


def _job(args):
    data, tag, passes, nms, vote = args
    recs, names = load_dataset(data)
    D = fuse(data, tag, passes, nms, vote, len(recs))
    m, _ = coco_eval(recs, D, list(range(len(recs))), n_cls=len(names))
    dyn = float(np.mean([sum(pass_area(tuple(p), r.w, r.h) for p in passes) for r in recs]))
    return {'passes': [list(p) for p in passes], 'nms': nms, 'vote': vote, 'static_units': static_units(passes),
            'dyn_units': dyn, **m}


def configs():
    out = []
    for S in SIZES:
        for a in ASPECTS:
            for nms, vote in FUSIONS:
                out.append(([G640, (S, a)], nms, vote))
    for a in ASPECTS:
        for nms, vote in FUSIONS:
            out.append(([G640, (1280, a), (1536, a)], nms, vote))
    return out


def grid(args):
    path = os.path.join(RUN_ROOT, args.data, args.tag, 'ms_grid.json')
    done = json.load(open(path)) if os.path.exists(path) else []
    seen = {(json.dumps(r['passes']), r['nms'], r['vote']) for r in done}
    cand = configs()
    if args.max_static or args.max_dyn:
        recs, _ = load_dataset(args.data)
        dyn = lambda ps: float(np.mean([sum(pass_area(tuple(p), r.w, r.h) for p in ps) for r in recs]))
        cand = [(ps, n, v) for ps, n, v in cand
                if (static_units(ps) is not None and static_units(ps) <= args.max_static + 1e-9)
                or (static_units(ps) is None and all(p[1] == 0.0 for p in ps[1:]) and dyn(ps) <= args.max_dyn + 1e-9)]
    todo = [(args.data, args.tag, [list(p) for p in ps], nms, vote) for ps, nms, vote in cand
            if (json.dumps([list(p) for p in ps]), nms, vote) not in seen]
    print(f'{len(todo)} configurations to evaluate', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(_job, todo):
            done.append(r)
            with open(path, 'w') as f:
                json.dump(done, f, indent=1)
            print(f"{r['passes']} nms {r['nms']} vote {r['vote']}  mAP {r['mAP']:.4f}  static {r['static_units']}  "
                  f"dyn {r['dyn_units']:.2f}", flush=True)


def check(args):
    """Offline fusion must reproduce the online MS runs exactly."""
    recs, names = load_dataset(args.data)
    pairs = [('MS@640x1280_aspect=1.7778_', [G640, (1280, 1.7778)], 0.6, 0.0),
             ('MS@640x1280_aspect=1.0_', [G640, (1280, 1.0)], 0.6, 0.0),
             ('MS@640x1280', [G640, (1280, 0.0)], 0.6, 0.0),
             ('MS@640x1536_aspect=1.7778,vote=0.6_', [G640, (1536, 1.7778)], 0.6, 0.6)]
    for run, passes, nms, vote in pairs:
        d = os.path.join(RUN_ROOT, args.data, args.tag, 'standard', run, 'metrics_thr0_ign1.json')
        if not os.path.exists(d):
            continue
        ref = json.load(open(d))['ALL']['mAP']
        D = fuse(args.data, args.tag, passes, nms, vote, len(recs))
        m, _ = coco_eval(recs, D, list(range(len(recs))), n_cls=len(names))
        print(f'{run:<40} online {ref:.6f}  offline {m["mAP"]:.6f}  diff {m["mAP"] - ref:+.6f}', flush=True)


def run_name(passes, nms, vote):
    def p(k):
        return f'{k[0]}{"d" if k[1] == 0 else ("s" if k[1] == 1.0 else "w")}'
    return 'MSG_' + '+'.join(p(k) for k in passes[1:]) + f'_n{nms:g}_v{vote:g}'


def materialise_one(data, tag, passes, nms, vote):
    """Write an offline-fused MS configuration as a regular run directory (for evaluate/stats)."""
    passes = [tuple(p) for p in passes]
    name = run_name(passes, nms, vote)
    out = os.path.join(RUN_ROOT, data, tag, 'standard', name)
    if os.path.exists(os.path.join(out, 'meta.json')):
        return name
    os.makedirs(out, exist_ok=True)
    recs, _ = load_dataset(data)
    D = fuse(data, tag, passes, nms, vote, len(recs))
    g640 = json.load(open(os.path.join(RUN_ROOT, data, tag, 'standard', 'global@640', 'meta.json')))['gflops640']
    per = [{'i': i, 'total': 0.0, 'global': 0.0, 'plan': 0.0, 'build': 0.0, 'local': 0.0, 'merge': 0.0, 'infer': 0.0,
            'n_inf': len(passes), 'gflops': g640 * sum(pass_area(p, r.w, r.h) for p in passes),
            'n_det': int(len(D[i][0]))} for i, r in enumerate(recs)]
    meta = {'spec': name, 'protocol': 'standard', 'passes': [list(p) for p in passes], 'nms': nms, 'vote': vote,
            'static_units': static_units(passes), 'offline_fusion': True, 'gflops640': g640,
            'peak_alloc_mb': 0.0, 'peak_reserved_mb': 0.0, 'weights': WEIGHTS[tag]}
    with open(os.path.join(out, 'dets.pkl'), 'wb') as f:
        pickle.dump(D, f)
    with open(os.path.join(out, 'per_image.json'), 'w') as f:
        json.dump(per, f)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return name


def select(grid_rows, budget, setting):
    """Best-on-val MS configuration within a cost budget. setting: 'static' (fixed-input engines, 640x640 units) or
    'dynamic' (GPU, letterboxed pixels / 640^2)."""
    key = 'static_units' if setting == 'static' else 'dyn_units'
    ok = [r for r in grid_rows if r[key] is not None and r[key] <= budget + 1e-9
          and (setting == 'dynamic') == all(p[1] == 0.0 for p in r['passes'][1:])]
    return max(ok, key=lambda r: r['mAP']) if ok else None


def materialise(args):
    rows = json.load(open(args.select))
    for setting, budgets in [('static', [float(b) for b in args.static.split(',') if b]),
                             ('dynamic', [float(b) for b in args.dynamic.split(',') if b])]:
        for b in budgets:
            r = select(rows, b, setting)
            if r is None:
                continue
            name = materialise_one(args.data, args.tag, r['passes'], r['nms'], r['vote'])
            print(f'{setting} budget {b}: {name} (val mAP {r["mAP"]:.4f}, static {r["static_units"]}, '
                  f'dyn {r["dyn_units"]:.2f})', flush=True)
    for spec in [s for s in args.extra.split(';') if s]:
        ps, nms, vote = json.loads(spec)
        print('extra:', materialise_one(args.data, args.tag, ps, nms, vote), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['collect', 'check', 'grid', 'materialise'])
    ap.add_argument('--data', default='vd_val')
    ap.add_argument('--tag', default='v8s')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--select', default='')
    ap.add_argument('--static', default='')
    ap.add_argument('--dynamic', default='')
    ap.add_argument('--extra', default='')
    ap.add_argument('--max_static', type=float, default=0.0)
    ap.add_argument('--max_dyn', type=float, default=0.0)
    a = ap.parse_args()
    {'collect': collect, 'check': check, 'grid': grid, 'materialise': materialise}[a.cmd](a)
