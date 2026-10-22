"""Run detection methods over a dataset and store raw detections + per-image cost statistics.

Example:
  python -m hahi_bench.run_eval --data vd_val --weights W.pt --tag v8s --protocol standard \
      --methods "global@640;HAHI;HAHI{sce=False};SAHI@512"
"""
import argparse
import json
import os
import pickle
import platform
import re
import time
from dataclasses import replace, fields

import cv2
import numpy as np
import torch

from . import baselines as BL
from . import thesis_methods as TM
from .data import load_dataset
from .detector import Detector, make_detector

RUN_ROOT = os.path.join(os.path.dirname(__file__), 'runs')
TIMER_KEYS = ['global', 'plan', 'build', 'local', 'merge']


def _parse_overrides(text, cls):
    if not text:
        return {}
    types = {f.name: f.type for f in fields(cls)}
    out = {}
    for kv in text.split(','):
        k, v = kv.split('=')
        k, v = k.strip(), v.strip()
        t = types[k]
        if t in (bool, 'bool'):
            out[k] = v.lower() in ('1', 'true', 'yes')
        elif t in (int, 'int'):
            out[k] = int(v)
        elif t in (float, 'float'):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def build_method(spec, protocol):
    """Return (callable(det, img, timer) -> dets, config dict)."""
    m = re.fullmatch(r'([\w+\-]+)(?:@([\w.]+))?(?:\{(.*)\})?', spec)
    if not m:
        raise ValueError(spec)
    name, arg, ov = m.group(1), m.group(2), m.group(3)
    std = protocol == 'standard'
    conf = 0.001 if std else 0.3
    if name in TM.THESIS_PRESETS:
        kind, cfg = TM.THESIS_PRESETS[name]
        if std:
            cfg = TM.standard(cfg)
        cfg = replace(cfg, **_parse_overrides(ov, TM.HahiCfg))
        fn = TM.run_uc if kind == 'uc' else TM.run_hahi
        if kind == 'uc':
            return (lambda det, img, timer, rec=None: fn(det, img, cfg, timer)), cfg.as_dict()
        return (lambda det, img, timer, rec=None: fn(det, img, cfg, timer, rec)), cfg.as_dict()
    if name == 'global':
        imgsz = int(arg or 640)
        return (lambda det, img, timer, rec=None: BL.run_global(det, img, timer, imgsz=imgsz, conf=conf)), \
            {'imgsz': imgsz, 'conf': conf}
    if name == 'MS':
        sizes = tuple(int(v) for v in (arg or '640x1280').split('x'))
        kw = _parse_kv(ov)
        return (lambda det, img, timer, rec=None: BL.run_multiscale(det, img, timer, sizes=sizes, conf=conf, **kw)), \
            {'sizes': sizes, 'conf': conf, **kw}
    if name == 'MSTILE':
        kw = {'size': int(arg or 1280)}
        kw.update(_parse_kv(ov))
        return (lambda det, img, timer, rec=None: BL.run_mstile(det, img, timer, conf=conf, **kw)), {'conf': conf, **kw}
    if name == 'SAHI':
        kw = {'slice_size': int(arg or 512)}
        kw.update(_parse_kv(ov))
        kw.setdefault('conf', conf)
        return (lambda det, img, timer, rec=None: BL.run_sahi(det, img, timer, **kw)), kw
    if name in ('DAHI', 'PAN'):
        from . import literature as LT
        fn, cfg = LT.build(name, arg, _parse_kv(ov), conf)
        return fn, cfg
    if name == 'DYNTILE':
        from .dyntile import run_dyntile
        kw = _parse_kv(ov)
        kw.setdefault('conf', conf)
        return (lambda det, img, timer, rec=None: run_dyntile(det, img, timer, **kw)), kw
    raise ValueError(f'unknown method {spec}')


def _parse_kv(text):
    out = {}
    if not text:
        return out
    for kv in text.split(','):
        k, v = kv.split('=')
        v = v.strip()
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ('true', 'false'):
                    v = v.lower() == 'true'
        out[k.strip()] = v
    return out


def safe(spec):
    return re.sub(r'[^\w@.+\-=,]', '_', spec).replace('{', '(').replace('}', ')')


def run_one(det, recs, spec, protocol, out_dir, limit=None, warmup=8, force=False):
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'meta.json')) and not force:
        print(f'skip {spec} (exists)')
        return
    fn, cfg = build_method(spec, protocol)
    idx = list(range(len(recs)))[:limit] if limit else list(range(len(recs)))
    for i in idx[:warmup]:
        fn(det, cv2.imread(recs[i].path), {k: 0.0 for k in TIMER_KEYS}, rec=recs[i])
    det.reset()
    cuda = det.use_cuda
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    dets, per = {}, []
    t_all = time.perf_counter()
    for i in idx:
        img = cv2.imread(recs[i].path)
        timer = {k: 0.0 for k in TIMER_KEYS}
        n0, g0, s0 = det.n_images, det.gflops, det.seconds
        t0 = time.perf_counter()
        b, s, c = fn(det, img, timer, rec=recs[i])
        if cuda:
            torch.cuda.synchronize()
        tt = time.perf_counter() - t0
        dets[i] = (b, s, c)
        per.append({'i': i, 'total': tt, **timer, 'infer': det.seconds - s0, 'n_inf': det.n_images - n0,
                    'gflops': det.gflops - g0, 'n_det': int(len(b))})
    wall = time.perf_counter() - t_all
    meta = {'spec': spec, 'protocol': protocol, 'cfg': cfg, 'n_images': len(idx), 'wall_s': wall,
            'weights': det.weights, 'gflops640': det.gflops640, 'force_batch': det.force_batch,
            'peak_alloc_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0,
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / 2 ** 20 if cuda else 0,
            'gpu': torch.cuda.get_device_name(0) if cuda else 'cpu',
            'torch': torch.__version__, 'platform': platform.platform()}
    with open(os.path.join(out_dir, 'dets.pkl'), 'wb') as f:
        pickle.dump(dets, f)
    with open(os.path.join(out_dir, 'per_image.json'), 'w') as f:
        json.dump(per, f)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    tot = np.mean([p['total'] for p in per]) * 1000
    inf = np.mean([p['infer'] for p in per]) * 1000
    print(f'{spec:<45} {len(idx)} imgs  total {tot:6.1f} ms  infer {inf:6.1f} ms  '
          f'n_inf {np.mean([p["n_inf"] for p in per]):.2f}  peak {meta["peak_alloc_mb"]:.1f} MB', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='standard', choices=['standard', 'thesis'])
    ap.add_argument('--methods', required=True, help='semicolon separated specs')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--max-det', type=int, default=None)
    ap.add_argument('--batch1', action='store_true', help='force batch size 1 for every inference (memory study)')
    args = ap.parse_args()
    recs, _ = load_dataset(args.data)
    det = make_detector(args.weights, max_det=args.max_det or (1000 if args.protocol == 'standard' else 300),
                        force_batch=1 if args.batch1 else 0)
    det.warmup()
    for spec in [s.strip() for s in args.methods.split(';') if s.strip()]:
        out = os.path.join(RUN_ROOT, args.data, args.tag, args.protocol + args.suffix, safe(spec))
        run_one(det, recs, spec, args.protocol, out, args.limit, force=args.force)


if __name__ == '__main__':
    main()
