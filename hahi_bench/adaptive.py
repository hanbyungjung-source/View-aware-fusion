"""Resolution-adaptive global pass (REPORT section 22, exploratory): per image, take the run whose global pass is 640
when the long side is <= LIMIT px and the run whose global pass is 1280 otherwise. The composite is written as an
ordinary run directory (dets.pkl, per_image.json, meta.json), so evaluate.py and stats.py apply unchanged.

  python -m hahi_bench.adaptive --data dota_val --tag v8s_dota --lo "HAHI_..._" --hi "HAHI_...,global_imgsz=1280_" \
      --name "HAHI-S+F_adaptive-global_"
"""
import argparse
import json
import os
import pickle

from .data import load_dataset
from .run_eval import RUN_ROOT

LIMIT = 1920


def compose(data, tag, protocol, lo, hi, name, limit=LIMIT):
    """lo / hi are run directory names inside `protocol`, or 'other_protocol/run_name' to read from elsewhere."""
    recs, _ = load_dataset(data)
    root = os.path.join(RUN_ROOT, data, tag, protocol)
    base = os.path.join(RUN_ROOT, data, tag)
    path = lambda d: os.path.join(base, d) if '/' in d else os.path.join(root, d)
    load = lambda d, f: (pickle.load(open(os.path.join(path(d), f), 'rb')) if f.endswith('.pkl')
                         else json.load(open(os.path.join(path(d), f))))
    D_lo, D_hi = load(lo, 'dets.pkl'), load(hi, 'dets.pkl')
    P_lo = {p['i']: p for p in load(lo, 'per_image.json')}
    P_hi = {p['i']: p for p in load(hi, 'per_image.json')}
    m_lo, m_hi = load(lo, 'meta.json'), load(hi, 'meta.json')
    big = {i for i in D_lo if max(recs[i].w, recs[i].h) > limit}
    dets = {i: (D_hi[i] if i in big else D_lo[i]) for i in D_lo}
    per = [P_hi[i] if i in big else P_lo[i] for i in sorted(D_lo)]
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'dets.pkl'), 'wb') as f:
        pickle.dump(dets, f)
    with open(os.path.join(out, 'per_image.json'), 'w') as f:
        json.dump(per, f)
    meta = dict(m_lo)
    meta.update({'spec': f'adaptive(long side > {limit}: {m_hi["spec"]}; else: {m_lo["spec"]})',
                 'peak_alloc_mb': max(m_lo['peak_alloc_mb'], m_hi['peak_alloc_mb']),
                 'peak_reserved_mb': max(m_lo['peak_reserved_mb'], m_hi['peak_reserved_mb']),
                 'n_big': len(big), 'limit': limit})
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    for fn in os.listdir(out):
        if fn.startswith(('metrics_', 'boot_')):
            os.remove(os.path.join(out, fn))
    print(f'{name}: {len(big)}/{len(D_lo)} images use the 1280 global pass')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='standard')
    ap.add_argument('--lo', required=True)
    ap.add_argument('--hi', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--limit', type=int, default=LIMIT)
    a = ap.parse_args()
    compose(a.data, a.tag, a.protocol, a.lo, a.hi, a.name, a.limit)
