"""Markdown tables generated directly from cached metric files (avoids manual transcription)."""
import json
import os
import pickle

from .data import load_dataset
from .evaluate import SUBSETS, evaluate_run, subset_indices
from .metrics import coco_eval
from .run_eval import RUN_ROOT

COLS = {
    'mAP': ('mAP', '{:.4f}'), 'mAP@100': ('mAP (thesis, @100 dets)', '{:.4f}'), 'AP50': ('AP50', '{:.4f}'),
    'AP75': ('AP75', '{:.4f}'), 'APs': ('APs', '{:.4f}'),
    'APm': ('APm', '{:.4f}'), 'APl': ('APl', '{:.4f}'), 'AR500': ('AR', '{:.4f}'),
    'n_inf': ('#Inf', '{:.2f}'), 'gflops': ('GFLOPs', '{:.0f}'), 'infer_ms': ('Inf ms', '{:.1f}'),
    'total_ms': ('Total ms', '{:.1f}'), 'total_p95': ('p95 ms', '{:.1f}'), 'plan_ms': ('Plan ms', '{:.1f}'),
    'build_ms': ('Build ms', '{:.1f}'), 'merge_ms': ('Merge ms', '{:.1f}'), 'global_ms': ('Global ms', '{:.1f}'),
    'local_ms': ('Local ms', '{:.1f}'), 'n_img': ('N', '{:d}'),
}
_recs = {}


def metrics(data, tag, protocol, run, thr=0.0, use_ignore=True, need100=False):
    if data not in _recs:
        _recs[data] = load_dataset(data)
    recs, names = _recs[data]
    run_dir = os.path.join(RUN_ROOT, data, tag, protocol, run)
    out = evaluate_run(run_dir, recs, thr, use_ignore, n_cls=len(names))
    # Subset rows are re-accumulated at maxDets[-1]; pycocotools' stats[0] (maxDets=100, the thesis number) needs a
    # direct COCOeval on the subset.
    missing = [s for s in SUBSETS if need100 and s in out and 'mAP@100' not in out[s]]
    if missing:
        with open(os.path.join(run_dir, 'dets.pkl'), 'rb') as f:
            dets = pickle.load(f)
        idx = sorted(dets)
        for sub in missing:
            m, _ = coco_eval(recs, dets, subset_indices(recs, idx, sub), score_thr=thr, use_ignore=use_ignore,
                             n_cls=len(names))
            out[sub]['mAP@100'] = m['mAP@100']
        with open(os.path.join(run_dir, f'metrics_thr{thr:g}_ign{int(use_ignore)}.json'), 'w') as f:
            json.dump(out, f, indent=1)
    return out


def md_table(data, tag, protocol, rows, subsets=('ALL',), cols=('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf', 'gflops'),
             vram=True, thr=0.0, use_ignore=True, best=('mAP', 'AP50', 'APs', 'APm', 'APl')):
    """rows: list of (label, run_dir_name). Bold = best within each subset/column for accuracy columns."""
    vals = {}
    for label, run in rows:
        vals[label] = metrics(data, tag, protocol, run, thr, use_ignore, need100='mAP@100' in cols)
    head = ['Method'] + (['Subset'] if len(subsets) > 1 else []) + [COLS[c][0] for c in cols] + (['Peak VRAM MB'] if vram else [])
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for sub in subsets:
        bests = {c: max(vals[l][sub][c] for l, _ in rows if sub in vals[l]) for c in cols if c in best}
        for label, _ in rows:
            m = vals[label]
            if sub not in m:
                continue
            cells = [label] + ([sub] if len(subsets) > 1 else [])
            for c in cols:
                v = m[sub][c]
                s = COLS[c][1].format(int(v) if COLS[c][1] == '{:d}' else v)
                if c in bests and abs(v - bests[c]) < 1e-12:
                    s = f'**{s}**'
                cells.append(s)
            if vram:
                cells.append(f"{m['peak_alloc_mb']:.0f}")
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def load_bootstrap(data, tag, protocol, ref, subset='ALL', thr=0.0):
    p = os.path.join(RUN_ROOT, data, tag, protocol, f'bootstrap_{subset}_thr{thr:g}_ref-{ref}.json')
    with open(p) as f:
        return json.load(f)
