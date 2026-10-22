"""Figures for the report: accuracy-vs-cost Pareto plots and sensitivity curves."""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .run_eval import RUN_ROOT


def load_metrics(data, tag, protocol, name, thr=0.0):
    p = os.path.join(RUN_ROOT, data, tag, protocol, name, f'metrics_thr{thr:g}_ign1.json')
    if not os.path.exists(p):
        p = os.path.join(RUN_ROOT, data, tag, protocol, name, f'metrics_thr{thr:g}_ign0.json')
    with open(p) as f:
        return json.load(f)


def pareto(data, tag, protocol, items, out, subset='ALL', title='', cost_protocol='standard_timing', offsets=None):
    """items: list of (label, run_dir_name, marker, color). Accuracy from `protocol`, latency/VRAM from the
    idle-GPU timing runs in `cost_protocol`; methods without a dedicated timing run are left out of the latency
    and VRAM panels (they are 'n/a' in the tables too). offsets: label -> (dx, dy, ha) for the annotation."""
    from .make_tables import _exists, cost_only
    offsets = offsets or {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for label, name, mk, col in items:
        if not _exists(data, tag, protocol, name):
            continue
        m = load_metrics(data, tag, protocol, name)
        s = m[subset]
        xs = [s['gflops']]
        if _exists(data, tag, cost_protocol, name):
            c = cost_only(data, tag, cost_protocol, name)
            xs += [c[subset]['total_ms'], c['peak_alloc_mb']]
        dx, dy, ha = offsets.get(label, (6, 4, 'left'))
        for ax, x in zip(axes, xs):
            ax.scatter(x, s['mAP'], marker=mk, s=110, color=col, edgecolor='k', zorder=3, label=label)
            ax.annotate(label, (x, s['mAP']), textcoords='offset points', xytext=(dx, dy), ha=ha, fontsize=8)
    for ax, xl in zip(axes, ['GFLOPs / image', 'End-to-end latency (ms / image, RTX 5050)', 'Peak VRAM (MB)']):
        ax.set_xlabel(xl)
        ax.set_ylabel(f'mAP@[.5:.95] ({subset})')
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def sensitivity(data, tag, protocol, groups, out, default_name='HAHI'):
    """groups: dict param -> list of (value, run_dir_name); default value marked separately."""
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    base = load_metrics(data, tag, protocol, default_name)['ALL']
    for ax, (param, pts) in zip(axes, groups.items()):
        xs, ys, ss = [], [], []
        for v, name in pts:
            s = load_metrics(data, tag, protocol, name)['ALL']
            xs.append(v)
            ys.append(s['mAP'])
            ss.append(s['APs'])
        order = sorted(range(len(xs)), key=lambda k: xs[k])
        ax.plot([xs[k] for k in order], [ys[k] for k in order], 'o-', label='mAP')
        ax.plot([xs[k] for k in order], [ss[k] for k in order], 's--', label='APs')
        ax.axhline(base['mAP'], color='gray', lw=0.8, ls=':')
        ax.set_title(param)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True, help='json file describing the figure')
    a = ap.parse_args()
    with open(a.spec) as f:
        spec = json.load(f)
    kind = spec.pop('kind')
    {'pareto': pareto, 'sensitivity': sensitivity}[kind](**spec)
