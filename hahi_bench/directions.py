"""Tables and figures for the 'further directions' section of REPORT.md (§14):
oracle crop selection, merge-NMS threshold, per-image adaptive gating, and the fixed-input-shape regime.

  python -m hahi_bench.directions
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .make_tables import _exists, boot_table, cost_only, fill_report
from .run_eval import RUN_ROOT
from .tables import md_table, metrics

FIG = os.path.join(os.path.dirname(__file__), 'figures')
COLS = ('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf', 'gflops')

ORACLE = [
    ('HAHI (thesis default)', 'HAHI'),
    ('HAHI + GT-oracle ROIs', 'HAHI_roi_source=gt_'),
    ('HAHI (val-tuned)', 'HAHI_gcc=1.0,route_ratio=0.1_'),
    ('HAHI (val-tuned) + GT-oracle ROIs', 'HAHI_roi_source=gt,gcc=1.0,route_ratio=0.1_'),
    ('HAHI-ext', 'HAHI_global_imgsz=1280,gcc=1.0_'),
    ('HAHI-ext + GT-oracle ROIs', 'HAHI_roi_source=gt,global_imgsz=1280,gcc=1.0_'),
    ('MS 640+1280 (reference)', 'MS@640x1280'),
]
NMS = [
    ('HAHI, NMS 0.40/0.45 (thesis code)', 'HAHI'),
    ('HAHI, NMS 0.5/0.5', 'HAHI_nms_local=0.5,nms_final=0.5_'),
    ('HAHI, NMS 0.6/0.6', 'HAHI_nms_local=0.6,nms_final=0.6_'),
    ('HAHI, NMS 0.7/0.7', 'HAHI_nms_local=0.7,nms_final=0.7_'),
    ('HAHI (val-tuned)', 'HAHI_gcc=1.0,route_ratio=0.1_'),
    ('HAHI (val-tuned), NMS 0.6/0.6', 'HAHI_gcc=1.0,route_ratio=0.1,nms_local=0.6,nms_final=0.6_'),
    ('HAHI-ext', 'HAHI_global_imgsz=1280,gcc=1.0_'),
    ('HAHI-ext, NMS 0.6/0.6', 'HAHI_global_imgsz=1280,gcc=1.0,nms_local=0.6,nms_final=0.6_'),
    ('UC 2x2, NMS 0.4 (thesis code)', 'UC2x2'),
    ('UC 2x2, NMS 0.6', 'UC2x2_nms_local=0.6_'),
    ('MS 640+1280, NMS 0.5', 'MS@640x1280_iou=0.5_'),
    ('MS 640+1280, NMS 0.6 (used)', 'MS@640x1280'),
    ('MS 640+1280, NMS 0.7', 'MS@640x1280_iou=0.7_'),
]
FIXED = [
    ('Global 640 (1 pass)', 'global@640'),
    ('MS-tiled 960 (640 + 640x640 tiles)', 'MSTILE@960'),
    ('MS-tiled 1280', 'MSTILE@1280'),
    ('MS-tiled 1280, overlap 0.15', 'MSTILE@1280_overlap=0.15_'),
    ('MS-tiled 1536', 'MSTILE@1536'),
    ('UC 2x2 (thesis)', 'UC2x2'),
    ('SAHI 512 (NMS)', 'SAHI@512_post=NMS,metric=IOU_'),
    ('DAHI (N_ROI=2)', 'DAHI_n_roi=2_'),
    ('PAN fine+packing', 'PAN_bandwidth=256_'),
    ('**HAHI (thesis default)**', 'HAHI'),
    ('**HAHI (val-tuned)**', 'HAHI_gcc=1.0,route_ratio=0.1_'),
]
SQ = 'standard_b1_sq'   # static square 640x640 input emulation, batch 1, idle GPU
SQ_DEP = 'thesis_b1_sq'  # same, deployment score floor 0.3 (fewer boxes to merge)
_S = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True'
_T = _S + ',nms_impl=tv'
SIMPLE = [
    # label, accuracy run (outputs identical with/without fast_merge), timing run name
    ('HAHI (thesis code)', 'HAHI', 'HAHI'),
    ('HAHI + vectorised merge (identical output)', 'HAHI', 'HAHI_fast_merge=True_'),
    ('HAHI-S (simplified, cv2 NMS)', f'HAHI_{_S}_', f'HAHI_{_S}_'),
    ('**HAHI-S** (simplified, torchvision NMS)', f'HAHI_{_T}_', f'HAHI_{_T}_'),
    ('HAHI-S, route 0.2 (cv2 NMS)', f'HAHI_{_S},route_ratio=0.2_', f'HAHI_{_S},route_ratio=0.2_'),
    ('HAHI val-tuned (thesis code)', 'HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI_gcc=1.0,route_ratio=0.1_'),
    ('HAHI val-tuned + vectorised merge', 'HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI_gcc=1.0,route_ratio=0.1,fast_merge=True_'),
    ('HAHI-S, route 0.1 (cv2 NMS)', f'HAHI_{_S},route_ratio=0.1_', f'HAHI_{_S},route_ratio=0.1_'),
    ('**HAHI-S, route 0.1** (torchvision NMS)', f'HAHI_{_T},route_ratio=0.1_', f'HAHI_{_T},route_ratio=0.1_'),
    ('MS-tiled 960', 'MSTILE@960', 'MSTILE@960'),
    ('MS-tiled 1280', 'MSTILE@1280', 'MSTILE@1280'),
    ('MS-tiled 1280, overlap 0.15', 'MSTILE@1280_overlap=0.15_', 'MSTILE@1280_overlap=0.15_'),
    ('MS-tiled 1536', 'MSTILE@1536', 'MSTILE@1536'),
]
DEP_NAME = {k: k for k in ['HAHI', 'HAHI_fast_merge=True_', 'HAHI_gcc=1.0,route_ratio=0.1_',
                           'HAHI_gcc=1.0,route_ratio=0.1,fast_merge=True_', f'HAHI_{_S}_', f'HAHI_{_S},route_ratio=0.1_',
                           f'HAHI_{_T}_', f'HAHI_{_T},route_ratio=0.1_', 'MSTILE@960', 'MSTILE@1280', 'MSTILE@1536']}


def _timing(data, tag, protocol, run):
    if not _exists(data, tag, protocol, run):
        return None
    per = json.load(open(os.path.join(RUN_ROOT, data, tag, protocol, run, 'per_image.json')))
    a = lambda k: 1000 * sum(p[k] for p in per) / len(per)
    return {'total': a('total'), 'cpu': a('total') - a('infer'), 'plan': a('plan'), 'build': a('build'),
            'merge': a('merge')}


def simplify_table():
    head = ['Method', 'val mAP', 'test-dev mAP', 'APs', '#640x640 passes',
            'Latency ms (score ≥ 0.001)', 'non-inference ms (plan / build / merge)',
            'Latency ms (score ≥ 0.3, deployment)', 'non-inference ms (score ≥ 0.3)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, acc, tim in SIMPLE:
        if not _exists('vd_testdev', 'v8s', 'standard', acc):
            continue
        v = metrics('vd_val', 'v8s', 'standard', acc)['ALL']['mAP'] if _exists('vd_val', 'v8s', 'standard', acc) else None
        t = metrics('vd_testdev', 'v8s', 'standard', acc)['ALL']
        s, d = _timing('vd_testdev', 'v8s', SQ, tim), _timing('vd_testdev', 'v8s', SQ_DEP, DEP_NAME.get(tim, ''))
        cells = [label, f'{v:.4f}' if v is not None else 'n/a', f"{t['mAP']:.4f}", f"{t['APs']:.4f}",
                 f"{t['n_inf']:.2f}",
                 f"{s['total']:.1f}" if s else 'n/a',
                 f"{s['cpu']:.1f} ({s['plan']:.1f} / {s['build']:.1f} / {s['merge']:.1f})" if s else 'n/a',
                 f"{d['total']:.1f}" if d else 'n/a',
                 f"{d['cpu']:.1f} ({d['plan']:.1f} / {d['build']:.1f} / {d['merge']:.1f})" if d else 'n/a']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def simplify_steps_table():
    steps = [('HAHI (thesis code)', 'HAHI'),
             ('A: weight uniform, no snap', 'HAHI_weight=uniform,snap=False,fast_merge=True_'),
             ('B: A + no upscale', 'HAHI_weight=uniform,snap=False,upscale=none,fast_merge=True_'),
             ('C: B + no GCC', 'HAHI_weight=uniform,snap=False,upscale=none,gcc=1.0,fast_merge=True_'),
             ('HAHI-S: C + merge NMS 0.6', f'HAHI_{_S}_'),
             ('HAHI-S with torchvision NMS', f'HAHI_{_T}_'),
             ('HAHI-S, route 0.2', f'HAHI_{_S},route_ratio=0.2_'),
             ('HAHI-S, route 0.1', f'HAHI_{_S},route_ratio=0.1_'),
             ('HAHI-S, route 0.1, torchvision NMS', f'HAHI_{_T},route_ratio=0.1_'),
             ('HAHI val-tuned (thesis code)', 'HAHI_gcc=1.0,route_ratio=0.1_'),
             ('MS-tiled 1280', 'MSTILE@1280'), ('MS-tiled 1536', 'MSTILE@1536')]
    return md_table('vd_val', 'v8s', 'standard', [r for r in steps if _exists('vd_val', 'v8s', 'standard', r[1])],
                    cols=('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf'), vram=False, best=())


def fixed_table(data, tag, rows, latency=True):
    rows = [r for r in rows if _exists(data, tag, 'standard', r[1])]
    head = ['Method', 'mAP', 'AP50', 'APs', 'APm', 'APl', '#640x640 passes']
    if latency:
        head += ['Latency ms, static 640x640, batch 1 (mean / p95)']
    vals = {l: metrics(data, tag, 'standard', r)['ALL'] for l, r in rows}
    best = {c: max(v[c] for v in vals.values()) for c in ['mAP', 'AP50', 'APs', 'APm', 'APl']}
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, run in rows:
        m = vals[label]
        cells = [label] + [(f'**{m[c]:.4f}**' if abs(m[c] - best[c]) < 1e-12 else f'{m[c]:.4f}')
                           for c in ['mAP', 'AP50', 'APs', 'APm', 'APl']] + [f"{m['n_inf']:.2f}"]
        if latency:
            if _exists(data, tag, SQ, run):
                c = cost_only(data, tag, SQ, run)['ALL']
                cells.append(f"{c['total_ms']:.1f} / {c['total_p95']:.1f}")
            else:
                cells.append('n/a')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def gating_table(data, tag, picks):
    g = json.load(open(os.path.join(RUN_ROOT, data, tag, 'standard', 'gating_curves.json')))
    lines = ['| Configuration | Images with extra pass | mAP | APs | GFLOPs |', '|---|---|---|---|---|']
    for label, name in picks['static']:
        if name in g['points']:
            p = g['points'][name]
            lines.append(f"| {label} (static) | {'0%' if name == 'global@640' else '100%'} | {p['mAP']:.4f} | "
                         f"{p['APs']:.4f} | {p['gflops']:.0f} |")
    for label, key, fracs in picks['gated']:
        for c in g['curves'][key]:
            if any(abs(c['frac'] - f) < 1e-9 for f in fracs):
                lines.append(f"| {label} | {c['frac']:.0%} | {c['mAP']:.4f} | {c['APs']:.4f} | {c['gflops']:.0f} |")
    return '\n'.join(lines)


def fig_fixed(out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    style = {'global@640': ('o', '#7f7f7f'), 'MSTILE@960': ('s', '#1f77b4'), 'MSTILE@1280': ('s', '#1f77b4'),
             'MSTILE@1280_overlap=0.15_': ('s', '#17becf'), 'MSTILE@1536': ('s', '#0b3d91'),
             'UC2x2': ('v', '#8c564b'), 'SAHI@512_post=NMS,metric=IOU_': ('v', '#e377c2'),
             'DAHI_n_roi=2_': ('^', '#dbdb8d'), 'PAN_bandwidth=256_': ('D', '#9467bd'), 'HAHI': ('*', '#d62728'),
             'HAHI_gcc=1.0,route_ratio=0.1_': ('*', '#ff7f0e')}
    short = {'global@640': 'Global640', 'MSTILE@960': 'MS-tiled 960', 'MSTILE@1280': 'MS-tiled 1280',
             'MSTILE@1280_overlap=0.15_': 'MS-tiled 1280 ov', 'MSTILE@1536': 'MS-tiled 1536', 'UC2x2': 'UC2x2',
             'SAHI@512_post=NMS,metric=IOU_': 'SAHI512', 'DAHI_n_roi=2_': 'DAHI N_ROI=2', 'PAN_bandwidth=256_': 'PAN',
             'HAHI': 'HAHI', 'HAHI_gcc=1.0,route_ratio=0.1_': 'HAHI tuned'}
    panels = [('vd_testdev', 'v8s', 'n_inf', 'VisDrone test-dev: mAP vs #640x640 passes'),
              ('vd_testdev', 'v8s', 'lat', 'VisDrone test-dev: mAP vs latency (static 640x640, batch 1)'),
              ('dota_val', 'v8s_dota', 'n_inf', 'DOTA v1.0 val: mAP vs #640x640 passes')]
    for ax, (data, tag, xkey, title) in zip(axes, panels):
        line = []
        for run, (mk, col) in style.items():
            if not _exists(data, tag, 'standard', run):
                continue
            m = metrics(data, tag, 'standard', run)['ALL']
            if xkey == 'lat':
                if not _exists(data, tag, SQ, run):
                    continue
                x = cost_only(data, tag, SQ, run)['ALL']['total_ms']
            else:
                x = m['n_inf']
            if data == 'dota_val' and x > 12:
                ax.annotate(f"{short[run]}: {x:.1f} passes, {m['mAP']:.4f} (off-chart)", (0.98, 0.03),
                            xycoords='axes fraction', ha='right', fontsize=8)
                continue
            ax.scatter(x, m['mAP'], marker=mk, s=110 if mk != '*' else 220, color=col, edgecolor='k', zorder=3)
            ax.annotate(short[run], (x, m['mAP']), textcoords='offset points', xytext=(6, 4), fontsize=8)
            if run in ('global@640', 'MSTILE@960', 'MSTILE@1280', 'MSTILE@1536'):
                line.append((x, m['mAP']))
        if line:
            line.sort()
            ax.plot(*zip(*line), '-', color='#1f77b4', lw=1, alpha=0.6, zorder=1)
        ax.set_xlabel('ms / image' if xkey == 'lat' else 'fixed-size 640x640 passes / image')
        ax.set_ylabel('mAP@[.5:.95]')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gating(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cfg = [('vd_val', 'v8s', 'VisDrone val (YOLOv8s)',
            [('MS@640x1280|n_small', '640 -> +1280 if many small dets', '#1f77b4', '-'),
             ('MS@640x1280|ORACLE_gt_small', 'same, oracle (GT small count)', '#1f77b4', ':'),
             ('MS@640x1280x1536|n_small', '640 -> +1280+1536 if many small dets', '#0b3d91', '-'),
             ('HAHI|n_small', '640 -> HAHI crops if many small dets', '#d62728', '-')],
            ['global@640', 'global@960', 'global@1280', 'global@1536', 'MS@640x960', 'MS@640x1280', 'MS@640x1536',
             'MS@640x1280x1536', 'HAHI', 'HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI_global_imgsz=1280,gcc=1.0_']),
           ('dota_val', 'v8s_dota', 'DOTA v1.0 val (YOLOv8s)',
            [('MS@640x1280|pixels', '640 -> +1280 for largest images', '#1f77b4', '-'),
             ('MS@640x1536|pixels', '640 -> +1536 for largest images', '#17becf', '-'),
             ('MS@640x1536|ORACLE_gt_small', '640 -> +1536, oracle (GT small count)', '#17becf', ':'),
             ('HAHI_global_imgsz=1280,gcc=1.0_|pixels', '640 -> HAHI-ext for largest images', '#2ca02c', '-')],
            ['global@640', 'global@1280', 'MS@640x1280', 'MS@640x1536', 'MS@640x1280x1536', 'HAHI',
             'HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI_global_imgsz=1280,gcc=1.0_'])]
    for ax, (data, tag, title, curves, statics) in zip(axes, cfg):
        g = json.load(open(os.path.join(RUN_ROOT, data, tag, 'standard', 'gating_curves.json')))
        for key, label, col, ls in curves:
            c = g['curves'][key]
            ax.plot([p['gflops'] for p in c], [p['mAP'] for p in c], ls, marker='.', color=col, label=label)
        for run in statics:
            if not _exists(data, tag, 'standard', run):
                continue
            m = metrics(data, tag, 'standard', run)['ALL']
            mk = '*' if run.startswith('HAHI') else ('o' if run.startswith('global') else 's')
            ax.scatter(m['gflops'], m['mAP'], marker=mk, s=90, color='w', edgecolor='k', zorder=3)
            lab = run.replace('HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI tuned').replace(
                'HAHI_global_imgsz=1280,gcc=1.0_', 'HAHI-ext').replace('global@', 'G').replace('MS@', 'MS')
            ax.annotate(lab, (m['gflops'], m['mAP']), textcoords='offset points', xytext=(5, -10), fontsize=7)
        ax.set_xlabel('GFLOPs / image')
        ax.set_ylabel('mAP@[.5:.95]')
        ax.set_title(title + ': static methods (markers) vs per-image gating (curves)', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def static_engine_table():
    """Cost in 640x640-equivalent units when every pass runs on a static engine (640x640 for the 640 pass and for
    crops/tiles; a dedicated larger static engine for the high-resolution pass of MS)."""
    rows = [('**HAHI-S** (one 640x640 engine)', f'HAHI_{_T}_', None),
            ('**HAHI-S, route 0.1** (one 640x640 engine)', f'HAHI_{_T},route_ratio=0.1_', None),
            ('MS-tiled 1280 (one 640x640 engine)', 'MSTILE@1280', None),
            ('MS-tiled 1536 (one 640x640 engine)', 'MSTILE@1536', None),
            ('MS 640 + 1280x736 engine (16:9)', 'MS@640x1280_aspect=1.7778_', (1280, 736)),
            ('MS 640 + 1536x864 engine (16:9)', 'MS@640x1536_aspect=1.7778_', (1536, 864)),
            ('MS 640 + 1280x1280 engine', 'MS@640x1280_aspect=1.0_', (1280, 1280))]
    head = ['Method', 'Engines', 'Cost (640x640 units)', 'mAP', 'AP50', 'APs', 'APm', 'APl']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, run, eng in rows:
        if not _exists('vd_testdev', 'v8s', 'standard', run):
            continue
        m = metrics('vd_testdev', 'v8s', 'standard', run)['ALL']
        units = m['n_inf'] if eng is None else 1 + eng[0] * eng[1] / 640 ** 2
        engines = '640x640' if eng is None else f'640x640 + {eng[0]}x{eng[1]}'
        lines.append('| ' + ' | '.join([label, engines, f'{units:.2f}'] +
                                       [f"{m[c]:.4f}" for c in ['mAP', 'AP50', 'APs', 'APm', 'APl']]) + ' |')
    return '\n'.join(lines)


def improve_tables():
    """§15: diagnostics and candidate improvements of HAHI-S."""
    import json as _j
    T = {}
    dj = _j.load(open(os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'standard', 'diag_scale.json')))
    sr = [r for r in dj['scale_response'] if r['n'] >= 100]
    T['T17a'] = '| object size at detector input (px) | ' + ' | '.join(r['in_px'].replace('-1e+09', '+') for r in sr) + \
        ' |\n|' + '---|' * (len(sr) + 1) + '\n| recall@IoU0.5 (score ≥ 0.3) | ' + \
        ' | '.join(f"{r['recall']:.2f}" for r in sr) + ' |'
    cov = dj['hahis_magnification']
    head = ['native size (px)', 'n', 'dense window', 'packed canvas', 'global only', 'median magnification',
            'median input px: HAHI-S', 'median input px: MS 1280']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for c in cov:
        lines.append(f"| {c['native_px'].replace('-1e+09', '+')} | {c['n']} | {c['frac_window']:.2f} | "
                     f"{c['frac_packed']:.2f} | {c['frac_global_only']:.2f} | {c['median_mag']:.2f} | "
                     f"{c['median_input_px_hahis']:.1f} | {c['median_input_px_ms1280']:.1f} |")
    T['T17b'] = '\n'.join(lines)
    bins = ['0-8', '8-16', '16-24', '24-32', '32-48', '48-64', '64-96', '96-128', '128-192', '192+']
    lab = {'global@640': 'Global 640', 'global@1280': 'Global 1280', 'MS@640x1280': 'MS 640+1280',
           'HAHI': 'HAHI (thesis code)', f'HAHI_{_T}_': 'HAHI-S', f'HAHI_{_T},route_ratio=0.1_': 'HAHI-S, route 0.1'}
    lines = ['| AP@[.5:.95] by native size (val) | ' + ' | '.join(bins) + ' |', '|' + '---|' * (len(bins) + 1)]
    for k, v in dj['by_size'].items():
        best = None
        lines.append(f"| {lab.get(k, k)} | " + ' | '.join(f'{x:.3f}' for x in v['ap']) + ' |')
    T['T17c'] = '\n'.join(lines)
    sra = lambda l, t='': f'HAHI_{_T}{t},packing_mode=sra,canvas=640,sra_lambda={l}_'
    rows = [('HAHI-S', f'HAHI_{_T}_'), ('HAHI-S + TAF', f'HAHI_{_T},trunc=drop_')] + \
        [(f'HAHI-S-SRA, λ={l}', sra(l)) for l in ['10', '6', '4', '3', '2', '1.5']] + \
        [(f'HAHI-S-SRA + TAF, λ={l}', sra(l, ',trunc=drop')) for l in ['3', '2', '1']] + \
        [('HAHI-S, route 0.1', f'HAHI_{_T},route_ratio=0.1_'),
         ('**HAHI-S, route 0.1 + TAF**', f'HAHI_{_T},trunc=drop,route_ratio=0.1_'),
         ('MS 640+1280 (dynamic)', 'MS@640x1280'), ('MS 640+1536 (dynamic)', 'MS@640x1536'),
         ('MS 640 + 1280x736 engine', 'MS@640x1280_aspect=1.7778_'),
         ('MS 640 + 1536x864 engine', 'MS@640x1536_aspect=1.7778_'),
         ('MS 640 + 1280x1280 engine', 'MS@640x1280_aspect=1.0_')]
    T['T17d'] = md_table('vd_val', 'v8s', 'standard', [r for r in rows if _exists('vd_val', 'v8s', 'standard', r[1])],
                         cols=('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf'), vram=False)
    eng = {'MS@640x1280_aspect=1.7778_': (1280, 736), 'MS@640x1536_aspect=1.7778_': (1536, 864),
           'MS@640x1280_aspect=1.0_': (1280, 1280)}
    td = [('HAHI-S', f'HAHI_{_T}_'), ('**HAHI-S + TAF**', f'HAHI_{_T},trunc=drop_'),
          ('HAHI-S, route 0.1', f'HAHI_{_T},route_ratio=0.1_'),
          ('**HAHI-S, route 0.1 + TAF**', f'HAHI_{_T},trunc=drop,route_ratio=0.1_'),
          ('MS-tiled 1280', 'MSTILE@1280'), ('MS-tiled 1536', 'MSTILE@1536'),
          ('MS 640 + 1280x736 engine', 'MS@640x1280_aspect=1.7778_'),
          ('MS 640 + 1536x864 engine', 'MS@640x1536_aspect=1.7778_'),
          ('MS 640 + 1280x1280 engine', 'MS@640x1280_aspect=1.0_'),
          ('MS 640+1280 (dynamic GPU input)', 'MS@640x1280'), ('MS 640+1536 (dynamic GPU input)', 'MS@640x1536'),
          ('MS 640+1280+1536 (dynamic GPU input)', 'MS@640x1280x1536')]
    head = ['Method', 'cost (640x640 units)', 'mAP', 'AP50', 'APs', 'APm', 'APl']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, run in td:
        if not _exists('vd_testdev', 'v8s', 'standard', run):
            continue
        m = metrics('vd_testdev', 'v8s', 'standard', run)['ALL']
        if run in eng:
            u = f'{1 + eng[run][0] * eng[run][1] / 640 ** 2:.2f}'
        elif run.startswith('MS@'):
            u = 'n/a (dynamic)'
        else:
            u = f"{m['n_inf']:.2f}"
        lines.append('| ' + ' | '.join([label, u] + [f"{m[c]:.4f}" for c in ['mAP', 'AP50', 'APs', 'APm', 'APl']]) + ' |')
    T['T17e'] = '\n'.join(lines)
    a, b = f'HAHI_{_T},trunc=drop_', f'HAHI_{_T},trunc=drop,route_ratio=0.1_'
    T['T17f'] = '\n\n'.join([
        'HAHI-S + TAF 기준:\n\n' + boot_table('vd_testdev', 'v8s', a, [
            ('HAHI-S (TAF ablation)', f'HAHI_{_T}_'), ('MS 640 + 1280x736 engine', 'MS@640x1280_aspect=1.7778_'),
            ('MS-tiled 1280', 'MSTILE@1280'), ('MS 640+1280 (dynamic)', 'MS@640x1280')]),
        'HAHI-S, route 0.1 + TAF 기준:\n\n' + boot_table('vd_testdev', 'v8s', b, [
            ('HAHI-S, route 0.1 (TAF ablation)', f'HAHI_{_T},route_ratio=0.1_'),
            ('MS 640 + 1536x864 engine', 'MS@640x1536_aspect=1.7778_'),
            ('MS 640 + 1280x1280 engine', 'MS@640x1280_aspect=1.0_'), ('MS-tiled 1536', 'MSTILE@1536'),
            ('MS 640+1280 (dynamic)', 'MS@640x1280'), ('MS 640+1536 (dynamic)', 'MS@640x1536')])])
    # fusion: NMS vs box voting, applied identically to HAHI-S and MS
    fz = [('HAHI-S + TAF', f'HAHI_{_T},trunc=drop_', None),
          ('HAHI-S + TAF + voting', f'HAHI_{_T},trunc=drop,vote=0.6_', None),
          ('HAHI-S, route 0.1 + TAF', f'HAHI_{_T},trunc=drop,route_ratio=0.1_', None),
          ('HAHI-S, route 0.1 + TAF + voting', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_', None),
          ('MS 640 + 1280x736 engine', 'MS@640x1280_aspect=1.7778_', (1280, 736)),
          ('MS 640 + 1280x736 engine + voting', 'MS@640x1280_aspect=1.7778,vote=0.6_', (1280, 736)),
          ('MS 640 + 1536x864 engine', 'MS@640x1536_aspect=1.7778_', (1536, 864)),
          ('MS 640 + 1536x864 engine + voting', 'MS@640x1536_aspect=1.7778,vote=0.6_', (1536, 864)),
          ('MS 640 + 1280x1280 engine', 'MS@640x1280_aspect=1.0_', (1280, 1280)),
          ('MS 640 + 1280x1280 engine + voting', 'MS@640x1280_aspect=1.0,vote=0.6_', (1280, 1280)),
          ('MS 640+1280 (dynamic) + voting', 'MS@640x1280_vote=0.6_', 'dyn'),
          ('MS 640+1536 (dynamic) + voting', 'MS@640x1536_vote=0.6_', 'dyn')]
    head = ['Method', 'cost (640x640 units)', 'val mAP', 'test-dev mAP', 'AP50', 'APs', 'APm', 'APl']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, run, e in fz:
        if not _exists('vd_testdev', 'v8s', 'standard', run):
            continue
        m = metrics('vd_testdev', 'v8s', 'standard', run)['ALL']
        v = metrics('vd_val', 'v8s', 'standard', run)['ALL']['mAP'] if _exists('vd_val', 'v8s', 'standard', run) else None
        u = 'n/a (dynamic)' if e == 'dyn' else (f"{m['n_inf']:.2f}" if e is None else f'{1 + e[0] * e[1] / 640 ** 2:.2f}')
        lines.append('| ' + ' | '.join([label, u, f'{v:.4f}' if v is not None else 'n/a'] +
                                       [f"{m[c]:.4f}" for c in ['mAP', 'AP50', 'APs', 'APm', 'APl']]) + ' |')
    T['T17g'] = '\n'.join(lines)
    a2, b2 = f'HAHI_{_T},trunc=drop,vote=0.6_', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_'
    T['T17h'] = '\n\n'.join([
        'HAHI-S + TAF + voting 기준:\n\n' + boot_table('vd_testdev', 'v8s', a2, [
            ('HAHI-S + TAF (voting ablation)', f'HAHI_{_T},trunc=drop_'),
            ('MS 640 + 1280x736 engine + voting', 'MS@640x1280_aspect=1.7778,vote=0.6_'),
            ('MS 640+1280 (dynamic) + voting', 'MS@640x1280_vote=0.6_')]),
        'HAHI-S, route 0.1 + TAF + voting 기준:\n\n' + boot_table('vd_testdev', 'v8s', b2, [
            ('HAHI-S, route 0.1 + TAF (voting ablation)', f'HAHI_{_T},trunc=drop,route_ratio=0.1_'),
            ('MS 640 + 1536x864 engine + voting', 'MS@640x1536_aspect=1.7778,vote=0.6_'),
            ('MS 640 + 1280x1280 engine + voting', 'MS@640x1280_aspect=1.0,vote=0.6_'),
            ('MS 640+1536 (dynamic) + voting', 'MS@640x1536_vote=0.6_'),
            ('MS 640+1280+1536 (dynamic, no voting)', 'MS@640x1280x1536')])])
    dota = [('HAHI (thesis code)', 'HAHI', None), ('HAHI-S', f'HAHI_{_T}_', None),
            ('HAHI-S + TAF + voting', f'HAHI_{_T},trunc=drop,vote=0.6_', None),
            ('HAHI-S, route 0.1', f'HAHI_{_T},route_ratio=0.1_', None),
            ('**HAHI-S, route 0.1 + TAF + voting**', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_', None),
            ('MS-tiled 1280 (640 engine only)', 'MSTILE@1280', None), ('MS-tiled 1536 (640 engine only)', 'MSTILE@1536', None),
            ('MS 640 + 1280x1280 engine + voting', 'MS@640x1280_aspect=1.0,vote=0.6_', (1280, 1280)),
            ('MS 640 + 1536x1536 engine + voting', 'MS@640x1536_aspect=1.0,vote=0.6_', (1536, 1536)),
            ('MS 640+1280 (dynamic) + voting', 'MS@640x1280_vote=0.6_', 'dyn'),
            ('MS 640+1536 (dynamic) + voting', 'MS@640x1536_vote=0.6_', 'dyn'),
            ('HAHI-ext (global 1280, thesis code)', 'HAHI_global_imgsz=1280,gcc=1.0_', 'ext')]
    head = ['Method', 'cost (640x640 units)', 'mAP', 'AP50', 'APs', 'APm', 'APl']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, run, e in dota:
        if not _exists('dota_val', 'v8s_dota', 'standard', run):
            continue
        m = metrics('dota_val', 'v8s_dota', 'standard', run)['ALL']
        u = {'dyn': 'n/a (dynamic)', 'ext': 'n/a (1280 global pass)'}.get(e) if isinstance(e, str) else \
            (f"{m['n_inf']:.2f}" if e is None else f'{1 + e[0] * e[1] / 640 ** 2:.2f}')
        lines.append('| ' + ' | '.join([label, u] + [f"{m[c]:.4f}" for c in ['mAP', 'AP50', 'APs', 'APm', 'APl']]) + ' |')
    T['T17i'] = '\n'.join(lines)
    return T


def _holm(ps):
    """Holm step-down adjustment of one-sided p-values (bootstrap resolution 1/1000: p=0 is reported as <0.001)."""
    p = [max(x, 1e-3) for x in ps]
    order = sorted(range(len(p)), key=lambda k: p[k])
    adj, run = [0.0] * len(p), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[k]))
        adj[k] = run
    return adj


def fair_tables():
    """§16: tuning-budget audit, frozen-configuration comparison with Holm correction, and MS tuned on test-dev."""
    import json as _j
    T = {}
    A, B = f'HAHI_{_T},trunc=drop,vote=0.6_', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_'
    cnt = lambda d, pre: sum(1 for n in os.listdir(os.path.join(RUN_ROOT, d, 'v8s', 'standard'))
                             if _exists(d, 'v8s', 'standard', n) and any(n.startswith(p) for p in pre))
    grid = _j.load(open(os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'ms_grid.json')))
    hv, mv = cnt('vd_val', ('HAHI', 'SBSI', 'Packing')), cnt('vd_val', ('MS@', 'MSTILE'))
    T['T18a'] = '\n'.join([
        '| | HAHI family | MS family |', '|---|---|---|',
        f'| configurations evaluated on val (before this section) | {hv} | {mv} |',
        f'| + equal-budget MS grid (offline fusion, this section) | — | +{len(grid)} |',
        f'| total on val | {hv} | {mv + len(grid)} |'])
    g640 = {t: _j.load(open(os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', 'global@640', 'meta.json')))['gflops640']
            for t in ('v8s', 'v8n', 'y11s')}

    def cost(t, run, setting):
        m = metrics('vd_testdev', t, 'standard', run)['ALL']
        if setting == 'dynamic':
            return m['gflops'] / g640[t]
        meta = _j.load(open(os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', run, 'meta.json')))
        return meta.get('static_units') or m['n_inf']

    plan = [('static', A, 'MSG_1280w_n0.6_v0.6'), ('static', B, 'MSG_1536w_n0.7_v0.7'),
            ('dynamic', A, 'MSG_1280d_n0.6_v0.6'), ('dynamic', B, 'MSG_1536d_n0.7_v0.7')]
    lab = {A: 'HAHI-S + TAF + voting', B: 'HAHI-S, route 0.1 + TAF + voting',
           'MSG_1280w_n0.6_v0.6': 'MS 640 + 1280x736 engine, NMS 0.6, vote 0.6',
           'MSG_1536w_n0.7_v0.7': 'MS 640 + 1536x864 engine, NMS 0.7, vote 0.7',
           'MSG_1280d_n0.6_v0.6': 'MS 640 + 1280 dynamic, NMS 0.6, vote 0.6',
           'MSG_1536d_n0.7_v0.7': 'MS 640 + 1536 dynamic, NMS 0.7, vote 0.7'}
    rows = []
    for t in ('v8s', 'v8n', 'y11s'):
        for setting, h, ms in plan:
            if not (_exists('vd_testdev', t, 'standard', h) and _exists('vd_testdev', t, 'standard', ms)):
                continue
            bp = os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', f'bootstrap_ALL_thr0_ref-{h}.json')
            b = _j.load(open(bp)).get(ms) if os.path.exists(bp) else None
            mh, mm = metrics('vd_testdev', t, 'standard', h)['ALL'], metrics('vd_testdev', t, 'standard', ms)['ALL']
            rows.append([t, setting, lab[h], f'{cost(t, h, setting):.2f}', lab[ms], f'{cost(t, ms, setting):.2f}',
                         mh['mAP'], mm['mAP'], b])
    adj = _holm([r[8]['mAP']['p_le_0'] if r[8] else 1.0 for r in rows])
    head = ['Detector', 'Setting', 'HAHI', 'cost', 'MS (selected on val, frozen)', 'cost', 'HAHI mAP', 'MS mAP',
            'ΔmAP [95% CI]', 'P(Δ≤0)', 'Holm-adjusted', 'ΔAPs [95% CI]']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    f = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    for r, a in zip(rows, adj):
        b = r[8]
        lines.append('| ' + ' | '.join(r[:6] + [f'{r[6]:.4f}', f'{r[7]:.4f}', f(b['mAP']) if b else 'n/a',
                                               (f"{max(b['mAP']['p_le_0'], 0.001):.3f}" if b else 'n/a'),
                                               f'{a:.3f}', f(b['APs']) if b else 'n/a']) + ' |')
    T['T18b'] = '\n'.join(lines)
    # stress test: MS tuned on test-dev itself (v8s)
    gp = os.path.join(RUN_ROOT, 'vd_testdev', 'v8s', 'ms_grid.json')
    if os.path.exists(gp):
        from .fairness import select
        gt = _j.load(open(gp))
        lines = ['| Setting | HAHI | cost | HAHI mAP | best MS on test-dev within cost (oracle) | cost | MS mAP | Δ |',
                 '|---|---|---|---|---|---|---|---|']
        for setting, h in [('static', A), ('static', B), ('dynamic', A), ('dynamic', B)]:
            c = float(cost('v8s', h, setting))
            r = select(gt, c, setting)
            if r is None:
                continue
            mh = metrics('vd_testdev', 'v8s', 'standard', h)['ALL']['mAP']
            u = r['static_units'] if setting == 'static' else r['dyn_units']
            name = '+'.join(f"{p[0]}{'d' if p[1] == 0 else ('s' if p[1] == 1.0 else 'w')}" for p in r['passes'][1:])
            lines.append(f"| {setting} | {lab[h]} | {c:.2f} | {mh:.4f} | MS 640+{name}, NMS {r['nms']}, vote {r['vote']} | "
                         f"{u:.2f} | {r['mAP']:.4f} | {mh - r['mAP']:+.4f} |")
        T['T18c'] = '\n'.join(lines)
    # transfer of the fusion gain (ablation) to other detectors
    lines = ['| Detector | HAHI-S | + TAF + voting | Δ | HAHI-S route 0.1 | + TAF + voting | Δ |', '|---|---|---|---|---|---|---|']
    for t in ('v8s', 'v8n', 'y11s'):
        g = lambda r: metrics('vd_testdev', t, 'standard', r)['ALL']['mAP'] if _exists('vd_testdev', t, 'standard', r) else None
        a0, a1 = g(f'HAHI_{_T}_'), g(A)
        b0, b1 = g(f'HAHI_{_T},route_ratio=0.1_'), g(B)
        if None in (a0, a1, b0, b1):
            continue
        lines.append(f'| {t} | {a0:.4f} | {a1:.4f} | {a1 - a0:+.4f} | {b0:.4f} | {b1:.4f} | {b1 - b0:+.4f} |')
    T['T18d'] = '\n'.join(lines)
    return T


def aspect_table():
    """§16.6: fixed-input comparison split by frame aspect ratio (a deployed camera has one aspect ratio)."""
    import json as _j
    import numpy as _np
    from .data import load_dataset
    from .evaluate import subset_indices
    A, B = f'HAHI_{_T},trunc=drop,vote=0.6_', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_'
    recs = load_dataset('vd_testdev')[0]
    idx = list(range(len(recs)))
    plan = [(A, 'MSG_1280w_n0.6_v0.6', (1280, 736)), (B, 'MSG_1536w_n0.7_v0.7', (1536, 864))]
    eng43 = {'MSG_1280w_n0.6_v0.6': (1280, 960), 'MSG_1536w_n0.7_v0.7': (1536, 1152)}
    head = ['Detector', 'Frames', 'HAHI', 'HAHI passes', 'HAHI mAP', 'MS 16:9 engine', 'MS cost', 'MS mAP',
            'ΔmAP [95% CI]', 'P(Δ≤0)', 'ΔAPs [95% CI]']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    f = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    lab = {A: 'HAHI-S + TAF + voting', B: 'HAHI-S, route 0.1 + TAF + voting'}
    for t in ('v8s', 'v8n', 'y11s'):
        for sub, name in [('ALL', 'all (1,610)'), ('W169', '16:9 only (1,336)'), ('W43', '4:3 only (274)')]:
            si = set(subset_indices(recs, idx, sub))
            for h, ms, e in plan:
                bp = os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', f'bootstrap_{sub}_thr0_ref-{h}.json')
                if not os.path.exists(bp):
                    continue
                b = _j.load(open(bp)).get(ms)
                if b is None:
                    continue
                val = lambda run: float(_np.load(os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', run,
                                                              f'boot_reps_{sub}_thr0_n1000_s0.npy'))[0, 0])
                per = _j.load(open(os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', h, 'per_image.json')))
                hp = _np.mean([p['n_inf'] for p in per if p['i'] in si])
                if sub == 'W43':
                    mc = 1 + e[0] * e[1] / 640 ** 2
                    mcs = f'{mc:.2f} (as 16:9 engine: pads 4:3 frames)'
                else:
                    mcs = f'{1 + e[0] * e[1] / 640 ** 2:.2f}'
                lines.append('| ' + ' | '.join([t, name, lab[h], f'{hp:.2f}', f'{val(h):.4f}',
                                                ms.replace('MSG_', '').replace('w_', ' 16:9, ').replace('_', ' '),
                                                mcs if sub != 'W43' else f'{1 + e[0] * e[1] / 640 ** 2:.2f}',
                                                f'{val(ms):.4f}', f(b['mAP']),
                                                f"{max(b['mAP']['p_le_0'], 0.001):.3f}", f(b['APs'])]) + ' |')
    return '\n'.join(lines)


def single_engine_table():
    """§16.6: only one 640x640 engine available -> MS must be tiled; both sides use the same view-aware fusion."""
    import json as _j
    A, B = f'HAHI_{_T},trunc=drop,vote=0.6_', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_'
    rows = [('HAHI-S + TAF + voting', A), ('HAHI-S, route 0.1 + TAF + voting', B)] + \
        [(f'MS-tiled {s} + TAF + voting', f'MSTILE@{s}_trunc=drop,vote=0.6_') for s in (960, 1136, 1280, 1536)]
    head = ['Detector', 'Method', 'val mAP', '#640x640 passes', 'test-dev mAP', 'APs', 'APm', 'APl']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for t in ('v8s', 'v8n', 'y11s'):
        for label, run in rows:
            if not _exists('vd_testdev', t, 'standard', run):
                continue
            m = metrics('vd_testdev', t, 'standard', run)['ALL']
            v = metrics('vd_val', 'v8s', 'standard', run)['ALL']['mAP'] if t == 'v8s' and \
                _exists('vd_val', 'v8s', 'standard', run) else None
            lines.append('| ' + ' | '.join([t, label, f'{v:.4f}' if v is not None else '', f"{m['n_inf']:.2f}"] +
                                           [f"{m[c]:.4f}" for c in ['mAP', 'APs', 'APm', 'APl']]) + ' |')
    out = ['\n'.join(lines)]
    for t in ('v8s', 'v8n', 'y11s'):
        for ref, lab in [(A, 'HAHI-S + TAF + voting'), (B, 'HAHI-S, route 0.1 + TAF + voting')]:
            bp = os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', f'bootstrap_ALL_thr0_ref-{ref}.json')
            if not os.path.exists(bp):
                continue
            b = _j.load(open(bp))
            sel = [(f'MS-tiled {s} + TAF + voting', f'MSTILE@{s}_trunc=drop,vote=0.6_') for s in (960, 1136, 1280, 1536)
                   if f'MSTILE@{s}_trunc=drop,vote=0.6_' in b]
            if sel:
                out.append(f'{t}, {lab} 기준:\n\n' + boot_table('vd_testdev', t, ref, sel))
    return '\n\n'.join(out)


def plugin_table():
    """§17: view-aware fusion (TAF + box voting) as a training-free plug-in for crop-based inference methods."""
    import json as _j
    methods = [('UC 2x2 (global + 4 quadrants)', 'UC2x2', 'UC2x2_vote=0.6_', 'UC2x2_trunc=drop,vote=0.6_'),
               ('SAHI 512 (+ full-image pass)', 'SAHI@512_post=NMS,metric=IOU_', 'SAHI@512_post=NMS,metric=IOU,vote=0.6_',
                'SAHI@512_post=NMS,metric=IOU,trunc=drop,vote=0.6_'),
               ('MS-tiled 960', 'MSTILE@960', None, 'MSTILE@960_trunc=drop,vote=0.6_'),
               ('MS-tiled 1280', 'MSTILE@1280', 'MSTILE@1280_vote=0.6_', 'MSTILE@1280_trunc=drop,vote=0.6_'),
               ('MS-tiled 1536', 'MSTILE@1536', None, 'MSTILE@1536_trunc=drop,vote=0.6_'),
               ('DAHI (paper default; voting replaces CMA-NMS)', 'DAHI', None, 'DAHI_trunc=drop,vote=0.6_'),
               ('DAHI (N_ROI=2)', 'DAHI_n_roi=2_', None, 'DAHI_n_roi=2,trunc=drop,vote=0.6_'),
               ('PAN fine + packing', 'PAN_bandwidth=256_', None, 'PAN_bandwidth=256,trunc=drop,vote=0.6_'),
               ('HAHI-S', f'HAHI_{_T}_', None, f'HAHI_{_T},trunc=drop,vote=0.6_'),
               ('HAHI-S, route 0.1', f'HAHI_{_T},route_ratio=0.1_', None, f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_')]
    head = ['Detector', 'Method', 'passes', 'base mAP', '+ voting', '+ TAF + voting', 'Δ (TAF + voting)', 'ΔAPs']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    g = lambda d, t, r: metrics(d, t, 'standard', r)['ALL'] if r and _exists(d, t, 'standard', r) else None
    for data, tags in [('vd_val', ('v8s',)), ('vd_testdev', ('v8s', 'v8n', 'y11s'))]:
        for t in tags:
            for label, base, vo, tv in methods:
                b, v, f = g(data, t, base), g(data, t, vo), g(data, t, tv)
                if b is None or f is None:
                    continue
                lines.append('| ' + ' | '.join([f'{t} ({"val" if data == "vd_val" else "test-dev"})', label,
                                                f"{b['n_inf']:.2f}", f"{b['mAP']:.4f}",
                                                f"{v['mAP']:.4f}" if v else '—', f"{f['mAP']:.4f}",
                                                f"**{f['mAP'] - b['mAP']:+.4f}**", f"{f['APs'] - b['APs']:+.4f}"]) + ' |')
    out = ['\n'.join(lines)]
    for t in ('v8s', 'v8n', 'y11s'):
        rows = []
        for label, base, vo, tv in methods:
            bp = os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', f'bootstrap_ALL_thr0_ref-{tv}.json')
            if os.path.exists(bp) and base in _j.load(open(bp)):
                rows.append((label, tv, base))
        if rows:
            head = ['Method', 'Δ (fusion − base) [95% CI]', 'P(Δ≤0)', 'ΔAPs [95% CI]']
            ls = [f'{t}, test-dev — 융합 적용 전후 대응 부트스트랩:', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
            for label, tv, base in rows:
                x = _j.load(open(os.path.join(RUN_ROOT, 'vd_testdev', t, 'standard', f'bootstrap_ALL_thr0_ref-{tv}.json')))[base]
                fm = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                ls.append(f"| {label} | {fm(x['mAP'])} | {max(x['mAP']['p_le_0'], 0.001):.3f} | {fm(x['APs'])} |")
            out.append('\n'.join(ls))
    return '\n\n'.join(out)


def holdout_tables():
    """§18: pre-registered hold-out evaluation (prereg.json): VisDrone-VID (3 detectors) and UAVDT (YOLOv8s)."""
    import json as _j
    from .holdout import PREREG, safe
    M = PREREG['methods']
    F1 = PREREG['primary_families']['F1_plugin']['pairs']
    F2 = PREREG['primary_families']['F2_single_engine']['pairs']
    D = PREREG['descriptive_pairs']['pairs']
    sets = [('vid_holdout', t) for t in ('v8s', 'v8n', 'y11s')] + [('uavdt_test', 'v8s')]
    T = {}

    def boot(data, tag, a, b):
        p = os.path.join(RUN_ROOT, data, tag, 'standard', f'bootstrap_ALL_thr0_ref-{safe(M[a])}.json')
        return _j.load(open(p)).get(safe(M[b])) if os.path.exists(p) else None

    fm = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    # overview: mAP and passes of every pre-registered method
    head = ['Method'] + [f"{'VID' if d == 'vid_holdout' else 'UAVDT'} {t}" for d, t in sets] + ['passes (VID v8s)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, spec in M.items():
        cells = [label]
        for d, t in sets:
            r = safe(spec)
            cells.append(f"{metrics(d, t, 'standard', r)['ALL']['mAP']:.4f}" if _exists(d, t, 'standard', r) else 'n/a')
        r = safe(spec)
        cells.append(f"{metrics('vid_holdout', 'v8s', 'standard', r)['ALL']['n_inf']:.2f}"
                     if _exists('vid_holdout', 'v8s', 'standard', r) else '')
        lines.append('| ' + ' | '.join(cells) + ' |')
    T['T22a'] = '\n'.join(lines)
    # primary families with Holm correction within (dataset, family)
    for key, fam, title in [('T22b', F1, 'F1'), ('T22c', F2, 'F2')]:
        head = ['Data', 'Detector', 'Comparison', 'ΔmAP [95% CI, sequence bootstrap]', 'P(Δ≤0)', 'Holm', 'ΔAPs [95% CI]']
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for data in ('vid_holdout', 'uavdt_test'):
            rows = []
            for d, t in sets:
                if d != data:
                    continue
                for a, b in fam:
                    x = boot(d, t, a, b)
                    if x:
                        rows.append((d, t, a, b, x))
            adj = _holm([x['mAP']['p_le_0'] for *_, x in rows])
            for (d, t, a, b, x), h in zip(rows, adj):
                lines.append('| ' + ' | '.join(['VID' if d == 'vid_holdout' else 'UAVDT', t, f'{a} vs {b}', fm(x['mAP']),
                                                f"{max(x['mAP']['p_le_0'], 0.001):.3f}", f'{h:.3f}', fm(x['APs'])]) + ' |')
        T[key] = '\n'.join(lines)
    head = ['Data', 'Detector', 'Comparison', 'ΔmAP [95% CI]', 'P(Δ≤0)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for d, t in sets:
        for a, b in D:
            x = boot(d, t, a, b)
            if x:
                lines.append('| ' + ' | '.join(['VID' if d == 'vid_holdout' else 'UAVDT', t, f'{a} vs {b}', fm(x['mAP']),
                                                f"{max(x['mAP']['p_le_0'], 0.001):.3f}"]) + ' |')
    T['T22d'] = '\n'.join(lines)
    # exploratory (not pre-registered): VID split by frame width
    import pickle as _p
    from .data import load_dataset
    from .evaluate import subset_indices
    recs = load_dataset('vid_holdout')[0]
    head = ['Detector', 'Method', 'passes ≤1920', 'mAP ≤1920 (772 frames, 20 seq.)', 'passes >1920',
            'mAP >1920 (184 frames, 4 seq.)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for t in ('v8s', 'v8n', 'y11s'):
        for label in ['HAHI-S + F', 'HAHI-S route 0.1 + F', 'MS-tiled 1136 + F', 'MS-tiled 1280 + F', 'MS-tiled 1536 + F',
                      'MS dyn 1280 (A budget)']:
            rd = os.path.join(RUN_ROOT, 'vid_holdout', t, 'standard', safe(M[label]))
            bt = os.path.join(rd, 'boot_tables_ALL_thr0.pkl')
            if not os.path.exists(bt):
                continue
            ta = _p.load(open(bt, 'rb'))[0]
            per = {p['i']: p for p in _j.load(open(os.path.join(rd, 'per_image.json')))}
            cells = [t, label]
            for sub in ('LE1920', 'GT1920'):
                si = subset_indices(recs, range(len(recs)), sub)
                w = _np_zeros(len(ta.img_pos))
                for i in si:
                    w[ta.img_pos[i]] = 1.0
                cells += [f"{sum(per[i]['n_inf'] for i in si) / len(si):.2f}", f'{ta.ap(w):.4f}']
            lines.append('| ' + ' | '.join(cells) + ' |')
    T['T22e'] = '\n'.join(lines)
    return T


def _np_zeros(n):
    import numpy as _np
    return _np.zeros(n)


def trt_tables():
    """§19: TensorRT static engines on the local GPU (FP16 / INT8): accuracy, per-engine INT8 robustness, latency."""
    import json as _j
    T = {}
    rows = [('Global 640', 'global@640', 'global@640'),
            ('HAHI-S + F', f'HAHI_{_T},trunc=drop,vote=0.6_', f'HAHI_{_T},trunc=drop,vote=0.6_'),
            ('HAHI-S route 0.1 + F', f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_',
             f'HAHI_{_T},trunc=drop,vote=0.6,route_ratio=0.1_'),
            ('MS-tiled 1136 + F (640 engine only)', 'MSTILE@1136_trunc=drop,vote=0.6_', 'MSTILE@1136_trunc=drop,vote=0.6_'),
            ('MS-tiled 1280 + F (640 engine only)', 'MSTILE@1280_trunc=drop,vote=0.6_', 'MSTILE@1280_trunc=drop,vote=0.6_'),
            ('MS-tiled 1536 + F (640 engine only)', 'MSTILE@1536_trunc=drop,vote=0.6_', 'MSTILE@1536_trunc=drop,vote=0.6_'),
            ('MS 640 + 1280x736 engine', 'MS@640x1280_aspect=1.7778,vote=0.6_', 'MS@640x1280_aspect=1.7778,vote=0.6_'),
            ('MS 640 + 1536x864 engine', 'MSG_1536w_n0.7_v0.7', 'MS@640x1536_aspect=1.7778,iou=0.7,vote=0.7_')]
    head = ['Method', 'passes', 'mAP FP32 (PyTorch)', 'mAP FP16 (TRT)', 'mAP INT8 (TRT)', 'INT8 − FP32', 'APs INT8',
            'latency FP16 ms (score ≥ 0.3)', 'latency INT8 ms (score ≥ 0.3)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, fp32, trt in rows:
        g = lambda prot, r: metrics('vd_testdev', 'v8s', prot, r)['ALL'] if _exists('vd_testdev', 'v8s', prot, r) else None
        a, h, i = g('standard', fp32), g('standard_trt_fp16', trt), g('standard_trt_int8', trt)
        lat = []
        for p in ('fp16', 'int8'):
            c = cost_only('vd_testdev', 'v8s', f'thesis_trt_{p}', trt)['ALL'] if _exists('vd_testdev', 'v8s', f'thesis_trt_{p}', trt) else None
            lat.append(f"{c['total_ms']:.1f}" if c else 'n/a')
        cells = [label, f"{i['n_inf']:.2f}" if i else '', f"{a['mAP']:.4f}" if a else 'n/a', f"{h['mAP']:.4f}" if h else 'n/a',
                 f"{i['mAP']:.4f}" if i else 'n/a', f"{i['mAP'] - a['mAP']:+.4f}" if (a and i) else 'n/a',
                 f"{i['APs']:.4f}" if i else 'n/a'] + lat
        lines.append('| ' + ' | '.join(cells) + ' |')
    T['T23a'] = '\n'.join(lines)
    p = os.path.join(os.path.dirname(__file__), 'trt_work', 'engine_single_pass.json')
    if os.path.exists(p):
        r = _j.load(open(p))['results']
        lines = ['| Engine (static) | FP16 mAP | INT8 mAP | INT8 − FP16 | FP16 APs | INT8 APs |', '|---|---|---|---|---|---|']
        for s, shape in [(640, '640x640'), (1280, '1280x736'), (1536, '1536x864')]:
            a, b = r[f'fp16_{s}'], r[f'int8_{s}']
            lines.append(f"| {shape} | {a['mAP']:.4f} | {b['mAP']:.4f} | {b['mAP'] - a['mAP']:+.4f} | {a['APs']:.4f} | {b['APs']:.4f} |")
        T['T23b'] = '\n'.join(lines)
    return T


def overhead_tables():
    """§21: fusion overhead (merge-stage time, batch 1, idle GPU), deployment-threshold accuracy, alternative fusions."""
    import json as _j
    import numpy as _np
    T = {}
    R = os.path.join(RUN_ROOT, 'vd_testdev', 'v8s')
    F = 'trunc=drop,vote=0.6'
    pairs = [('UC 2x2', 'UC2x2', f'UC2x2_{F}_'), ('SAHI 512', 'SAHI@512_post=NMS,metric=IOU_', f'SAHI@512_post=NMS,metric=IOU,{F}_'),
             ('DAHI (CMA-NMS → F)', 'DAHI_c_th=0.3_', f'DAHI_c_th=0.3,{F}_'), ('PAN', 'PAN_bandwidth=256_', f'PAN_bandwidth=256,{F}_'),
             ('MS-tiled 1280', 'MSTILE@1280', f'MSTILE@1280_{F}_'), ('HAHI-S', f'HAHI_{_T}_', f'HAHI_{_T},{F}_')]

    def per(prot, r):
        p = os.path.join(R, prot, r, 'per_image.json')
        return _j.load(open(p)) if os.path.exists(p) else None

    def met(prot, r):
        p = os.path.join(R, prot, r, 'metrics_thr0_ign1.json')
        return _j.load(open(p))['ALL'] if os.path.exists(p) else None

    ms = lambda q, k: 1000 * _np.mean([x[k] for x in q])
    head = ['Method', 'passes', 'inference ms', 'merge ms: base', 'merge ms: + F', 'overhead ms (% of inference)',
            'mAP base', 'mAP + F', 'Δ', 'ΔAP50', 'ΔAP75']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, a, b in pairs:
        pa, pb, ma, mb = per('thesis_b1_lat', a), per('thesis_b1_lat', b), met('thesis_b1_lat', a), met('thesis_b1_lat', b)
        if not (pa and pb and ma and mb):
            continue
        o = ms(pb, 'merge') - ms(pa, 'merge')
        lines.append('| ' + ' | '.join([label, f"{_np.mean([x['n_inf'] for x in pa]):.2f}", f"{ms(pa, 'infer'):.1f}",
                                        f"{ms(pa, 'merge'):.2f}", f"{ms(pb, 'merge'):.2f}", f"{o:+.2f} ({100 * o / ms(pa, 'infer'):+.1f}%)",
                                        f"{ma['mAP']:.4f}", f"{mb['mAP']:.4f}", f"{mb['mAP'] - ma['mAP']:+.4f}",
                                        f"{mb['AP50'] - ma['AP50']:+.4f}", f"{mb['AP75'] - ma['AP75']:+.4f}"]) + ' |')
    T['T24a'] = '\n'.join(lines)
    head = ['Method', 'merge ms: NMS (CPU)', 'merge ms: + F, CPU voting', 'merge ms: + F, GPU voting', 'mAP + F (CPU)', 'mAP + F (GPU)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, a, b in [pairs[1], pairs[4], pairs[5]]:
        pa, pb, pg = per('standard_b1_lat', a), per('standard_b1_lat', b), per('standard_b1_latgpu', b)
        mc, mg = met('standard', b), met('standard_b1_latgpu', b)
        if not (pa and pb and pg):
            continue
        lines.append('| ' + ' | '.join([label, f"{ms(pa, 'merge'):.1f}", f"{ms(pb, 'merge'):.1f}", f"{ms(pg, 'merge'):.1f}",
                                        f"{mc['mAP']:.5f}" if mc else 'n/a', f"{mg['mAP']:.5f}" if mg else 'n/a']) + ' |')
    T['T24b'] = '\n'.join(lines)
    # alternative fusion rules (all with TAF), val and test-dev
    lines = ['| Data | Method | NMS (no TAF) | TAF only (NMS) | voting only | **TAF + voting** | TAF + WBF (max) | TAF + WBF (avg) | TAF + GREEDYNMM (SAHI) |',
             '|---|---|---|---|---|---|---|---|---|']
    alt = [('SAHI 512', 'SAHI@512_post=NMS,metric=IOU_', 'SAHI@512_post=NMS,metric=IOU,trunc=drop_',
            'SAHI@512_post=NMS,metric=IOU,vote=0.6_', f'SAHI@512_post=NMS,metric=IOU,{F}_'),
           ('MS-tiled 1280', 'MSTILE@1280', 'MSTILE@1280_trunc=drop_', 'MSTILE@1280_vote=0.6_', f'MSTILE@1280_{F}_'),
           ('HAHI-S', f'HAHI_{_T}_', f'HAHI_{_T},trunc=drop_', None, f'HAHI_{_T},{F}_'),
           ('UC 2x2', 'UC2x2', 'UC2x2_trunc=drop_', 'UC2x2_vote=0.6_', f'UC2x2_{F}_')]
    for data in ('vd_val', 'vd_testdev'):
        for label, base, taf, vote, both in alt:
            g = lambda prot, r: (metrics(data, 'v8s', prot, r)['ALL']['mAP'] if r and _exists(data, 'v8s', prot, r) else None)
            cells = [data.replace('vd_', ''), label] + [f'{v:.4f}' if v is not None else '—' for v in (
                g('standard', base), g('standard', taf), g('standard', vote), g('standard', both),
                g('standard_fzwbf_max', both), g('standard_fzwbf_avg', both), g('standard_fznmm', both))]
            lines.append('| ' + ' | '.join(cells) + ' |')
    T['T24c'] = '\n'.join(lines)
    return T


def review_tables():
    """§22: where the remaining error is (TIDE-style oracles), realisable view-trust voting rules, and the
    resolution-adaptive global pass (exploratory)."""
    import json as _j
    T = {}
    V = os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'views')
    hp = os.path.join(V, 'headroom.json')
    if os.path.exists(hp):
        H = _j.load(open(hp))
        h, m = H['HAHI-S']['metrics'], H['MS-1280x736']['metrics']
        rows = [('plain', 'Without the plug-in (thesis merge / plain NMS)', 'reference'),
                ('base', '**With the plug-in (HAHI-S: TAF + voting; MS: voting)**', 'reference'),
                ('miss_pool', 'Recover GT that some view found but the fusion dropped', 'fusion: recall'),
                ('cls_pool', 'Fix a class confusion when another view proposed the right class', 'fusion: class consensus'),
                ('loc_view', 'Per object, take the top box of the best view', 'fusion: which view to trust (oracle)'),
                ('loc_pool', 'Per object, take the best of all candidate boxes', 'fusion: best-of-N (optimistic)'),
                ('fusion', 'All fusion oracles together', 'fusion: ceiling'),
                ('bkg', 'Remove background false positives', 'detector: scoring'),
                ('cls', 'Remove class-confusion false positives', 'detector: classification'),
                ('miss_det', 'Remove GT that no view detected', 'detector: recall')]
        head = ['Oracle', 'error type', 'HAHI-S mAP', 'Δ', 'ΔAP75', 'MS 640+1280x736 mAP', 'Δ']
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for k, label, kind in rows:
            dh = '' if k in ('plain', 'base') else f"{h[k]['mAP'] - h['base']['mAP']:+.4f}"
            da = '' if k in ('plain', 'base') else f"{h[k]['AP75'] - h['base']['AP75']:+.4f}"
            dm = '' if k in ('plain', 'base') else f"{m[k]['mAP'] - m['base']['mAP']:+.4f}"
            lines.append(f"| {label} | {kind} | {h[k]['mAP']:.4f} | {dh} | {da} | {m[k]['mAP']:.4f} | {dm} |")
        T['T25a'] = '\n'.join(lines)
        sh, sm = H['HAHI-S']['stats'], H['MS-1280x736']['stats']
        n = sh['n_gt']
        pc = lambda x: f'{x:,} ({100 * x / n:.1f}%)'
        lines = ['| GT objects (val, after ignore regions) | HAHI-S + F | MS 640+1280x736 + voting |', '|---|---|---|',
                 f"| matched after fusion (IoU ≥ 0.5) | {pc(sh['fused50'])} | {pc(sm['fused50'])} |",
                 f"| some candidate of some view (IoU ≥ 0.5) | {pc(sh['pool50'])} | {pc(sm['pool50'])} |",
                 f"| dropped by the fusion | {pc(sh['fusion_miss'])} | {pc(sm['fusion_miss'])} |",
                 f"| **never detected by any view** | **{pc(sh['det_miss'])}** | **{pc(sm['det_miss'])}** |"]
        for j, lab in enumerate(['< 16 px', '16–32 px', '32–96 px', '≥ 96 px']):
            lines.append(f"| … of which {lab} (share of GT of that size) | {sh['det_miss_size'][j]:,} "
                         f"({100 * sh['det_miss_size'][j] / sh['gt_size'][j]:.0f}%) | {sm['det_miss_size'][j]:,} "
                         f"({100 * sm['det_miss_size'][j] / sm['gt_size'][j]:.0f}%) |")
        lines += [f"| fused box IoU ≥ 0.75 / some candidate IoU ≥ 0.75 | {sh['fused75']:,} / {sh['pool75']:,} | "
                  f"{sm['fused75']:,} / {sm['pool75']:,} |",
                  f"| candidates / distinct views per matched object | {sh['tp_cluster_sum'] / sh['tp_with_cluster']:.2f} / "
                  f"{sh['tp_views_sum'] / sh['tp_with_cluster']:.2f} | {sm['tp_cluster_sum'] / sm['tp_with_cluster']:.2f} / "
                  f"{sm['tp_views_sum'] / sm['tp_with_cluster']:.2f} |",
                  f"| false positives: background / class confusion | {sh['types']['bkg']:,} / {sh['types']['cls']:,} | "
                  f"{sm['types']['bkg']:,} / {sm['types']['cls']:,} |"]
        T['T25b'] = '\n'.join(lines)
    fp = os.path.join(V, 'fusion_rules.json')
    if os.path.exists(fp):
        F = _j.load(open(fp))
        lab = {'score': 'score (current box voting)', 'score^2': 'score²', 'score^4': 'score⁴ (trust the top box)',
               'score*mag^1': 'score × view magnification', 'score*mag^2': 'score × magnification²',
               'score*R': 'score × scale response R(object px in view)', 'score*R^2': 'score × R²',
               'topview': 'one vote per view (its top box)', 'topview*R': 'one vote per view, × R'}
        h, m = F.get('HAHI-S', {}), F.get('MS-1280x736', {})
        lines = ['| Voting weight | HAHI-S mAP | Δ | ΔAP75 | MS mAP | Δ |', '|---|---|---|---|---|---|']
        for k, label in lab.items():
            if k not in h:
                continue
            cm = (f"{m[k]['mAP']:.4f} | {m[k]['mAP'] - m['score']['mAP']:+.4f}" if k in m and 'score' in m else '— | —')
            d = '' if k == 'score' else f"{h[k]['mAP'] - h['score']['mAP']:+.4f}"
            d75 = '' if k == 'score' else f"{h[k]['AP75'] - h['score']['AP75']:+.4f}"
            if k == 'score' and 'score' in m:
                cm = f"{m[k]['mAP']:.4f} | "
            lines.append(f"| {label} | {h[k]['mAP']:.4f} | {d} | {d75} | {cm} |")
        if h:
            lines.append(f"| *oracle: best view (T25a)* | *{_j.load(open(hp))['HAHI-S']['metrics']['loc_view']['mAP']:.4f}* | "
                         f"*{_j.load(open(hp))['HAHI-S']['metrics']['loc_view']['mAP'] - h['score']['mAP']:+.4f}* | | "
                         f"*{_j.load(open(hp))['MS-1280x736']['metrics']['loc_view']['mAP']:.4f}* | |")
        T['T25c'] = '\n'.join(lines)
    # resolution-adaptive global pass
    F_ = f'HAHI_{_T},trunc=drop,vote=0.6'
    rows = [('HAHI-S + F (global 640)', 'standard', f'{F_}_'),
            ('HAHI-S + F (global 1280 for every image)', 'standard', f'{F_},global_imgsz=1280_'),
            ('**HAHI-S + F, adaptive global (1280 if long side > 1920 px)**', 'standard', 'ADAPT_HAHI-S+F_'),
            ('HAHI-S route 0.1 + F, adaptive global', 'standard', 'ADAPT_HAHI-S_r0.1+F_'),
            ('HAHI-ext (thesis code, global 1280)', 'standard', 'HAHI_global_imgsz=1280,gcc=1.0_'),
            ('MS 640 + 1280x1280 engine + voting', 'standard', 'MS@640x1280_aspect=1.0,vote=0.6_'),
            ('MS 640 + 1536x1536 engine + voting', 'standard', 'MS@640x1536_aspect=1.0,vote=0.6_'),
            ('MS 640+1536 (dynamic) + voting', 'standard', 'MS@640x1536_vote=0.6_'),
            ('MS 640+1280+1536 (dynamic)', 'standard', 'MS@640x1280x1536')]
    head = ['Method', 'passes', 'GFLOPs', 'mAP', 'AP50', 'APs', 'APm', 'APl']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, prot, r in rows:
        if not _exists('dota_val', 'v8s_dota', prot, r):
            continue
        a = metrics('dota_val', 'v8s_dota', prot, r)['ALL']
        lines.append(f"| {label} | {a['n_inf']:.2f} | {a['gflops']:.0f} | {a['mAP']:.4f} | {a['AP50']:.4f} | {a['APs']:.4f} | "
                     f"{a['APm']:.4f} | {a['APl']:.4f} |")
    T['T25d'] = '\n'.join(lines)
    bp = os.path.join(RUN_ROOT, 'dota_val', 'v8s_dota', 'standard', 'bootstrap_ALL_thr0_ref-ADAPT_HAHI-S+F_.json')
    vp = os.path.join(RUN_ROOT, 'vid_holdout', 'v8s', 'standard_explore', 'adaptive_summary.json')
    lines = ['| Data | Adaptive method | Compared with | ΔmAP [95% CI] | P(Δ≤0) | ΔAPs [95% CI] |', '|---|---|---|---|---|---|']
    f = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    if os.path.exists(bp):
        B = _j.load(open(bp))
        for o, lab in [(f'{F_}_', 'HAHI-S + F (global 640)'), (f'{F_},global_imgsz=1280_', 'HAHI-S + F (global 1280 always)'),
                       ('HAHI_global_imgsz=1280,gcc=1.0_', 'HAHI-ext (thesis code)'),
                       ('MS@640x1280_aspect=1.0,vote=0.6_', 'MS 640 + 1280x1280 engine + voting'),
                       ('MS@640x1536_aspect=1.0,vote=0.6_', 'MS 640 + 1536x1536 engine + voting'),
                       ('MS@640x1536_vote=0.6_', 'MS 640+1536 (dynamic) + voting'), ('MS@640x1280x1536', 'MS 3-scale (dynamic)')]:
            if o in B:
                r = B[o]
                lines.append(f"| DOTA | HAHI-S + F, adaptive | {lab} | {f(r['mAP'])} | {r['mAP']['p_le_0']:.3f} | {f(r['APs'])} |")
    if os.path.exists(vp):
        B = _j.load(open(vp))['boot']
        for key, r in B.items():
            a, o = key.split('|')
            lines.append(f"| VID (seq. bootstrap) | {a.replace('ADAPT ', '').replace('+F', ' + F')}, adaptive | {o} | {f(r['mAP'])} | "
                         f"{r['mAP']['p_le_0']:.3f} | {f(r['APs'])} |")
    T['T25e'] = '\n'.join(lines)
    if os.path.exists(vp):
        import pickle as _p
        from .data import load_dataset
        from .evaluate import subset_indices
        recs = load_dataset('vid_holdout')[0]
        R = os.path.join(RUN_ROOT, 'vid_holdout', 'v8s')
        runs = [('HAHI-S + F (global 640)', 'standard', f'{F_}_'),
                ('HAHI-S + F (global 1280 always)', 'standard_explore', f'{F_},global_imgsz=1280_'),
                ('**HAHI-S + F, adaptive global**', 'standard_explore', 'ADAPT_HAHI-S+F_'),
                ('HAHI-S route 0.1 + F (global 640)', 'standard', f'{F_},route_ratio=0.1_'),
                ('HAHI-S route 0.1 + F, adaptive global', 'standard_explore', 'ADAPT_HAHI-S_r0.1+F_'),
                ('MS-tiled 1136 + F', 'standard', 'MSTILE@1136_trunc=drop,vote=0.6_'),
                ('MS 640+1280 (dynamic) + voting', 'standard', 'MS@640x1280_vote=0.6_'),
                ('MS 640 + 1280x736 engine + voting', 'standard', 'MS@640x1280_aspect=1.7778,vote=0.6_'),
                ('MS 640+1536 (dynamic) + voting', 'standard', 'MS@640x1536_iou=0.7,vote=0.7_'),
                ('MS 640+1280+1536 (dynamic) + voting', 'standard', 'MS@640x1280x1536_vote=0.6_')]
        head = ['Method (VID hold-out, v8s)', 'GFLOPs', 'mAP (956 frames)', 'mAP ≤ 1920 px (772)', 'mAP > 1920 px (184)']
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for label, prot, r in runs:
            rd = os.path.join(R, prot, r)
            bt = os.path.join(rd, 'boot_tables_ALL_thr0.pkl')
            if not os.path.exists(os.path.join(rd, 'dets.pkl')):
                continue
            if not os.path.exists(bt):
                from .stats import tables_for
                tables_for(rd, recs, 'ALL', 0.0, 10)
            ta = _p.load(open(bt, 'rb'))[0]
            a = metrics('vid_holdout', 'v8s', prot, r)['ALL']
            cells = [label, f"{a['gflops']:.0f}", f"{a['mAP']:.4f}"]
            for sub in ('LE1920', 'GT1920'):
                w = _np_zeros(len(ta.img_pos))
                for i in subset_indices(recs, range(len(recs)), sub):
                    w[ta.img_pos[i]] = 1.0
                cells.append(f'{ta.ap(w):.4f}')
            lines.append('| ' + ' | '.join(cells) + ' |')
        T['T25f'] = '\n'.join(lines)
    return T


def main():
    T = {}
    T['T11'] = md_table('vd_val', 'v8s', 'standard', ORACLE, cols=COLS, vram=False)
    T['T12'] = md_table('vd_val', 'v8s', 'standard', [r for r in NMS if _exists('vd_val', 'v8s', 'standard', r[1])],
                        cols=COLS, vram=False, best=())
    T['T13a'] = gating_table('vd_val', 'v8s', {
        'static': [('Global 640', 'global@640'), ('Global 1280', 'global@1280'), ('MS 640+1280', 'MS@640x1280'),
                   ('MS 640+1280+1536', 'MS@640x1280x1536'), ('HAHI', 'HAHI')],
        'gated': [('640 → +1280 (gate: #small dets)', 'MS@640x1280|n_small', [0.7, 0.8, 0.9]),
                  ('640 → +1280 (gate: GT oracle)', 'MS@640x1280|ORACLE_gt_small', [0.8, 0.9]),
                  ('640 → +1280+1536 (gate: #small dets)', 'MS@640x1280x1536|n_small', [0.8, 0.9])]})
    T['T13b'] = gating_table('dota_val', 'v8s_dota', {
        'static': [('Global 640', 'global@640'), ('Global 1280', 'global@1280'), ('MS 640+1280', 'MS@640x1280'),
                   ('MS 640+1536', 'MS@640x1536'), ('HAHI-ext', 'HAHI_global_imgsz=1280,gcc=1.0_')],
        'gated': [('640 → +1280 (gate: image size)', 'MS@640x1280|pixels', [0.7, 0.8]),
                  ('640 → +1536 (gate: image size)', 'MS@640x1536|pixels', [0.6, 0.7, 0.8]),
                  ('640 → +1536 (gate: #small dets)', 'MS@640x1536|n_small', [0.7, 0.8])]})
    T['T14a'] = fixed_table('vd_testdev', 'v8s', FIXED)
    T['T14b'] = '\n\n'.join([
        'HAHI(논문 기본값) 기준:\n\n' + boot_table('vd_testdev', 'v8s', 'HAHI',
                                              [('MS-tiled 960', 'MSTILE@960'), ('MS-tiled 1280', 'MSTILE@1280')]),
        'HAHI(val 튜닝판) 기준:\n\n' + boot_table('vd_testdev', 'v8s', 'HAHI_gcc=1.0,route_ratio=0.1_',
                                             [('MS-tiled 1280', 'MSTILE@1280'),
                                              ('MS-tiled 1280, overlap 0.15', 'MSTILE@1280_overlap=0.15_'),
                                              ('MS-tiled 1536', 'MSTILE@1536')])])
    T['T14c'] = fixed_table('vd_val', 'v8s', FIXED, latency=False)
    T['T14d'] = fixed_table('dota_val', 'v8s_dota', FIXED, latency=False)
    T['T15a'] = simplify_steps_table()
    T['T15b'] = simplify_table()
    bs = f'HAHI_{_T}_'
    bt = f'HAHI_{_T},route_ratio=0.1_'
    T['T15c'] = '\n\n'.join([
        'HAHI-S 기준:\n\n' + boot_table('vd_testdev', 'v8s', bs, [('HAHI (thesis code)', 'HAHI'),
                                                              ('MS-tiled 960', 'MSTILE@960'),
                                                              ('MS-tiled 1280', 'MSTILE@1280')]),
        'HAHI-S, route 0.1 기준:\n\n' + boot_table('vd_testdev', 'v8s', bt,
                                                [('HAHI val-tuned (thesis code)', 'HAHI_gcc=1.0,route_ratio=0.1_'),
                                                 ('MS-tiled 1280', 'MSTILE@1280'),
                                                 ('MS-tiled 1280, overlap 0.15', 'MSTILE@1280_overlap=0.15_'),
                                                 ('MS-tiled 1536', 'MSTILE@1536')])])
    T['T16a'] = static_engine_table()
    T['T16b'] = '\n\n'.join([
        'HAHI-S 기준:\n\n' + boot_table('vd_testdev', 'v8s', bs, [('MS 640 + 1280x736 engine', 'MS@640x1280_aspect=1.7778_')]),
        'HAHI-S, route 0.1 기준:\n\n' + boot_table('vd_testdev', 'v8s', bt,
                                                [('MS 640 + 1536x864 engine', 'MS@640x1536_aspect=1.7778_'),
                                                 ('MS 640 + 1280x1280 engine', 'MS@640x1280_aspect=1.0_')])])
    T.update(improve_tables())
    T.update(fair_tables())
    T['T19'] = aspect_table()
    T['T20'] = single_engine_table()
    T['T21'] = plugin_table()
    T.update(holdout_tables())
    T.update(trt_tables())
    T.update(overhead_tables())
    T.update(review_tables())
    fill_report(T)
    os.makedirs(FIG, exist_ok=True)
    fig_fixed(os.path.join(FIG, 'fixed_input.png'))
    fig_gating(os.path.join(FIG, 'gating.png'))
    for k, v in T.items():
        print(f'## {k}\n{v}\n')


if __name__ == '__main__':
    main()
