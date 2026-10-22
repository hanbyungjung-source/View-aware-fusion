"""Supporting Information for the manuscript (paper/supplementary_en.html).

Tables are taken from the auto-generated tables of REPORT.md (same numbers as the runs) and from the run directories.
AP-type values with four decimals are converted to percent with two decimals, the unit used in the manuscript.

  python -m hahi_bench.paper_si
"""
import html
import json
import os
import re

import numpy as np

from .make_tables import _exists
from .run_eval import RUN_ROOT
from .tables import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'paper', 'supplementary_en.html')
REPORT = open(os.path.join(HERE, 'REPORT.md'), encoding='utf-8').read()
_T = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv'

NUM4 = re.compile(r'(?<![\d.])([+\-\u2212]?)0\.(\d{4})(?!\d)')
HEAD = {'mAP': 'AP', 'ΔmAP': 'ΔAP', '#Inf': 'Passes', 'passes': 'Passes', 'mAP (conf>=0.3)': 'AP (score ≥ 0.3)',
        'mAP (thesis, @100 dets)': 'AP (thesis protocol)'}
FULL = {'DAHI+Packing-thesis': '"DAHI"+Packing (thesis code)', 'DAHI-thesis': '"DAHI" grid (thesis code)',
        'Packing-thesis': '"Packing" (thesis code)', 'UC2x2': 'UC 2×2', 'HAHI': 'HAHI (original code)',
        'SBSI+Packing': 'SBSI+Packing (thesis code)', 'DAHI': 'DAHI (paper default)', 'DAHI_n_roi=2_': 'DAHI, N_ROI = 2'}


def pct(cell):
    def f(m):
        v = float(m.group(1).replace('\u2212', '-') + '0.' + m.group(2)) * 100
        return f'{v:+.2f}' if m.group(1) in ('+', '-', '\u2212') else f'{v:.2f}'
    return NUM4.sub(f, cell)


def md_blocks(key):
    """All markdown tables inside a REPORT block, each with the last non-table line before it as its label."""
    m = re.search(rf'<!-- TABLE:{key} -->\n(.*?)\n<!-- /TABLE:{key} -->', REPORT, re.S)
    if not m:
        raise KeyError(key)
    split = lambda r: [c.strip() for c in r.strip().strip('|').split('|')]
    blocks, label, cur = [], '', None
    for r in m.group(1).strip().split('\n'):
        if r.startswith('|'):
            if cur is None:
                cur = [label, []]
                blocks.append(cur)
            cur[1].append(r)
        else:
            label = r.strip() or label
            cur = None
    out = []
    for lab, rows in blocks:
        if len(rows) < 2:
            raise KeyError(key)
        out.append((lab, [HEAD.get(h, h.replace('mAP', 'AP')) for h in split(rows[0])], [split(r) for r in rows[2:]]))
    return out


def md_table(key):
    blocks = md_blocks(key)
    if len(blocks) != 1:
        raise ValueError(f'{key} holds {len(blocks)} tables; use md_blocks')
    return blocks[0][1], blocks[0][2]


def to_html(head, body, convert=True, rename=None):
    def cell(c):
        c = pct(c) if convert == True else c
        if convert == 'num3':
            c = re.sub(r'(?<![\d.])0\.(\d{3})(?!\d)', lambda m: f'{int(m.group(1)) / 10:.1f}', c)
        c = FULL.get(c, c)
        if rename:
            for a, b in rename.items():
                c = c.replace(a, b)
        c = c[:-1] if c.endswith('_') else c
        bold = len(c) > 4 and c.startswith('**') and c.endswith('**')
        c = html.escape(c[2:-2] if bold else c).replace('&quot;', '"')
        return f'<b>{c}</b>' if bold else c
    out = ['<table class="bt"><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in head) + '</tr></thead><tbody>']
    for r in body:
        out.append('<tr>' + ''.join(f'<td>{cell(c)}</td>' for c in r) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def table(cap, key=None, head=None, body=None, note='', convert=True, rename=None):
    if key:
        head, body = md_table(key)
    s = f'<div class="tablewrap"><div class="tcap"><b>{cap.split(" ", 2)[0]} {cap.split(" ", 2)[1]}</b>&nbsp; {cap.split(" ", 2)[2]}</div>\n'
    s += to_html(head, body, convert, rename)
    if note:
        s += f'\n<div class="tnote">{note}</div>'
    return s + '</div>'


RENAME_RUNS = {'HAHI_weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv': 'HAHI-S',
               'trunc=drop,vote=0.6': 'F'}


def dt_rows(data, tag):
    specs = [('Dynamic Tiling, α = 0.0005', 'DYNTILE_alpha=0.0005_'), ('Dynamic Tiling, α = 0.001', 'DYNTILE_alpha=0.001_'),
             ('Dynamic Tiling, α = 0.002', 'DYNTILE_alpha=0.002_'), ('Dynamic Tiling, α = 0.005', 'DYNTILE_alpha=0.005_'),
             ('Dynamic Tiling, α = 0.01', 'DYNTILE_alpha=0.01_'), ('Dynamic Tiling, no FI', 'DYNTILE_fi=False_'),
             ('Dynamic Tiling, α = 0.002, NMS 0.5', 'DYNTILE_alpha=0.002,iou=0.5_'),
             ('Dynamic Tiling, α = 0.002, NMS 0.7', 'DYNTILE_alpha=0.002,iou=0.7_'),
             ('Dynamic Tiling, α = 0.002, planning score 0.1', 'DYNTILE_alpha=0.002,plan_conf=0.1_'),
             ('Dynamic Tiling, α = 0.002, planning score 0.5', 'DYNTILE_alpha=0.002,plan_conf=0.5_'),
             ('Dynamic Tiling, no large-small filter', 'DYNTILE_filt=False_'),
             ('Dynamic Tiling, no filter, NMS 0.7', 'DYNTILE_filt=False,iou=0.7_'),
             ('Dynamic Tiling, no filter, planning score 0.1 (selected)', 'DYNTILE_filt=False,plan_conf=0.1_'),
             ('Dynamic Tiling, no filter, NMS 0.7, planning score 0.1', 'DYNTILE_filt=False,iou=0.7,plan_conf=0.1_'),
             ('Dynamic Tiling, α = 0.002, NMS 0.7, planning score 0.1 (best with filter)', 'DYNTILE_alpha=0.002,iou=0.7,plan_conf=0.1_'),
             ('Dynamic Tiling, α = 0.002 + voting', 'DYNTILE_alpha=0.002,vote=0.6_'),
             ('Dynamic Tiling, no filter + voting', 'DYNTILE_filt=False,vote=0.6_'),
             ('Dynamic Tiling, selected + voting', 'DYNTILE_filt=False,plan_conf=0.1,vote=0.6_'),
             ('3×2 tiles + FI, large-small filter (no dynamic patches)', 'DYNTILE_dynamic=False,alpha=0.002_'),
             ('3×2 tiles + FI, NMS', 'DYNTILE_dynamic=False,filt=False_'),
             ('3×2 tiles + FI + TAF', 'DYNTILE_trunc=drop,filt=False_'),
             ('3×2 tiles + FI + TAF + voting (+ F)', 'DYNTILE_trunc=drop,filt=False,vote=0.6_')]
    body = []
    for lab, r in specs:
        if _exists(data, tag, 'standard', r):
            a = metrics(data, tag, 'standard', r)['ALL']
            body.append([lab, f"{a['n_inf']:.2f}", f"{a['gflops']:.0f}", f"{100 * a['mAP']:.2f}", f"{100 * a['AP50']:.2f}",
                         f"{100 * a['AP75']:.2f}", f"{100 * a['APs']:.2f}"])
    return ['Configuration', 'Passes', 'GFLOPs', 'AP', 'AP50', 'AP75', 'APS'], body


def frcnn_rows(data='vd_testdev'):
    rows = [('Global 640', 'global@640', None), ('Global 1280', 'global@1280', None), ('MS 640+1280', 'MS@640x1280', 'MS@640x1280_vote=0.6_'),
            ('UC 2×2', 'UC2x2', 'UC2x2_trunc=drop,vote=0.6_'),
            ('SAHI 512', 'SAHI@512_post=NMS,metric=IOU_', 'SAHI@512_post=NMS,metric=IOU,trunc=drop,vote=0.6_'),
            ('DAHI (CMA-NMS → F)', 'DAHI', 'DAHI_trunc=drop,vote=0.6_'),
            ('PAN', 'PAN_bandwidth=256_', 'PAN_bandwidth=256,trunc=drop,vote=0.6_'),
            ('MS-tiled 1280', 'MSTILE@1280', 'MSTILE@1280_trunc=drop,vote=0.6_'),
            ('HAHI-S', f'HAHI_{_T}_', f'HAHI_{_T},trunc=drop,vote=0.6_')]
    body = []
    for lab, a, b in rows:
        if not _exists(data, 'frcnn', 'standard', a):
            continue
        x = metrics(data, 'frcnn', 'standard', a)['ALL']
        y = metrics(data, 'frcnn', 'standard', b)['ALL'] if b and _exists(data, 'frcnn', 'standard', b) else None
        body.append([lab, f"{x['n_inf']:.2f}", f"{x['gflops']:.0f}", f"{100 * x['mAP']:.2f}", f"{100 * x['APs']:.2f}",
                     f"{100 * y['mAP']:.2f}" if y else '—', f"{100 * (y['mAP'] - x['mAP']):+.2f}" if y else '—',
                     f"{100 * (y['AP75'] - x['AP75']):+.2f}" if y else '—'])
    return ['Method', 'Passes', 'GFLOPs', 'AP', 'APS', 'AP + F (or + voting)', 'ΔAP', 'ΔAP75'], body


TESTS = {
    # key: (label, replicates, function of (point estimate, replicate deltas))
    'perc': ('percentile bootstrap, (k+1)/(B+1)', lambda dh, ds: (int((ds <= 0).sum()) + 1) / (len(ds) + 1)),
    'cent': ('null-centred bootstrap, (1+#{Δ*−Δ̂ ≥ Δ̂})/(B+1)',
             lambda dh, ds: (int((ds - dh >= dh).sum()) + 1) / (len(ds) + 1)),
    'reg': ('as originally specified, P(Δ* ≤ 0) = k/B', lambda dh, ds: int((ds <= 0).sum()) / len(ds)),
}


def holdout_tests(family):
    """Every confirmatory pair of the hold-out evaluation with Δ, 95% percentile CI (10,000 replicates) and the p-values of
    the three tests in TESTS for B = 10,000 and B = 1,000, Holm-adjusted within each dataset (and family).
    Replicates are read from the cached sequence-level bootstrap files written by stats.py / boot_gpu.py."""
    from .holdout import PREREG, safe
    from .paper_figs import _holm
    from .run_eval import RUN_ROOT
    M = PREREG['methods']
    pairs = PREREG['primary_families'][family]['pairs']
    settings = [('vid_holdout', 'v8s', 'VID · YOLOv8s'), ('vid_holdout', 'v8n', 'VID · YOLOv8n'),
                ('vid_holdout', 'y11s', 'VID · YOLO11s'), ('uavdt_test', 'v8s', 'UAVDT · YOLOv8s')]

    def reps(d, t, run, n):
        return np.load(os.path.join(RUN_ROOT, d, t, 'standard', run, f'boot_reps_ALL_thr0_n{n}_s0_seq.npy'))[:, 0]
    rows = []
    for d, t, lab in settings:
        for a, b in pairs:
            r = {'data': d, 'setting': lab, 'pair': (a.replace(' + F', '') if family == 'F1_plugin' else f'{a} vs {b}').replace('PAN', 'PAN*')}
            for n in (10000, 1000):
                x = reps(d, t, safe(M[a]), n) - reps(d, t, safe(M[b]), n)
                dh, ds = x[0], x[1:]
                r[n] = {'d': dh, 'lo': float(np.percentile(ds, 2.5)), 'hi': float(np.percentile(ds, 97.5)),
                        'p': {k: f(dh, ds) for k, (_, f) in TESTS.items()}}
            rows.append(r)
    for n in (10000, 1000):
        for key in TESTS:
            for d in ('vid_holdout', 'uavdt_test'):
                sub = [r for r in rows if r['data'] == d]
                for r, adj in zip(sub, _holm([r[n]['p'][key] for r in sub])):
                    r[n].setdefault('adj', {})[key] = adj
    return rows


def holdout_p_table(family):
    """Table rows: Δ [CI] (10,000 replicates); percentile and null-centred p (10,000); originally specified analysis (1,000)."""
    rows = holdout_tests(family)
    head = ['Setting', 'Comparison', 'Δ AP [95% CI]', 'p, percentile', 'Holm', 'p, null-centred', 'Holm',
            'P(Δ*≤0), B = 1,000 (original)', 'Holm']
    fmt = lambda v, sig: f'**{v:.4f}**' if sig else f'{v:.4f}'
    body = []
    for r in rows:
        x, o = r[10000], r[1000]
        cells = [r['setting'], r['pair'], f"{100 * x['d']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"]
        for key in ('perc', 'cent'):
            cells += [f"{x['p'][key]:.4f}", fmt(x['adj'][key], x['adj'][key] < 0.05)]
        cells += [f"{o['p']['reg']:.3f}", fmt(o['adj']['reg'], o['adj']['reg'] < 0.05)]
        body.append(cells)
    n_sig = {key: sum(1 for r in rows if r[10000]['adj'][key] < 0.05) for key in ('perc', 'cent')}
    n_sig['reg1000'] = sum(1 for r in rows if r[1000]['adj']['reg'] < 0.05)
    return head, body, n_sig, len(rows)


def holdout_count_table():
    """How the number of Holm-significant F1 comparisons depends on the test and on B (decomposition of 43 -> 39)."""
    rows = holdout_tests('F1_plugin')
    head = ['p-value', 'B = 1,000', 'B = 10,000']
    body = []
    for key, (lab, _) in [('reg', TESTS['reg']), ('perc', TESTS['perc']), ('cent', TESTS['cent'])]:
        cells = [lab]
        for n in (1000, 10000):
            tot = sum(1 for r in rows if r[n]['adj'][key] < 0.05)
            vid = sum(1 for r in rows if r['data'] == 'vid_holdout' and r[n]['adj'][key] < 0.05)
            cells.append(f'{tot} of 44 (VID {vid}/33, UAVDT {tot - vid}/11)')
        body.append(cells)
    return head, body


def holdout_history_table():
    """Every analysis of family F1 that has been reported, in order, with the settings that produced its count."""
    from .paper_figs import _holm
    rows = holdout_tests('F1_plugin')

    def count(n, key):
        per = sum(1 for r in rows if r[n]['adj'][key] < 0.05)
        pooled = int((_holm([r[n]['p'][key] for r in rows]) < 0.05).sum())
        return per, pooled
    fam = 'per dataset (VID 33, UAVDT 11)'
    spec = [('First report (originally specified analysis)', 1000, 'reg', 'k/B (0 reported as 0.001)'),
            ('First revision, column labelled "registered" in error', 1000, 'perc', '(k+1)/(B+1)'),
            ('First revision', 10000, 'perc', '(k+1)/(B+1)'),
            ('**Current primary analysis (percentile inversion)**', 10000, 'perc', '(k+1)/(B+1)'),
            ('Current sensitivity analysis (null-centred)', 10000, 'cent', '(1 + #{Δ*_b − Δ̂ ≥ Δ̂})/(B+1)'),
            ('Current: originally specified analysis, reproduced', 1000, 'reg', 'k/B')]
    head = ['Analysis', 'B', 'p-value', 'Holm family', 'Significant of 44', 'With one Holm family of 44']
    body = []
    for lab, n, key, f in spec:
        per, pooled = count(n, key)
        body.append([lab, f'{n:,}', f, fam, str(per), str(pooled)])
    return head, body


def taf_loss_table():
    from .run_eval import RUN_ROOT
    head = ['Data', 'Objects', 'Lost to TAF', '<16 px', '16–32 px', '32–96 px', '≥96 px',
            'width ≤1400', '1401–1920', '>1920', 'AP after F', 'AP restored']
    body = []
    for d, lab in (('vd_val', 'VisDrone val'), ('vd_testdev', 'VisDrone test-dev'), ('vid_holdout', 'VisDrone-VID')):
        p = os.path.join(RUN_ROOT, d, 'v8s', 'views', 'taf_loss.json')
        if not os.path.exists(p):
            continue
        x = json.load(open(p))['thr0.001']
        f = lambda lg: f'{lg[0]} ({100 * lg[0] / lg[1]:.2f}%)' if lg[1] else '—'
        body.append([lab, f"{x['n_gt']:,}", f"{x['n_lost']} ({100 * x['n_lost'] / x['n_gt']:.2f}%)"]
                    + [f(v) for v in x['size'].values()] + [f(v) for v in x['width'].values()]
                    + [f"{100 * x['AP_fused']:.2f}", f"{100 * x['AP_restored']:.2f}"])
    return head, body


TAF_DELTA = [('0', 0), ('0.4', 2), (None, 5), ('2', 10), ('4', 20)]
TAF_METHODS = [('HAHI-S + F', 'HAHI_weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,'
                'nms_impl=tv,trunc=drop,vote=0.6_'), ('SAHI 512 + F', 'SAHI@512_post=NMS,metric=IOU,trunc=drop,vote=0.6_'),
               ('MS-tiled 1280 + F', 'MSTILE@1280_trunc=drop,vote=0.6_'), ('UC 2×2 + F', 'UC2x2_trunc=drop,vote=0.6_')]


def taf_delta_table():
    from .make_tables import _exists
    from .tables import metrics
    head = ['Method'] + [f'δ = {px} px' + (' (default)' if k is None else '') for k, px in TAF_DELTA]
    body = []
    for lab, run in TAF_METHODS:
        cells = [lab]
        for k, _ in TAF_DELTA:
            prot = 'standard' if k is None else f'standard_taf{k}'
            cells.append(f"{100 * metrics('vd_val', 'v8s', prot, run)['ALL']['mAP']:.2f}" if _exists('vd_val', 'v8s', prot, run)
                         else '—')
        body.append(cells)
    return head, body


def soft_table():
    from .make_tables import _exists
    from .tables import metrics
    T = TAF_METHODS[0][1].replace(',trunc=drop,vote=0.6_', '')
    rows = [('SAHI 512', 'SAHI@512_post=NMS,metric=IOU'), ('MS-tiled 1280', 'MSTILE@1280'), ('HAHI-S', T), ('UC 2×2', 'UC2x2')]
    head = ['Split', 'Method', 'original rule', 'voting', 'Soft-NMS', 'TAF + voting (+ F)', 'TAF + Soft-NMS']

    def cell(d, prot, run):
        if not run or not _exists(d, 'v8s', prot, run):
            return '—'
        m = metrics(d, 'v8s', prot, run)['ALL']
        return f"{100 * m['mAP']:.2f} / {100 * m['AP50']:.2f} / {100 * m['AP75']:.2f}"
    body = []
    for d, split in (('vd_val', 'val'), ('vd_testdev', 'test-dev')):
        for lab, base in rows:
            if base.startswith('HAHI') or base.endswith('IOU'):   # specs whose name already carries options
                orig = base + '_'
                vote_only = None if base.startswith('HAHI') else base + ',vote=0.6_'
                fplug = base + ',trunc=drop,vote=0.6_'
            else:
                orig, vote_only, fplug = base, base + '_vote=0.6_', base + '_trunc=drop,vote=0.6_'
            body.append([split, lab, cell(d, 'standard', orig), cell(d, 'standard', vote_only),
                         cell(d, 'standard_fzsoft', vote_only), cell(d, 'standard', fplug),
                         cell(d, 'standard_fzsoft', fplug)])
    return head, body


def main():
    css = open(os.path.join(os.path.dirname(OUT), 'paper.css'), encoding='utf-8').read()
    P = []
    add = P.append
    add('<div class="article-type">Supporting Information</div>')
    add('<h1 class="title">View-aware fusion for crop-based small-object detection in high-resolution drone imagery: '
        'Supporting Information</h1><div class="authors">Byung Jung Han</div>')
    add('<p class="small">All AP values are in percent (COCO AP over IoU 0.50:0.95, up to 500 detections per image; 2,000 for '
        'DOTA). "Passes" are detector passes per image, one per view or canvas whatever its input size. Δ values are in AP '
        'points; brackets give 95% paired bootstrap intervals (1,000 resamples, and 10,000 for the confirmatory hold-out tests; '
        'whole sequences for video data). Method names follow the manuscript; PAN* is the crop-planning variant of PAN; '
        '"thesis code" denotes the baselines as implemented in the original HAHI code [12].</p>')

    add('<h2>S1 | Development results on VisDrone2019-DET</h2>')
    add(table('TABLE S1 Every configuration evaluated on val for selection (YOLOv8s). For each method, the configuration with '
              'the highest val AP was frozen and evaluated once on test-dev.', 'T3',
              rename={'_post=NMS,metric=IOU_': ' (NMS, IoU)', '_post=GREEDYNMM,conf=0.1_': ' (GREEDYNMM, score ≥ 0.1)',
                      '_post=GREEDYNMM,conf=0.3_': ' (GREEDYNMM, score ≥ 0.3)', 'SAHI@': 'SAHI ', 'MS@': 'MS ', 'global@': 'Global ',
                      '_bandwidth=': ', bandwidth ', '_n_roi=2_': ', N_ROI = 2', 'DAHI_size_min=320,size_max=640': 'DAHI, region size 320–640',
                      ',n_roi=2_': ', N_ROI = 2', ',conf_coarse=0.25_': ', coarse score 0.25', ',fuse_global=False_': ', no global fusion',
                      'x1280x1536': '+1280+1536', 'x1536': '+1536', 'x1280': '+1280', 'x960': '+960'}))
    add(table('TABLE S2 Full test-dev results (YOLOv8s), including the baselines of the original HAHI code. Latency: idle GPU, '
              'mean / 95th percentile per image including CPU stages. Peak VRAM: all crops of an image in one batch / batch size 1.', 'T4'))
    add(table('TABLE S3 Test-dev results with YOLOv8n (configurations transferred from YOLOv8s without retuning).', 'T9a'))
    add(table('TABLE S4 Test-dev results with YOLO11s (configurations transferred from YOLOv8s without retuning).', 'T9b'))
    add(table('TABLE S5 DOTA v1.0 val (458 images, mean width 2,398 px; YOLOv8s trained on DOTA train; VisDrone val '
              'configurations). Latency was not measured for methods that need dozens of passes per image (n/a).', 'T8'))
    (_, h1, b1), *boot = md_blocks('T21')
    s6 = table('TABLE S6 View-aware fusion as a plug-in. (a) AP before and after the plug-in on val (YOLOv8s) and test-dev '
               '(three detectors). (b) Paired bootstrap of the test-dev gains (fusion &minus; base, AP points, 95% CI).',
               head=h1, body=b1)
    bhead = ['Detector'] + boot[0][1]
    bbody = [[lab.split(',')[0].strip()] + r for lab, _, rows in boot for r in rows]
    s6 = s6.replace('</div>\n<table', '</div>\n<div class="tnote"><b>(a)</b></div>\n<table', 1)
    add(s6[:-len('</div>')] + '\n<div class="tnote"><b>(b)</b></div>\n' + to_html(bhead, bbody) + '</div>')

    add('<h2>S2 | Pre-specified hold-out evaluation</h2>')
    add('<p><b>Tests.</b> All analyses use the same sequence-level paired bootstrap replicates (seed 0); Δ*<sub>b</sub> is the '
        'AP difference (method + F minus method) in replicate b = 1, …, B, Δ̂ the observed difference and k the number of '
        'replicates with Δ*<sub>b</sub> ≤ 0. Three one-sided p-values are reported:</p>')
    add('<ul class="tight">'
        '<li><i>Originally specified</i>: p = k/B with B = 1,000, as fixed in the protocol. It is zero whenever no replicate '
        'is non-positive, so it cannot resolve small p-values after multiplicity correction.</li>'
        '<li><i>Percentile inversion</i> (primary): p = (k + 1)/(B + 1) with B = 10,000. Up to the added one, this is the '
        'smallest α for which the one-sided (1 − α) percentile lower bound, the α-quantile of Δ*, exceeds zero. A one-sided '
        'test at level α therefore corresponds to a two-sided (1 − 2α) percentile interval: unadjusted p &lt; 0.05 holds for '
        '43 F1 comparisons, whereas the two-sided 95% intervals of Table S8(c) exclude zero for 42. Holm-adjusted results need '
        'not agree with individual intervals.</li>'
        '<li><i>Null-centred</i> (sensitivity): the shifted replicates Δ*<sub>b</sub> − Δ̂ approximate the null distribution '
        'of Δ̂, and p = (1 + #{b : Δ*<sub>b</sub> − Δ̂ ≥ Δ̂})/(B + 1), i.e. the observed difference is the threshold for the '
        'shifted replicates, with the same add-one Monte Carlo correction (Hall and Wilson, Biometrics 1991; Davison and '
        'Hinkley 1997).</li></ul>')
    add('<p>All three are approximate tests. Holm\'s procedure controls the family-wise error rate only as well as the '
        'individual p-values are calibrated, which, with 24 VisDrone-VID and 20 UAVDT sequences, is uncertain; results near '
        '0.05 are therefore regarded as inconclusive. <b>Holm family.</b> The protocol prescribes Holm correction within each '
        'primary family; its amendment added UAVDT with the same methods, pairs and statistics as VisDrone-VID. Each family '
        'is therefore corrected separately per dataset: F1 has 33 tests on VisDrone-VID (11 pairs × 3 detectors) and 11 on '
        'UAVDT, F2 has 6 and 2. Table S8(a) also gives the counts for a single Holm family of all 44 F1 tests.</p>')
    add('<p><b>Change history.</b> Table S8(a) lists every analysis of F1 that has been reported. The first report (43 '
        'significant) used the originally specified statistic. The first revision replaced it by percentile inversion and '
        'labelled its 1,000-replicate column "registered" in error; that column was not the originally specified analysis. '
        'The present version reports the percentile test as primary (39), the null-centred test (43) and the originally '
        'specified analysis (43). The four UAVDT comparisons on which they disagree (SAHI, PAN*, MS-tiled 1136 and '
        'MS-tiled 1280), and both F2 comparisons on UAVDT, are inconclusive. Bold in Tables S8(c) and S9: significant at '
        '0.05 after correction.</p>')
    add(table('TABLE S7 AP of all 30 pre-specified methods on the hold-out data. PAN*: crop planning of PAN only.', 'T22a', rename={'PAN': 'PAN*'}))
    h_hist, b_hist = holdout_history_table()
    h0, b0 = holdout_count_table()
    h, b, ns, nt = holdout_p_table('F1_plugin')
    s8 = table(f'TABLE S8 Confirmatory family F1 (plug-in, method + F minus method). (a) Every reported analysis with the '
               f'settings that produced its count. (b) Number of comparisons that are significant after Holm correction '
               f'(per dataset), by p-value and number of replicates B. (c) All comparisons: Δ and two-sided 95% percentile '
               f'interval (B = 10,000); p-values of the primary percentile test and of the null-centred test (B = 10,000), '
               f'and of the originally specified analysis (B = 1,000). Significant after correction: {ns["perc"]} of {nt} '
               f'(percentile), {ns["cent"]} of {nt} (null-centred), {ns["reg1000"]} of {nt} (original).',
               head=h_hist, body=b_hist, convert=False)
    s8 = s8.replace('</div>\n<table', '</div>\n<div class="tnote"><b>(a)</b></div>\n<table', 1)
    add(s8[:-len('</div>')] + '\n<div class="tnote"><b>(b)</b></div>\n' + to_html(h0, b0, convert=False)
        + '\n<div class="tnote"><b>(c)</b></div>\n' + to_html(h, b, convert=False) + '</div>')
    h, b, ns, nt = holdout_p_table('F2_single_engine')
    add(table(f'TABLE S9 Confirmatory family F2 (single-engine efficiency of HAHI-S; directional hypothesis: HAHI-S + F more '
              f'accurate). Columns as in Table S8(c). Significant after Holm correction: {ns["perc"]} of {nt} (percentile), '
              f'{ns["cent"]} of {nt} (null-centred), {ns["reg1000"]} of {nt} (original).', head=h, body=b, convert=False))
    add(table('TABLE S10 Descriptive comparisons without a directional hypothesis (pre-specified; 1,000 replicates).', 'T22d'))

    add('<h2>S3 | Selected configurations</h2>')
    cfg = [['SAHI 512', 'official library v0.11.36; slice 512, overlap 0.2, full-image prediction; NMS IoU 0.5 (GREEDYNMM default rejected on val)'],
           ['DAHI', 'paper defaults: β = 0.001, 1,000 random regions (width 480–960, 16:9), RDE λ = 0.001, DACS IoS 0.25, C_TH = 0.01, CMA-NMS t = 0.65, θ = 10, t_MT = 0.9; N_ROI = 2 variant from the paper'],
           ['PAN*', '"fine-grained + packing" configuration (the learned patch-arrangement modules are not reproduced); mean-shift bandwidth 256, up to 4 patches per canvas'],
           ['UC 2×2', 'global 640 pass + four quadrants (thesis code), merging NMS 0.4'],
           ['MS a+b', 'full image at each size, class-wise NMS IoU 0.6 (0.7 for 640+1536 in the hold-out evaluation, selected on val); voting IoU equal to the NMS IoU'],
           ['MS-tiled S', 'global 640 pass + minimum number of 640×640 tiles covering the image resized to longer side S; NMS 0.6'],
           ['HAHI-S', 'uniform density weights, no snapping, no upscaling, no global calibration, merging NMS 0.6; window 512, canvas 960, routing ratio 0.3 (0.1 for "route 0.1")'],
           ['+ F (view-aware fusion)', 'TAF border distance δ (fixed rule, not tuned): 5 px at the detector input for UC, SAHI, DAHI, MS-tiled and Dynamic Tiling, i.e. 5 × crop side / detector input size in the view coordinates of each crop, and 5 canvas px for PAN* (640-px canvases = detector input); HAHI-S keeps the border test of the original HAHI code, 5 view px on 512-px dense windows (6.25 px at the input) and 3 canvas px on 960-px canvases (2 px at the input). The test is made in view coordinates before boxes are mapped to the image. Then class-wise NMS at the method\'s own merging threshold τn (UC 0.4, SAHI 0.5, PAN* 0.5, HAHI-S 0.6, MS-tiled 0.6, DAHI 0.65) and box voting with τv = 0.6, the only value selected on val (for HAHI-S). Sensitivity to δ in S10'],
           ['Dynamic Tiling', 'resize to 1920×1080, 3×2 tiles, edge threshold 2%, Tile Minimizer, full-image pass; selected on val: no large-small filter, NMS 0.6, planning score 0.1 (see S8)']]
    add(table('TABLE S11 Configurations used in the manuscript.', head=['Method', 'Configuration'], body=cfg, convert=False))
    add('<p><b>When δ was fixed.</b> The border distances were set when the plug-in was implemented for each method and '
        'were never changed. The DAHI and PAN* code (literature.py) is byte-identical to the version whose SHA-256 is '
        'recorded in the hold-out protocol (S5); the other files were later extended with switches that are off by default, '
        'and with these defaults the current code reproduces the stored detections exactly (checked on 15 val images for each of the four methods concerned: UC, SAHI, MS-tiled and HAHI-S).</p>')

    add('<h2>S4 | Audit of the original HAHI implementation</h2>')
    add('<p>The original implementation of HAHI [12] was ported and reproduced exactly (all twelve AP values of the thesis '
        'to four decimals). The audit found that the logistic upscaling described in the thesis was not implemented (a step '
        'function was used), that the 960-pixel canvases were resized to 640 by the detector, that an undocumented global '
        'confidence factor (×1.10) and a border decay (×0.8) were applied, and that the thesis baselines labelled "DAHI" and '
        '"Patch Packing" were not the published methods. The thesis evaluation used a redistributed split that mixes official '
        'test-dev images into training and thresholded outputs at a score of 0.3; all numbers in the manuscript use the official '
        'splits and the protocol of Section 4.4 instead. Tables S12 and S13 give the component ablation.</p>')
    add(table('TABLE S12 Component ablation of HAHI on val (YOLOv8s).', 'T6a'))
    add(table('TABLE S13 Component ablation of HAHI on test-dev (YOLOv8s).', 'T6b'))

    add('<h2>S5 | Hold-out protocol files</h2>')
    add('<p>The files were written on the author\'s computer and were not lodged with an external registry. The hashes '
        'identify the files; the creation times come from the local file system and cannot be verified independently.</p>')
    add('<p><b>Changes after the confirmatory evaluation.</b> The protocol files themselves are unchanged (Table S14). '
        'Everything below was decided after the hold-out results had been seen and is reported as such:</p>')
    add('<ul class="tight">'
        '<li><i>Statistical analysis.</i> The confirmatory test was changed from the originally specified statistic '
        '(k/B, 1,000 replicates) to percentile inversion with 10,000 replicates; a null-centred test was added and the '
        'original analysis is still reported (S2).</li>'
        '<li><i>Wording.</i> An earlier draft called the protocol pre-registered; it is now called pre-specified.</li>'
        '<li><i>Code.</i> Switches that are off by default were added: alternative merging rules (WBF, GREEDYNMM, '
        'Soft-NMS), a scale factor for the TAF border distance, GPU voting, the Dynamic Tiling baseline, the Faster R-CNN '
        'wrapper, GPU computation of bootstrap replicates and environment-variable dataset roots. The hold-out detections '
        'were produced before these changes; the current code no longer matches the hashes in the protocol, and the '
        'released detections allow every hold-out number to be recomputed.</li>'
        '<li><i>Exploratory analyses of the hold-out data</i>, all labelled as such in the manuscript: the split of F2 by '
        'frame width (Section 5.5), the resolution-adaptive global view (Section 6.3), and the recall lost by TAF on '
        'VisDrone-VID (S10), which required new inference of HAHI-S with per-view logging.</li></ul>')
    man = json.load(open(os.path.join(os.path.dirname(OUT), 'prereg_bundle', 'MANIFEST.json')))
    body = [[f, v['sha256'][:16] + '…', v['created_local'].replace('T', ' ')] for f, v in man['files'].items()]
    body += [[f'first output, {k}', '', v.replace('T', ' ')] for k, v in man['first_holdout_output_local'].items()]
    add(table('TABLE S14 Protocol files (SHA-256 prefix, local creation time, KST) and the time of the first hold-out '
              'output. The bundle is archived at <span class="todo">[DOI]</span>.', head=['File / event', 'SHA-256', 'Time'],
              body=body, convert=False))

    add('<h2>S6 | Detector scale response and per-size accuracy</h2>')
    add(table('TABLE S15 Recall (IoU ≥ 0.5, score ≥ 0.3) of YOLOv8s as a function of object size at the detector input (val, '
              'global passes at 640–1536).', 'T17a', convert=False))
    add(table('TABLE S16 AP (%) by native object size on val (columns: object side in pixels).', 'T17c', convert='num3',
              rename={'AP@[.5:.95] by native size (val)': 'Method'}))

    add('<h2>S7 | Fixed-input engines and aspect ratio</h2>')
    add('<p>With a single 16:9 high-resolution engine, MS must pad 4:3 frames, which shrinks them. The table splits the '
        'test-dev comparison by aspect ratio: on 16:9 frames HAHI-S + F and the 16:9-engine MS are tied for all three '
        'detectors, and the apparent advantage on the full set comes from the 4:3 frames.</p>')
    add(table('TABLE S17 HAHI-S + F versus MS with a 16:9 static engine, split by aspect ratio (test-dev).', 'T19'))

    add('<h2>S8 | Dynamic Tiling re-implementation</h2>')
    add('<p>Dynamic Tiling [15] has no public code; we followed the paper. Every image is resized to 1920&times;1080 and cut '
        'into 3&times;2 non-overlapping tiles. Tile predictions within 2% of the image width (vertical shared boundaries) '
        'or height (horizontal ones) are flagged as Near-Edge or Near-Corner. For a shared edge, a dynamic patch symmetric '
        'about the edge is formed; its extent along the edge covers the flagged predictions, its half-width covers them or '
        'is half a tile when they lie on one side only, and its size is at most one tile. For a shared corner, one '
        'tile-sized patch is centred on the corner. Flagged predictions are discarded and replaced by the patch predictions '
        'that lie near or across the boundary. The Tile Minimizer packs the patches into tile-sized composites. Full-image '
        'inference uses the large-small filter (θ = α·W·H), and the outputs are merged with class-wise NMS. Not specified '
        'in the paper and therefore selected on val with the rule used for every method: α, the NMS threshold, and the '
        'score above which a flagged prediction creates a patch (Table S18). The paper also fine-tunes its detector on '
        'tiles; as for every method here, we use the detector trained on full images.</p>')
    h, b = dt_rows('vd_val', 'v8s')
    add(table('TABLE S18 Dynamic Tiling on VisDrone val (YOLOv8s): selection of unspecified parameters and ablations. '
              'The selected configuration (highest val AP) disables the large-small filter; the best configuration with the '
              'filter is also evaluated on test-dev.', head=h, body=b, convert=False))
    for t, name in (('v8s', 'YOLOv8s'), ('v8n', 'YOLOv8n'), ('y11s', 'YOLO11s')):
        h, b = dt_rows('vd_testdev', t)
        add(table(f'TABLE S{19 + ("v8s", "v8n", "y11s").index(t)} Dynamic Tiling and view-aware fusion on VisDrone2019-DET '
                  f'test-dev ({name}).', head=h, body=b, convert=False))

    add('<h2>S9 | Faster R-CNN</h2>')
    add('<p>Faster R-CNN with a ResNet-50-FPN v2 backbone (torchvision, COCO weights) was fine-tuned on the VisDrone2019-DET '
        'training split with the longer image side resized to 640 pixels, horizontal flips, SGD (lr 0.005, batch 4, '
        'momentum 0.9, weight decay 10<sup>&minus;4</sup>), 1,000 warm-up iterations, and 12 epochs with ×0.1 steps after '
        'epochs 8 and 11. Anchor sizes were halved (16&ndash;256 pixels). At test time the longer side of each view is '
        'resized to the input size requested by the method, as for the YOLO detectors; inference uses FP16 autocast, '
        'torchvision\'s default 1,000 RPN proposals and its internal class-wise NMS (IoU 0.5). Configurations were transferred '
        'from YOLOv8s without retuning. On val, a single 640-pixel pass reached 20.98 AP.</p>')
    h, b = frcnn_rows()
    add(table('TABLE S22 Faster R-CNN on VisDrone2019-DET test-dev: every method with and without view-aware fusion '
              '(MS: with and without voting).', head=h, body=b, convert=False))

    add('<h2>S10 | Recall lost by TAF and sensitivity to the border distance</h2>')
    add('<p>TAF uses border contact as a proxy for truncation. Table S23 counts, for HAHI-S + F with YOLOv8s, the objects '
        'whose only correct candidate (IoU &ge; 0.5, same class, score &ge; 0.001) in any view was a border-touching local box '
        'removed by TAF, so that no remaining candidate and no fused detection matched them. "AP restored" adds, for each '
        'such object, the removed box with its own score; this repair is optimistic, because a real rule cannot tell which '
        'border boxes are correct. VisDrone-VID is part of the hold-out data; this analysis is exploratory.</p>')
    h, b = taf_loss_table()
    add(table('TABLE S23 Objects lost to TAF (HAHI-S + F, YOLOv8s), by object size (square root of the box area) and by '
              'image width.', head=h, body=b, convert=False))
    add('<p>Table S24 varies the border distance δ on val. δ is given at the detector input for tiles and crops and in '
        'native pixels for the dense windows of HAHI-S; the canvas distance of HAHI-S (3 px by default) was scaled by the '
        'same factor. δ = 0 removes only boxes whose side lies exactly on an interior border.</p>')
    h, b = taf_delta_table()
    add(table('TABLE S24 AP (%) on VisDrone val (YOLOv8s) for different border distances δ.', head=h, body=b, convert=False))
    add('<p>Table S25 adds Soft-NMS (Gaussian, σ = 0.5, score floor 0.001; Bodla et al., ICCV 2017) to the merging rules '
        'compared on identical detections (YOLOv8s), with and without TAF.</p>')
    h, b = soft_table()
    add(table('TABLE S25 Soft-NMS compared with voting on identical detections (YOLOv8s). AP / AP<sub>50</sub> / '
              'AP<sub>75</sub> in %.', head=h, body=b, convert=False))

    doc = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Supporting Information</title>'
           '<link rel="stylesheet" href="paper.css"></head><body><div class="page">' + '\n'.join(P) + '</div></body></html>')
    open(OUT, 'w', encoding='utf-8').write(doc)
    print('written', OUT, len(doc))


if __name__ == '__main__':
    main()
