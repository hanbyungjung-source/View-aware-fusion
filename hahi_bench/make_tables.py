"""Build every markdown table of REPORT.md directly from the stored runs (writes report_tables.md)."""
import json
import os

from .tables import md_table, metrics
from .run_eval import RUN_ROOT

OUT = os.path.join(os.path.dirname(__file__), 'report_tables.md')

MAIN = [
    ('Global 640 (1 pass)', 'global@640'),
    ('Global 1280 (1 pass)', 'global@1280'),
    ('MS 640+1280', 'MS@640x1280'),
    ('MS 640+1536', 'MS@640x1536'),
    ('MS 640+1280+1536', 'MS@640x1280x1536'),
    ('UC 2x2 (thesis)', 'UC2x2'),
    ('SAHI 512 (NMS)', 'SAHI@512_post=NMS,metric=IOU_'),
    ('DAHI (paper default)', 'DAHI'),
    ('DAHI (N_ROI=2)', 'DAHI_n_roi=2_'),
    ('PAN fine+packing', 'PAN_bandwidth=256_'),
    ('"DAHI" grid (thesis code)', 'DAHI-thesis'),
    ('"Packing" (thesis code)', 'Packing-thesis'),
    ('"DAHI"+Packing (thesis code)', 'DAHI+Packing-thesis'),
    ('SBSI+Packing (thesis)', 'SBSI+Packing'),
    ('**HAHI (thesis default)**', 'HAHI'),
    ('**HAHI (val-tuned)**', 'HAHI_gcc=1.0,route_ratio=0.1_'),
    ('HAHI-ext (global 1280, exploratory)', 'HAHI_global_imgsz=1280,gcc=1.0_'),
]


def _exists(data, tag, protocol, run):
    return os.path.exists(os.path.join(RUN_ROOT, data, tag, protocol, run, 'meta.json'))


_recs_cache = {}


def cost_only(data, tag, protocol, run):
    """Cost statistics (per subset) + peak VRAM from a timing run, without re-running COCO evaluation."""
    from .data import load_dataset
    from .evaluate import SUBSETS, cost_stats, subset_indices
    if data not in _recs_cache:
        _recs_cache[data] = load_dataset(data)[0]
    recs = _recs_cache[data]
    d = os.path.join(RUN_ROOT, data, tag, protocol, run)
    per = json.load(open(os.path.join(d, 'per_image.json')))
    meta = json.load(open(os.path.join(d, 'meta.json')))
    idx = sorted(p['i'] for p in per)
    out = {'peak_alloc_mb': meta['peak_alloc_mb']}
    for sub in SUBSETS:
        si = subset_indices(recs, idx, sub)
        if si:
            out[sub] = cost_stats(per, set(si))
    return out


def cost_table(data, tag, rows, subset='ALL', thr03=True):
    """Accuracy from the standard runs + latency/VRAM from the dedicated idle-GPU timing runs."""
    head = ['Method', 'mAP', 'AP50', 'APs', 'APm', 'APl']
    if thr03:
        head.append('mAP (conf>=0.3)')
    head += ['#Inf', 'GFLOPs', 'Latency ms (mean / p95)', 'Peak VRAM MB (batched / batch-1)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    acc = {r: metrics(data, tag, 'standard', r) for _, r in rows if _exists(data, tag, 'standard', r)}
    best = {k: max(a[subset][k] for a in acc.values()) for k in ['mAP', 'AP50', 'APs', 'APm', 'APl']}
    for label, r in rows:
        if r not in acc:
            continue
        a = acc[r][subset]
        cells = [label]
        for k in ['mAP', 'AP50', 'APs', 'APm', 'APl']:
            s = f'{a[k]:.4f}'
            cells.append(f'**{s}**' if abs(a[k] - best[k]) < 1e-12 else s)
        if thr03:
            cells.append(f"{metrics(data, tag, 'standard', r, thr=0.3)[subset]['mAP']:.4f}")
        cells += [f"{a['n_inf']:.2f}", f"{a['gflops']:.0f}"]
        if _exists(data, tag, 'standard_timing', r):
            t = cost_only(data, tag, 'standard_timing', r)
            lat = f"{t[subset]['total_ms']:.1f} / {t[subset]['total_p95']:.1f}"
            v = f"{t['peak_alloc_mb']:.0f}"
        else:
            lat, v = 'n/a', 'n/a'
        if _exists(data, tag, 'standard_b1_timing', r):
            v += f" / {json.load(open(os.path.join(RUN_ROOT, data, tag, 'standard_b1_timing', r, 'meta.json')))['peak_alloc_mb']:.0f}"
        cells += [lat, v]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def stage_table(data, tag, rows):
    head = ['Method', 'Global ms', 'Plan ms', 'Build ms', 'Local ms', 'Merge ms', 'Total ms', 'CPU-only ms']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for label, r in rows:
        if not _exists(data, tag, 'standard_timing', r):
            continue
        t = cost_only(data, tag, 'standard_timing', r)['ALL']
        cpu = 'n/a'
        if _exists(data, tag, 'standard_cpu_timing', r):
            cpu = f"{cost_only(data, tag, 'standard_cpu_timing', r)['ALL']['total_ms']:.0f}"
        lines.append('| ' + ' | '.join([label] + [f"{t[k]:.1f}" for k in ['global_ms', 'plan_ms', 'build_ms', 'local_ms',
                                                                          'merge_ms', 'total_ms']] + [cpu]) + ' |')
    return '\n'.join(lines)


def boot_table(data, tag, ref, rows, subset='ALL'):
    p = os.path.join(RUN_ROOT, data, tag, 'standard', f'bootstrap_{subset}_thr0_ref-{ref}.json')
    if not os.path.exists(p):
        return f'(bootstrap file missing: {p})'
    b = json.load(open(p))
    head = ['Compared with', 'ΔmAP [95% CI]', 'P(Δ≤0)', 'ΔAP50 [95% CI]', 'ΔAPs [95% CI]', 'P(ΔAPs≤0)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    f = lambda r: f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    for label, r in rows:
        if r not in b:
            continue
        x = b[r]
        lines.append(f"| {label} | {f(x['mAP'])} | {x['mAP']['p_le_0']:.3f} | {f(x['AP50'])} | {f(x['APs'])} | {x['APs']['p_le_0']:.3f} |")
    return '\n'.join(lines)


def fill_report(sections, path=os.path.join(os.path.dirname(__file__), 'REPORT.md')):
    """Replace <!-- TABLE:ID --> ... <!-- /TABLE:ID --> blocks (or bare <!-- TABLE:ID --> markers) in REPORT.md."""
    import re
    with open(path, encoding='utf-8') as f:
        text = f.read()
    for key, body in sections.items():
        block = f'<!-- TABLE:{key} -->\n{body}\n<!-- /TABLE:{key} -->'
        pat = re.compile(rf'<!-- TABLE:{re.escape(key)} -->(?:.*?<!-- /TABLE:{re.escape(key)} -->)?', re.S)
        text = pat.sub(lambda _: block, text)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    T = {}
    rows = [('UC(2x2)', 'UC2x2'), ('DAHI', 'DAHI-thesis'), ('Packing', 'Packing-thesis'),
            ('DAHI+Packing', 'DAHI+Packing-thesis'), ('SBSI+Packing', 'SBSI+Packing'), ('SBSI+DCRP (HAHI)', 'HAHI')]
    T['T1'] = md_table('rf_valid', 'orig_model', 'thesis', rows, subsets=('ALL', 'HR'),
                       cols=('mAP@100', 'AP50', 'APs', 'APm', 'APl', 'n_inf'), use_ignore=False, best=())
    T['T2'] = md_table('rf_valid', 'orig_model', 'standard', rows + [('Global 640', 'global@640')], subsets=('ALL', 'HR'),
                       cols=('mAP', 'AP50', 'APs', 'n_inf'), use_ignore=False, vram=False)
    val_rows = [(r, r) for r in sorted(os.listdir(os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'standard')))
                if _exists('vd_val', 'v8s', 'standard', r) and not r.startswith('HAHI_')
                and not r.startswith('Packing-thesis_')]
    T['T3'] = md_table('vd_val', 'v8s', 'standard', val_rows, cols=('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf', 'gflops'),
                       vram=False)
    T['T4'] = cost_table('vd_testdev', 'v8s', MAIN)
    T['T4b'] = cost_table('vd_testdev', 'v8s', MAIN, subset='FHD', thr03=False)
    T['T5'] = boot_table('vd_testdev', 'v8s', 'HAHI', MAIN)
    T['T5b'] = boot_table('vd_testdev', 'v8s', 'HAHI_gcc=1.0,route_ratio=0.1_', MAIN)
    T['T5c'] = boot_table('vd_testdev', 'v8s', 'HAHI_global_imgsz=1280,gcc=1.0_', MAIN)
    abl = [('HAHI (full, thesis default)', 'HAHI'),
           ('− SCE (= SBSI+Packing)', 'SBSI+Packing'),
           ('− packing (SBSI windows only)', 'HAHI_packing=False_'),
           ('− SBSI routing (Packing+SCE)', 'Packing-thesis_sce=True_'),
           ('− SBSI routing, − SCE (Packing)', 'Packing-thesis'),
           ('upscale: none', 'HAHI_upscale=none_'), ('upscale: logistic (thesis Eq.)', 'HAHI_upscale=logistic_'),
           ('weight: uniform', 'HAHI_weight=uniform_'), ('weight: (1−c)', 'HAHI_weight=conf_'),
           ('weight: √A', 'HAHI_weight=sqrt_area_'), ('weight: (1−c)·A (thesis Eq.)', 'HAHI_weight=conf_area_'),
           ('no centre-of-mass snap', 'HAHI_snap=False_'), ('no GCC (×1.10)', 'HAHI_gcc=1.0_'),
           ('no boundary decay', 'HAHI_decay=1.0_'), ('no GCC, no decay', 'HAHI_gcc=1.0,decay=1.0_'),
           ('canvas 640 (no down-scaling)', 'HAHI_canvas=640_')]
    for data, key in [('vd_val', 'T6a'), ('vd_testdev', 'T6b')]:
        T[key] = md_table(data, 'v8s', 'standard', [r for r in abl if _exists(data, 'v8s', 'standard', r[1])],
                          cols=('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf', 'gflops'), vram=False)
    sens = [(r, r) for r in sorted(os.listdir(os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'standard')))
            if any(r.startswith(f'HAHI_{p}=') for p in ['route_ratio', 'window', 'sigma_div', 'upscale_ratio', 'conf_filter'])
            or r in ('HAHI_gcc=1.0,route_ratio=0.1_', 'HAHI_gcc=1.0,route_ratio=0.2_', 'HAHI_global_imgsz=1280_',
                     'HAHI_global_imgsz=1280,gcc=1.0_')]
    T['T7'] = md_table('vd_val', 'v8s', 'standard', [('HAHI (default)', 'HAHI')] + sens,
                       cols=('mAP', 'APs', 'n_inf', 'gflops'), vram=False)
    dota = [x for x in MAIN if _exists('dota_val', 'v8s_dota', 'standard', x[1])]
    T['T8'] = cost_table('dota_val', 'v8s_dota', dota, thr03=False)
    T['T8b'] = cost_table('dota_val', 'v8s_dota', dota, subset='FHD', thr03=False)
    for tag, key in [('v8n', 'T9a'), ('y11s', 'T9b')]:
        rows_g = [x for x in MAIN if _exists('vd_testdev', tag, 'standard', x[1])]
        T[key] = md_table('vd_testdev', tag, 'standard', rows_g, cols=('mAP', 'AP50', 'APs', 'APm', 'APl', 'n_inf', 'gflops'),
                          vram=False) if rows_g else '(not run)'
    T['T10'] = stage_table('vd_testdev', 'v8s', MAIN)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(f'## {k}\n\n{v}' for k, v in T.items()) + '\n')
    fill_report(T)
    print('written', OUT, 'and filled REPORT.md')


if __name__ == '__main__':
    main()
