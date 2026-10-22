"""Figures for the IET Image Processing manuscript draft (written to ../paper/figures as PNG 300 dpi and PDF).

  python -m hahi_bench.paper_figs [--only fig2]

Fig. 1  method overview (schematic)
Fig. 2  real example (VisDrone val): crop layout of HAHI-S and a truncated local detection removed by TAF
Fig. 3  bottleneck decomposition on val (selection / magnification / fusion)
Fig. 4  accuracy-cost on VisDrone test-dev (YOLOv8s): every crop method with and without the plug-in
Fig. 5  pre-registered hold-out (F1): plug-in gain with 95% sequence-bootstrap CIs
Fig. 6  (a) gain vs passes over all 74 confirmatory comparisons, (b) error decomposition (TIDE-style oracles)
"""
import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .make_tables import _exists
from .run_eval import RUN_ROOT
from .tables import metrics

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')
W1, W2 = 3.35, 6.85          # IET single / double column width (inch)
_T = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv'
F = 'trunc=drop,vote=0.6'
C_BASE, C_PLUG, C_MS, C_BAD = '#7f7f7f', '#1f77b4', '#d62728', '#d62728'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.linewidth': 0.6, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'lines.linewidth': 1.0,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'savefig.dpi': 300, 'pdf.fonttype': 42})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', name, flush=True)


def M(data, tag, run, prot='standard'):
    return metrics(data, tag, prot, run)['ALL'] if _exists(data, tag, prot, run) else None


# ----------------------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#ffffff', ec='#333333', bold=False, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.7, color='#333333'))

    box(0.5, 11, 11, 8, 'High-resolution\nimage', fc='#f2f2f2')
    box(15, 19.5, 17, 7.5, 'Global pass\n(whole image, 640)', fc='#e8f0fb')
    box(15, 3, 17, 9.5, 'Crop planner\n(any method: tiles,\ndensity windows,\npacked canvases)', fc='#f2f2f2')
    box(36, 3, 13, 9.5, 'Local passes\n(crops /\ncanvases)', fc='#e8f0fb')
    arrow(11.8, 16.5, 14.6, 22.5)
    arrow(11.8, 13.5, 14.6, 8)
    arrow(23.5, 19.2, 23.5, 12.9)
    arrow(32.3, 7.8, 35.6, 7.8)
    # plug-in
    ax.add_patch(Rectangle((52, 1.0), 34, 28.0, fc='#fff7e6', ec='#ff7f0e', lw=0.9, ls='--'))
    ax.text(69, 27.0, 'View-aware fusion (training-free plug-in)', ha='center', va='center', fontsize=6.8,
            weight='bold', color='#b35900')
    box(54, 3, 30, 9.5, 'Truncation-aware filtering (TAF):\ndrop local boxes touching a crop\n'
        'border that lies inside the image', fc='#ffffff', fs=6.3)
    box(54, 16, 30, 7.8, 'Class-wise NMS (IoU 0.6)\n+ score-weighted box voting', fc='#ffffff', fs=6.3)
    arrow(49.3, 7.8, 53.6, 7.8)
    arrow(32.3, 21.5, 53.6, 21.5)
    arrow(69, 12.9, 69, 15.6)
    ax.text(43, 22.6, 'global boxes', fontsize=5.8, ha='center', color='#555555')
    ax.text(51.0, 4.2, 'local\nboxes', fontsize=5.6, ha='center', color='#555555')
    box(88.5, 15.5, 11.2, 8.8, 'Detections', fc='#f2f2f2', bold=True, fs=6.6)
    arrow(84.4, 19.9, 88.1, 19.9)
    save(fig, 'fig1_overview')


# ----------------------------------------------------------------------------------------------------------- Fig. 2
def fig2():
    import cv2
    from torchvision.ops import box_iou
    import torch
    from . import diag_views as DV
    from . import thesis_methods as TM
    from .data import load_dataset
    from .detector import Detector
    recs, names = load_dataset('vd_val')
    V = pickle.load(open(os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'views', 'HAHI-S.pkl'), 'rb'))
    best, cand = None, []
    for i, v in V.items():
        if recs[i].w < 1300:
            continue
        loc = (v['kind'] > 0) & v['edge'] & (v['s'] >= 0.5)
        if not loc.any():
            continue
        gt = recs[i].boxes.astype(np.float64)
        I = box_iou(torch.from_numpy(v['b'][loc]), torch.from_numpy(gt)).numpy() if len(gt) else np.zeros((loc.sum(), 0))
        if not I.size:
            continue
        side = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
        glob = v['kind'] == 0
        Ig = box_iou(torch.from_numpy(v['b'][glob & (v['s'] >= 0.3)]), torch.from_numpy(gt)).numpy() \
            if (glob & (v['s'] >= 0.3)).any() else np.zeros((0, len(gt)))
        for r, j in enumerate(I.argmax(1)):
            if 0.25 <= I[r, j] <= 0.6 and side[j] >= 40 and len(Ig) and Ig[:, j].max() >= 0.75:
                cand.append((side[j] * (1 + loc.sum() / 50), i, int(np.flatnonzero(loc)[r]), j))
    cand.sort(reverse=True)
    _, i, k, j = cand[0]
    r = recs[i]
    img = cv2.imread(r.path)
    h, w = img.shape[:2]
    det = Detector('hahi_bench/train_runs/v8s_official/weights/best.pt', max_det=1000)
    cfg = DV.CFG_S
    gb, gs, gc = det([img], conf=cfg.out_conf, imgsz=cfg.global_imgsz, max_det=cfg.max_det)[0]
    roi = [b + [s] for b, s in zip(gb.tolist(), gs.tolist()) if s >= cfg.conf_filter]
    regions, remaining = TM.sbsi_regions(roi, w, h, cfg)
    infos = TM.build_canvases(img, remaining, cfg)[1] if cfg.packing and remaining else []
    v = V[i]
    rgb = img[:, :, ::-1]
    fig = plt.figure(figsize=(W2, 2.75))
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
    ax.imshow(rgb)
    ax.axis('off')
    for x1, y1, x2, y2 in regions:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc='none', ec='#00b0ff', lw=1.0))
    for inf in infos:
        sw, sh = (inf['cx2'] - inf['cx1']) / inf['scale'], (inf['cy2'] - inf['cy1']) / inf['scale']
        ax.add_patch(Rectangle((inf['ox'], inf['oy']), sw, sh, fc='none', ec='#ffb000', lw=0.7, ls='--'))
    tr = (v['kind'] > 0) & v['edge'] & (v['s'] >= 0.5)
    for b in v['b'][tr]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc='none', ec='#ff1744', lw=0.8))
    g = r.boxes[j]
    pad = max(40, 0.6 * max(g[2] - g[0], g[3] - g[1]))
    zx1, zy1 = int(max(0, g[0] - pad)), int(max(0, g[1] - pad))
    zx2, zy2 = int(min(w, g[2] + pad)), int(min(h, g[3] + pad))
    ax.add_patch(Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, fc='none', ec='white', lw=1.2, ls=':'))
    ax.text(8, 8, '(a)', color='white', fontsize=8, weight='bold', va='top',
            bbox=dict(fc='black', ec='none', alpha=0.5, pad=1.5))
    az = fig.add_axes([0.655, 0.36, 0.345, 0.64])
    az.imshow(rgb[zy1:zy2, zx1:zx2], extent=(zx1, zx2, zy2, zy1))
    az.set_xticks([])
    az.set_yticks([])
    for x1, y1, x2, y2 in regions:
        az.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc='none', ec='#00b0ff', lw=1.6))
    for inf in infos:
        sw, sh = (inf['cx2'] - inf['cx1']) / inf['scale'], (inf['cy2'] - inf['cy1']) / inf['scale']
        az.add_patch(Rectangle((inf['ox'], inf['oy']), sw, sh, fc='none', ec='#ffb000', lw=1.0, ls='--'))
    tb = v['b'][k]
    glob = np.flatnonzero((v['kind'] == 0) & (v['s'] >= 0.3))
    Ig = box_iou(torch.from_numpy(v['b'][glob]), torch.from_numpy(g[None].astype(np.float64))).numpy()[:, 0]
    gbx = v['b'][glob[Ig.argmax()]]
    az.add_patch(Rectangle((gbx[0], gbx[1]), gbx[2] - gbx[0], gbx[3] - gbx[1], fc='none', ec='#00e676', lw=1.4))
    az.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fc='none', ec='white', lw=1.0, ls=(0, (2, 2))))
    az.add_patch(Rectangle((tb[0], tb[1]), tb[2] - tb[0], tb[3] - tb[1], fc='none', ec='#ff1744', lw=1.6))
    iou_t = box_iou(torch.from_numpy(tb[None]), torch.from_numpy(g[None].astype(np.float64))).item()
    iou_g = Ig.max()
    az.set_xlim(zx1, zx2)
    az.set_ylim(zy2, zy1)
    az.text(0.02, 0.98, '(b)', transform=az.transAxes, color='white', fontsize=8, weight='bold', va='top',
            bbox=dict(fc='black', ec='none', alpha=0.5, pad=1.5))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color='#00b0ff', lw=1.4, label='dense-window border'),
          Line2D([], [], color='#ffb000', lw=1.0, ls='--', label='packed-canvas source region'),
          Line2D([], [], color='#ff1744', lw=1.4, label=f'truncated local box (score {v["s"][k]:.2f}, IoU {iou_t:.2f})'),
          Line2D([], [], color='#00e676', lw=1.4, label=f'global-view box (score {v["s"][glob[Ig.argmax()]]:.2f}, IoU {iou_g:.2f})'),
          Line2D([], [], color='#555555', lw=1.0, ls=(0, (2, 2)), label='ground truth')]
    fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.65, 0.345), ncol=1, frameon=False, fontsize=5.8)
    save(fig, 'fig2_example')
    json.dump({'image': r.name, 'index': i, 'gt_class': names[int(r.cls[j])], 'iou_trunc': iou_t, 'iou_global': float(iou_g),
               'n_trunc_shown': int(tr.sum()), 'n_regions': len(regions), 'n_canvas_regions': len(infos)},
              open(os.path.join(OUT, 'fig2_example.json'), 'w'), indent=1)


# ----------------------------------------------------------------------------------------------------------- Fig. 3
def fig3():
    # val, YOLOv8s; values from REPORT sections 14.1, 15.3, 15.4, 15.6 (offline truncation oracle: log_trunc_oracle.txt)
    rows = [('Crop selection: GT-oracle crops\n(vs. predicted crops, HAHI)', 0.2794 - 0.2814, 'selection'),
            ('Crop selection: GT-oracle crops\n(vs. predicted crops, HAHI tuned)', 0.2862 - 0.2887, 'selection'),
            ('Magnification: scale-response\nallocation (best λ, vs. HAHI-S)', 0.2841 - 0.2834, 'magnification'),
            ('Fusion: truncation-aware\nfiltering (TAF, vs. HAHI-S)', 0.2863 - 0.2834, 'fusion'),
            ('Fusion: + box voting\n(vs. HAHI-S + TAF)', 0.2897 - 0.2863, 'fusion'),
            ('Truncation-aware retraining\nupper bound (vs. TAF)', 0.2835 - 0.2865, 'fusion')]
    col = {'selection': '#9e9e9e', 'magnification': '#9e9e9e', 'fusion': C_PLUG}
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] * 100 for r in rows], color=[col[r[2]] for r in rows], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.axvline(0, color='black', lw=0.6)
    for yy, r in zip(y, rows):
        d = r[1] * 100
        ax.text(d + (0.02 if d >= 0 else -0.02), yy, f'{d:+.2f}', va='center', ha='left' if d >= 0 else 'right',
                fontsize=6.2)
    ax.set_xlabel('ΔAP (points, VisDrone val)')
    ax.set_xlim(-0.45, 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_bottleneck')


# ----------------------------------------------------------------------------------------------------------- Fig. 4
CROP = [('UC 2×2', 'UC2x2', 'UC2x2_trunc=drop,vote=0.6_'),
        ('SAHI 512', 'SAHI@512_post=NMS,metric=IOU_', 'SAHI@512_post=NMS,metric=IOU,trunc=drop,vote=0.6_'),
        ('DAHI', 'DAHI', 'DAHI_trunc=drop,vote=0.6_'),
        ('DAHI, $N_{\\mathrm{ROI}}$ = 2', 'DAHI_n_roi=2_', 'DAHI_n_roi=2,trunc=drop,vote=0.6_'),
        ('PAN', 'PAN_bandwidth=256_', 'PAN_bandwidth=256,trunc=drop,vote=0.6_'),
        ('MS-tiled 960', 'MSTILE@960', 'MSTILE@960_trunc=drop,vote=0.6_'),
        ('MS-tiled 1280', 'MSTILE@1280', 'MSTILE@1280_trunc=drop,vote=0.6_'),
        ('MS-tiled 1536', 'MSTILE@1536', 'MSTILE@1536_trunc=drop,vote=0.6_'),
        ('HAHI-S', f'HAHI_{_T}_', f'HAHI_{_T},{F}_'),
        ('HAHI-S, route 0.1', f'HAHI_{_T},route_ratio=0.1_', f'HAHI_{_T},{F},route_ratio=0.1_')]


def fig4():
    off = {'UC 2×2': (4, 2, 'left'), 'SAHI 512': (-4, 4, 'right'), 'DAHI': (4, 2, 'left'),
           'DAHI, $N_{\\mathrm{ROI}}$ = 2': (4, 0, 'left'),
           'PAN': (4, -9, 'left'), 'MS-tiled 960': (-4, 4, 'right'), 'MS-tiled 1280': (5, -4, 'left'),
           'MS-tiled 1536': (4, -9, 'left'), 'HAHI-S': (4, 3, 'left'), 'HAHI-S, route 0.1': (-4, 4, 'right')}
    moff = {'Global 1280': (-4, 3, 'right'), 'MS 640+1280 (+ voting)': (-4, 5, 'right'),
            'MS 640+1536 (+ voting)': (0, -8, 'center'), 'MS 640+1280+1536': (-4, 4, 'right')}
    fig, ax = plt.subplots(figsize=(W2, 3.1))
    for label, base, plug in CROP:
        b, p = M('vd_testdev', 'v8s', base), M('vd_testdev', 'v8s', plug)
        if not (b and p):
            continue
        x = b['gflops']
        ax.scatter([x], [100 * b['mAP']], s=16, facecolor='white', edgecolor=C_BASE, lw=0.8, zorder=3)
        ax.scatter([x], [100 * p['mAP']], s=18, color=C_PLUG, zorder=4)
        ax.annotate('', xy=(x, 100 * p['mAP']), xytext=(x, 100 * b['mAP']),
                    arrowprops=dict(arrowstyle='-|>', color=C_PLUG, lw=0.7, mutation_scale=6))
        dx, dy, ha = off[label]
        ax.annotate(label, (x, 100 * p['mAP']), xytext=(dx, dy), textcoords='offset points', fontsize=6, ha=ha,
                    va='bottom')
    ms = [('Global 1280', 'global@1280'), ('MS 640+1280 (+ voting)', 'MS@640x1280_vote=0.6_'),
          ('MS 640+1536 (+ voting)', 'MS@640x1536_vote=0.6_'), ('MS 640+1280+1536', 'MS@640x1280x1536')]
    pts = [(n, M('vd_testdev', 'v8s', r)) for n, r in ms]
    pts = [(n, m) for n, m in pts if m]
    ax.plot([m['gflops'] for _, m in pts], [100 * m['mAP'] for _, m in pts], '-s', color=C_MS, ms=3.5, lw=0.8,
            zorder=2)
    for n, m in pts:
        dx, dy, ha = moff[n]
        two = n.startswith('MS 640+1536')
        ax.annotate(n.replace(' (+', '\n(+') if two else n, (m['gflops'], 100 * m['mAP']), xytext=(dx, dy),
                    textcoords='offset points', fontsize=6, color=C_MS, ha=ha, va='top' if two else 'bottom',
                    linespacing=1.0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec=C_BASE, label='crop method (original merging)'),
                       Line2D([], [], marker='o', ls='', color=C_PLUG, label='+ view-aware fusion (this work)'),
                       Line2D([], [], marker='s', color=C_MS, ms=3.5, lw=0.8,
                              label='full-image single/multi-scale testing')],
              loc='lower right', frameon=False)
    ax.set_xscale('log')
    ax.set_xlim(60, 340)
    ax.set_xticks([70, 100, 150, 200, 300])
    ax.set_xticklabels(['70', '100', '150', '200', '300'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('GFLOPs per image (log scale)')
    ax.set_ylabel('AP (%)')
    ax.set_ylim(19.2, 24.3)
    ax.grid(alpha=0.25, lw=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig4_accuracy_cost')


# ----------------------------------------------------------------------------------------------------------- Fig. 5/6
def _holdout_rows():
    from .holdout import PREREG, safe
    Mm = PREREG['methods']
    rows = []
    for d, t in [('vid_holdout', 'v8s'), ('vid_holdout', 'v8n'), ('vid_holdout', 'y11s'), ('uavdt_test', 'v8s')]:
        for a, b in PREREG['primary_families']['F1_plugin']['pairs']:
            p = os.path.join(RUN_ROOT, d, t, 'standard', f'bootstrap_ALL_thr0_ref-{safe(Mm[a])}.json')
            x = json.load(open(p)).get(safe(Mm[b])) if os.path.exists(p) else None
            mb = M(d, t, safe(Mm[b]))
            if x and mb:
                rows.append({'data': d, 'tag': t, 'method': b, 'd': x['mAP']['delta'], 'lo': x['mAP']['ci_lo'],
                             'hi': x['mAP']['ci_hi'], 'p': x['mAP']['p_le_0'], 'passes': mb['n_inf'], 'base': mb['mAP']})
    return rows


def _dev_rows():
    rows = []
    for t in ('v8s', 'v8n', 'y11s'):
        for label, base, plug in CROP:
            b, p = M('vd_testdev', t, base), M('vd_testdev', t, plug)
            if b and p:
                rows.append({'data': 'vd_testdev', 'tag': t, 'method': label, 'd': p['mAP'] - b['mAP'],
                             'passes': b['n_inf'], 'base': b['mAP']})
    return rows


def _holm(ps):
    o = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, k in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * max(ps[k], 0.001)))
        adj[k] = run
    return adj


def fig5():
    rows = _holdout_rows()
    meth = []
    for r in rows:
        if r['method'] not in meth:
            meth.append(r['method'])
    sets = [('vid_holdout', 'v8s', 'VID · YOLOv8s', 'o', '#1f77b4'), ('vid_holdout', 'v8n', 'VID · YOLOv8n', 's', '#2ca02c'),
            ('vid_holdout', 'y11s', 'VID · YOLO11s', 'D', '#9467bd'), ('uavdt_test', 'v8s', 'UAVDT · YOLOv8s', '^', '#ff7f0e')]
    adj = {}
    for d in ('vid_holdout', 'uavdt_test'):
        sub = [r for r in rows if r['data'] == d]
        for r, a in zip(sub, _holm([r['p'] for r in sub])):
            adj[id(r)] = a
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    for k, (d, t, lab, mk, c) in enumerate(sets):
        sub = {r['method']: r for r in rows if r['data'] == d and r['tag'] == t}
        xs = [meth.index(m) + (k - 1.5) * 0.17 for m in meth if m in sub]
        rr = [sub[m] for m in meth if m in sub]
        ax.errorbar(xs, [r['d'] * 100 for r in rr], yerr=[[100 * (r['d'] - r['lo']) for r in rr],
                    [100 * (r['hi'] - r['d']) for r in rr]], fmt=mk, ms=3, color=c, lw=0.7, capsize=1.2, label=lab)
        for x, r in zip(xs, rr):
            if adj[id(r)] >= 0.05:
                ax.scatter([x], [r['d'] * 100], s=30, facecolor='none', edgecolor='black', lw=0.7, zorder=5)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(range(len(meth)))
    lab = {'UC': 'UC 2×2', 'SAHI': 'SAHI 512', 'DAHI': 'DAHI', 'DAHI N_ROI=2': 'DAHI,\n$N_{\\mathrm{ROI}}$ = 2', 'PAN': 'PAN',
           'HAHI-S route 0.1': 'HAHI-S,\nroute 0.1'}
    ax.set_xticklabels([lab.get(m, m.replace('MS-tiled ', 'MS-tiled\n')) for m in meth], fontsize=6.2)
    ax.set_ylabel('ΔAP, + F − original\n(points, 95% CI)')
    ax.legend(ncol=4, loc='upper right', frameon=False)
    ax.set_ylim(-0.3, 2.6)
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig5_holdout')
    n_pos = int(sum(r['d'] > 0 for r in rows))
    n_sig = int(sum(adj[id(r)] < 0.05 for r in rows))
    json.dump({'n': len(rows), 'positive': n_pos, 'holm_sig': n_sig}, open(os.path.join(OUT, 'fig5_holdout.json'), 'w'))


def fig6():
    dev, hold = _dev_rows(), _holdout_rows()
    allr = dev + hold
    x = np.array([r['passes'] for r in allr])
    y = np.array([r['d'] for r in allr]) * 100
    rel = np.array([r['d'] / r['base'] for r in allr]) * 100
    rr = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={'width_ratios': [1, 1.05], 'wspace': 0.95})
    grp = [('VisDrone-DET test-dev (development)', [r for r in dev], 'o', '#7f7f7f'),
           ('VisDrone-VID (hold-out)', [r for r in hold if r['data'] == 'vid_holdout'], 's', '#1f77b4'),
           ('UAVDT (hold-out)', [r for r in hold if r['data'] == 'uavdt_test'], '^', '#ff7f0e')]
    for lab, rs, mk, c in grp:
        a1.scatter([r['passes'] for r in rs], [r['d'] * 100 for r in rs], s=11, marker=mk, color=c, alpha=0.85, lw=0,
                   label=lab)
    k, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    a1.plot(xx, k * xx + b0, color='black', lw=0.7, ls='--')
    a1.text(0.03, 0.97, f'Pearson r = {rr:.2f}\nSpearman ρ = {rho:.2f}\nn = {len(allr)}', transform=a1.transAxes, va='top',
            fontsize=6.3)
    a1.set_xlabel('detector passes per image (640×640 units)')
    a1.set_ylabel('ΔAP, + F − original (points)')
    a1.set_xlim(2, 18)
    a1.set_ylim(0, 1.9)
    a1.legend(loc='upper center', bbox_to_anchor=(0.45, -0.24), ncol=2, frameon=False, fontsize=5.6, handletextpad=0.2,
              columnspacing=0.8)
    a1.set_title('(a)', loc='left', fontsize=8, weight='bold')
    a1.spines[['top', 'right']].set_visible(False)
    H = json.load(open(os.path.join(RUN_ROOT, 'vd_val', 'v8s', 'views', 'headroom.json')))
    keys = [('miss_pool', 'Fusion: recover\ndropped GT'), ('cls_pool', 'Fusion: cross-view\nclass consensus'),
            ('loc_view', 'Fusion: best view\nper object (oracle)'), ('bkg', 'Detector:\nbackground FPs'),
            ('cls', 'Detector:\nclass confusion'), ('miss_det', 'Detector: missed\nby every view')]
    yy = np.arange(len(keys))[::-1]
    for off, (mname, lab, c) in zip((0.18, -0.18), [('HAHI-S', 'HAHI-S + F', C_PLUG), ('MS-1280x736', 'MS 640+1280 + voting', C_MS)]):
        mm = H[mname]['metrics']
        a2.barh(yy + off, [(mm[kk]['mAP'] - mm['base']['mAP']) * 100 for kk, _ in keys], height=0.34, color=c, label=lab)
    a2.set_yticks(yy)
    a2.set_yticklabels([l for _, l in keys], fontsize=6.0)
    a2.set_xlabel('ΔAP if fixed (points)')
    a2.axvline(0, color='black', lw=0.6)
    a2.axvline(0.15, color='#1f77b4', lw=0.6, ls=':')
    a2.annotate('best training-free\nview-trust rule (+0.15)', xy=(0.15, yy[2]), xytext=(3.2, yy[1] + 0.1), fontsize=5.6,
                color='#1f77b4', va='center', arrowprops=dict(arrowstyle='-', color='#1f77b4', lw=0.5))
    a2.legend(loc='upper right', frameon=False, fontsize=5.6)
    a2.set_xlim(0, 9.8)
    a2.set_title('(b)', loc='left', fontsize=8, weight='bold')
    a2.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig6_gain_and_errors')
    json.dump({'pearson_r': rr, 'spearman_rho': rho, 'n': len(allr), 'median_gain': float(np.median(y)),
               'q1': float(np.percentile(y, 25)), 'q3': float(np.percentile(y, 75)), 'min': float(y.min()),
               'max': float(y.max()), 'median_rel_pct': float(np.median(rel)), 'min_rel': float(rel.min()),
               'max_rel': float(rel.max())}, open(os.path.join(OUT, 'fig6_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    for name, fn in [('fig1', fig1), ('fig2', fig2), ('fig3', fig3), ('fig4', fig4), ('fig5', fig5), ('fig6', fig6)]:
        if not a.only or name in a.only.split(','):
            fn()
