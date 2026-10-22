"""Supporting Information for the manuscript (paper/supplementary_en.html).

Tables are taken from the auto-generated tables of REPORT.md (same numbers as the runs) and from the run directories.
AP-type values with four decimals are converted to percent with two decimals, the unit used in the manuscript.

  python -m hahi_bench.paper_si
"""
import html
import json
import os
import re

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


def md_table(key):
    m = re.search(rf'<!-- TABLE:{key} -->\n(.*?)\n<!-- /TABLE:{key} -->', REPORT, re.S)
    if not m:
        raise KeyError(key)
    rows = [r for r in m.group(1).strip().split('\n') if r.startswith('|')]
    if len(rows) < 2:
        raise KeyError(key)
    split = lambda r: [c.strip() for c in r.strip().strip('|').split('|')]
    head = [HEAD.get(h, h.replace('mAP', 'AP')) for h in split(rows[0])]
    body = [split(r) for r in rows[2:]]
    return head, body


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
        bold = c.startswith('**') and c.endswith('**')
        c = html.escape(c.strip('*')).replace('&quot;', '"')
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


def main():
    css = open(os.path.join(os.path.dirname(OUT), 'paper.css'), encoding='utf-8').read()
    P = []
    add = P.append
    add('<div class="article-type">Supporting Information</div>')
    add('<h1 class="title">View-aware fusion for crop-based small-object detection in high-resolution drone imagery: '
        'Supporting Information</h1><div class="authors">Byung Jung Han</div>')
    add('<p class="small">All AP values are in percent (COCO AP over IoU 0.50:0.95, up to 500 detections per image; 2,000 for '
        'DOTA). "Passes" are detector passes per image in 640&times;640 units. Δ values are in AP points; brackets give '
        '95% paired bootstrap intervals (1,000 resamples; whole sequences for video data). Method names follow the manuscript; '
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
    add(table('TABLE S6 View-aware fusion as a plug-in: val (YOLOv8s) and test-dev (three detectors), with paired bootstrap '
              'intervals for test-dev.', 'T21'))

    add('<h2>S2 | Pre-registered hold-out evaluation</h2>')
    add(table('TABLE S7 AP of all 30 pre-registered methods on the hold-out data.', 'T22a'))
    add(table('TABLE S8 Confirmatory family F1 (plug-in), sequence-level bootstrap with Holm correction within each dataset. '
              'The Holm-adjusted value 0.033 is the bootstrap resolution (0.001) times the number of comparisons.', 'T22b'))
    add(table('TABLE S9 Confirmatory family F2 (single-engine efficiency of HAHI-S).', 'T22c'))
    add(table('TABLE S10 Descriptive comparisons without a directional hypothesis (pre-registered).', 'T22d'))

    add('<h2>S3 | Selected configurations</h2>')
    cfg = [['SAHI 512', 'official library v0.11.36; slice 512, overlap 0.2, full-image prediction; NMS IoU 0.5 (GREEDYNMM default rejected on val)'],
           ['DAHI', 'paper defaults: β = 0.001, 1,000 random regions (width 480–960, 16:9), RDE λ = 0.001, DACS IoS 0.25, C_TH = 0.01, CMA-NMS t = 0.65, θ = 10, t_MT = 0.9; N_ROI = 2 variant from the paper'],
           ['PAN', '"fine-grained + packing" configuration (the learned patch-arrangement modules are not reproduced); mean-shift bandwidth 256, up to 4 patches per canvas'],
           ['UC 2×2', 'global 640 pass + four quadrants (thesis code), merging NMS 0.4'],
           ['MS a+b', 'full image at each size, class-wise NMS IoU 0.6 (0.7 for 640+1536 in the hold-out evaluation, selected on val); voting IoU equal to the NMS IoU'],
           ['MS-tiled S', 'global 640 pass + minimum number of 640×640 tiles covering the image resized to longer side S; NMS 0.6'],
           ['HAHI-S', 'uniform density weights, no snapping, no upscaling, no global calibration, merging NMS 0.6; window 512, canvas 960, routing ratio 0.3 (0.1 for "route 0.1")'],
           ['+ F (view-aware fusion)', 'TAF with δ = 5 px at the detector input for tiles and crops (UC, SAHI, DAHI, MS-tiled), 5 native px for HAHI dense windows, 5 canvas px for PAN canvases and 3 canvas px for HAHI canvases; class-wise NMS at the method\'s own merging threshold τn (UC 0.4, SAHI 0.5, PAN 0.5, HAHI-S 0.6, MS-tiled 0.6, DAHI 0.65) followed by box voting with τv = 0.6'],
           ['Dynamic Tiling', 'resize to 1920×1080, 3×2 tiles, edge threshold 2%, Tile Minimizer, full-image pass; selected on val: no large-small filter, NMS 0.6, planning score 0.1 (see S8)']]
    add(table('TABLE S11 Configurations used in the manuscript.', head=['Method', 'Configuration'], body=cfg, convert=False))

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

    add('<h2>S5 | Pre-registration files</h2>')
    man = json.load(open(os.path.join(os.path.dirname(OUT), 'prereg_bundle', 'MANIFEST.json')))
    body = [[f, v['sha256'][:16] + '…', v['created_local'].replace('T', ' ')] for f, v in man['files'].items()]
    body += [[f'first output, {k}', '', v.replace('T', ' ')] for k, v in man['first_holdout_output_local'].items()]
    add(table('TABLE S14 Pre-registration files (SHA-256 prefix, local creation time, KST) and the time of the first hold-out '
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

    doc = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Supporting Information</title>'
           '<link rel="stylesheet" href="paper.css"></head><body><div class="page">' + '\n'.join(P) + '</div></body></html>')
    open(OUT, 'w', encoding='utf-8').write(doc)
    print('written', OUT, len(doc))


if __name__ == '__main__':
    main()
