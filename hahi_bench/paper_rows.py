"""Print HTML table rows for the manuscript from the run directories and bootstrap files (no hand-copied numbers).

  python -m hahi_bench.paper_rows dt       # Table 4(b): Dynamic Tiling vs view-aware fusion on the same 3x2 grid
  python -m hahi_bench.paper_rows frcnn    # Table 3: Faster R-CNN column
"""
import json
import os
import sys

from .make_tables import _exists
from .run_eval import RUN_ROOT
from .tables import metrics

_T = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv'
DT = [('3&times;2 tiles + FI, NMS', 'DYNTILE_dynamic=False,filt=False_'),
      ('3&times;2 tiles + FI + TAF', 'DYNTILE_trunc=drop,filt=False_'),
      ('<b>3&times;2 tiles + FI + TAF + voting (+ F)</b>', 'DYNTILE_trunc=drop,filt=False,vote=0.6_'),
      ('Dynamic Tiling, as published (with large&ndash;small filter)', 'DYNTILE_alpha=0.002,iou=0.7,plan_conf=0.1_'),
      ('Dynamic Tiling, best val configuration (no filter)', 'DYNTILE_filt=False,plan_conf=0.1_'),
      ('Dynamic Tiling, best val configuration + voting', 'DYNTILE_filt=False,plan_conf=0.1,vote=0.6_')]


def boot(data, tag, a, b):
    p = os.path.join(RUN_ROOT, data, tag, 'standard', f'bootstrap_ALL_thr0_ref-{a}.json')
    return json.load(open(p)).get(b) if os.path.exists(p) else None


def fm(r, scale=100):
    """Delta with CI; a negative scale flips the sign (stats.py reports ref - other)."""
    d, lo, hi = scale * r['delta'], scale * r['ci_lo'], scale * r['ci_hi']
    lo, hi = min(lo, hi), max(lo, hi)
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def dt():
    """Rows of Table 4(b): YOLOv8s metrics on test-dev, delta vs '+ F' with CI for v8s and point deltas for v8n / y11s."""
    ref = 'DYNTILE_trunc=drop,filt=False,vote=0.6_'
    rows = []
    for lab, r in DT:
        a = metrics('vd_testdev', 'v8s', 'standard', r)['ALL']
        cells = [f'<td colspan="2">{lab}</td>', f"<td>{a['n_inf']:.2f}</td>", f"<td>{a['gflops']:.0f}</td>",
                 f"<td>{100 * a['mAP']:.2f}</td>", f"<td>{100 * a['AP75']:.2f}</td>", f"<td>{100 * a['APs']:.2f}</td>"]
        if r == ref:
            cells += ['<td>reference</td>', '<td>reference</td>', '<td>reference</td>']
        else:
            b = boot('vd_testdev', 'v8s', ref, r)
            cells.append(f"<td>{fm(b['mAP'], -100) if b else '&mdash;'}</td>")
            for t in ('v8n', 'y11s'):
                if _exists('vd_testdev', t, 'standard', r) and _exists('vd_testdev', t, 'standard', ref):
                    x = metrics('vd_testdev', t, 'standard', r)['ALL']['mAP'] - metrics('vd_testdev', t, 'standard', ref)['ALL']['mAP']
                    bt = boot('vd_testdev', t, ref, r)
                    sig = '' if bt is None else ('' if (bt['mAP']['ci_lo'] > 0 or bt['mAP']['ci_hi'] < 0) else '<sup>n.s.</sup>')
                    cells.append(f'<td>{100 * x:+.2f}{sig}</td>')
                else:
                    cells.append('<td>&mdash;</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    print('\n'.join(rows))
    for t in ('v8s', 'v8n', 'y11s'):
        print(t, {r: (round(100 * metrics('vd_testdev', t, 'standard', r)['ALL']['mAP'], 2) if _exists('vd_testdev', t, 'standard', r) else None)
                  for _, r in DT})
    v = {r: round(100 * metrics('vd_val', 'v8s', 'standard', r)['ALL']['mAP'], 2) for _, r in DT if _exists('vd_val', 'v8s', 'standard', r)}
    print('val', v)


FR = [('UC 2&times;2', 'UC2x2', 'UC2x2_trunc=drop,vote=0.6_'),
      ('SAHI 512', 'SAHI@512_post=NMS,metric=IOU_', 'SAHI@512_post=NMS,metric=IOU,trunc=drop,vote=0.6_'),
      ('DAHI (CMA-NMS &rarr; F)', 'DAHI', 'DAHI_trunc=drop,vote=0.6_'),
      ('PAN', 'PAN_bandwidth=256_', 'PAN_bandwidth=256,trunc=drop,vote=0.6_'),
      ('MS-tiled 1280', 'MSTILE@1280', 'MSTILE@1280_trunc=drop,vote=0.6_'),
      ('HAHI-S', f'HAHI_{_T}_', f'HAHI_{_T},trunc=drop,vote=0.6_')]


def frcnn():
    """Rows of the Faster R-CNN block of Table 3 (EN), with the paired bootstrap CI of + F minus original."""
    for lab, a, b in FR:
        x, y = metrics('vd_testdev', 'frcnn', 'standard', a)['ALL'], metrics('vd_testdev', 'frcnn', 'standard', b)['ALL']
        bb = boot('vd_testdev', 'frcnn', b, a)
        d = (f"{100 * bb['mAP']['delta']:+.2f} [{100 * bb['mAP']['ci_lo']:.2f}, {100 * bb['mAP']['ci_hi']:.2f}]" if bb
             else f"{100 * (y['mAP'] - x['mAP']):+.2f}")
        print(f"<tr><td>{lab}</td><td>{x['n_inf']:.2f}</td><td>{100 * x['mAP']:.2f} &rarr; {100 * y['mAP']:.2f}</td><td>{d}</td>"
              f"<td colspan=\"4\" style=\"text-align:left\">&Delta;AP<sub>75</sub> {100 * (y['AP75'] - x['AP75']):+.2f}, "
              f"&Delta;AP<sub>S</sub> {100 * (y['APs'] - x['APs']):+.2f}</td></tr>")
    for lab, r in [('Global 640', 'global@640'), ('Global 1280', 'global@1280'), ('MS 640+1280', 'MS@640x1280'),
                   ('MS 640+1280 + voting', 'MS@640x1280_vote=0.6_')]:
        if _exists('vd_testdev', 'frcnn', 'standard', r):
            x = metrics('vd_testdev', 'frcnn', 'standard', r)['ALL']
            print(f"{lab}: AP {100 * x['mAP']:.2f} APs {100 * x['APs']:.2f} passes {x['n_inf']:.2f}")
    for a, b in [(f'HAHI_{_T},trunc=drop,vote=0.6_', 'MS@640x1280_vote=0.6_'), (f'HAHI_{_T}_', 'MS@640x1280')]:
        bb = boot('vd_testdev', 'frcnn', a, b)
        if bb:
            print(a[:8], 'vs', b, fm(bb['mAP']))


if __name__ == '__main__':
    {'dt': dt, 'frcnn': frcnn}[sys.argv[1]]()
