"""Named experiment groups (the exact method specs that were run), executed in one process per model.

Example: python -m hahi_bench.experiments --group val_sweep --data vd_val --weights W.pt --tag v8s
"""
import argparse
import os

from .data import load_dataset
from .detector import Detector
from .run_eval import RUN_ROOT, run_one, safe

THESIS_ROWS = ['UC2x2', 'DAHI-thesis', 'Packing-thesis', 'DAHI+Packing-thesis', 'SBSI+Packing', 'HAHI']

GROUPS = {
    # Final comparison on test-dev: every baseline at its best val configuration (selection rule: val mAP).
    'main': THESIS_ROWS + [
        'global@640', 'global@1280', 'MS@640x1280', 'MS@640x1536', 'MS@640x1280x1536',
        'SAHI@512{post=NMS,metric=IOU}', 'DAHI', 'DAHI{n_roi=2}', 'PAN{bandwidth=256}',
        'HAHI{gcc=1.0,route_ratio=0.1}',             # HAHI tuned on val (thesis design)
        'HAHI{global_imgsz=1280,gcc=1.0}',           # exploratory extension (global pass at 1280)
    ],
    # Baseline + method tuning on the official val split only.
    'val_sweep': THESIS_ROWS + [
        'global@640', 'global@960', 'global@1280', 'global@1536',
        'SAHI@512{post=GREEDYNMM,conf=0.3}', 'SAHI@512{post=GREEDYNMM,conf=0.1}', 'SAHI@512{post=NMS,metric=IOU}',
        'SAHI@640{post=GREEDYNMM,conf=0.3}', 'SAHI@640{post=GREEDYNMM,conf=0.1}', 'SAHI@640{post=NMS,metric=IOU}',
        'SAHI@384{post=NMS,metric=IOU}',
        'DAHI', 'DAHI{n_roi=2}', 'DAHI{size_min=320,size_max=640}', 'DAHI{size_min=320,size_max=640,n_roi=2}',
        'PAN{bandwidth=64}', 'PAN{bandwidth=128}', 'PAN{bandwidth=256}', 'PAN{bandwidth=128,conf_coarse=0.25}',
        'PAN{bandwidth=128,fuse_global=False}',
    ],
    # Same methods/configs as 'main', re-run on an otherwise idle GPU for latency and memory figures.
    'timing': THESIS_ROWS + [
        'global@640', 'global@1280', 'MS@640x1280', 'MS@640x1536', 'MS@640x1280x1536',
        'SAHI@512{post=NMS,metric=IOU}', 'DAHI', 'DAHI{n_roi=2}', 'PAN{bandwidth=256}',
        'HAHI{gcc=1.0,route_ratio=0.1}', 'HAHI{global_imgsz=1280,gcc=1.0}',
    ],
    # Detector generalisation (other detectors, same configs as chosen on val for YOLOv8s; no re-tuning).
    'generalize': [
        'global@640', 'global@1280', 'MS@640x1280', 'MS@640x1536', 'MS@640x1280x1536', 'UC2x2', 'SAHI@512{post=NMS,metric=IOU}',
        'DAHI', 'DAHI{n_roi=2}', 'PAN{bandwidth=256}', 'DAHI+Packing-thesis', 'SBSI+Packing', 'HAHI',
        'HAHI{gcc=1.0,route_ratio=0.1}', 'HAHI{global_imgsz=1280,gcc=1.0}',
    ],
    # CPU-only latency (no GPU at all) on a small subset.
    'cpu': ['global@640', 'global@1280', 'MS@640x1280', 'UC2x2', 'SAHI@512{post=NMS,metric=IOU}',
            'DAHI{n_roi=2}', 'PAN{bandwidth=256}', 'HAHI', 'HAHI{gcc=1.0,route_ratio=0.1}'],
    # Component ablation of HAHI (standard protocol).
    'ablation': [
        'HAHI{packing=False}',                       # SBSI only (+ global)
        'Packing-thesis{sce=True}',                   # packing + SCE, no routing
        'HAHI{upscale=none}', 'HAHI{upscale=logistic}',
        'HAHI{weight=uniform}', 'HAHI{weight=conf}', 'HAHI{weight=sqrt_area}', 'HAHI{weight=conf_area}',
        'HAHI{snap=False}', 'HAHI{gcc=1.0}', 'HAHI{decay=1.0}', 'HAHI{gcc=1.0,decay=1.0}',
        'HAHI{canvas=640}',
    ],
    # Hyper-parameter sensitivity of HAHI.
    'sensitivity': [
        'HAHI{route_ratio=0.1}', 'HAHI{route_ratio=0.2}', 'HAHI{route_ratio=0.4}', 'HAHI{route_ratio=0.5}',
        'HAHI{window=384}', 'HAHI{window=640}', 'HAHI{window=768}',
        'HAHI{sigma_div=2.0}', 'HAHI{sigma_div=8.0}',
        'HAHI{upscale_ratio=1.25}', 'HAHI{upscale_ratio=2.0}',
        'HAHI{conf_filter=0.05}', 'HAHI{conf_filter=0.2}',
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='standard')
    ap.add_argument('--extra', default='', help='additional semicolon-separated specs')
    ap.add_argument('--batch1', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--device', default='0', help="'0' for GPU, 'cpu' for CPU latency study")
    ap.add_argument('--suffix', default='', help='output sub-folder suffix, e.g. _timing')
    ap.add_argument('--square', action='store_true', help='emulate a static square-input engine (pad to square)')
    args = ap.parse_args()
    specs = [] if args.group == 'none' else list(GROUPS[args.group])
    specs += [s.strip() for s in args.extra.split(';') if s.strip()]
    recs, _ = load_dataset(args.data)
    device = 'cpu' if args.device == 'cpu' else int(args.device)
    det = Detector(args.weights, max_det=1000 if args.protocol == 'standard' else 300,
                   force_batch=1 if args.batch1 else 0, device=device, square=args.square)
    det.warmup()
    sub = args.protocol + ('_b1' if args.batch1 else '') + ('_cpu' if device == 'cpu' else '') + \
        ('_sq' if args.square else '') + args.suffix
    for spec in specs:
        out = os.path.join(RUN_ROOT, args.data, args.tag, sub, safe(spec))
        run_one(det, recs, spec, args.protocol, out, args.limit)


if __name__ == '__main__':
    main()
