"""Verify that HahiCfg(fast_merge=True) gives bit-identical detections to the thesis-code merge loops.

The detector is called once per image (slow path, outputs recorded) and the same outputs are replayed to the
fast path, so any difference can only come from the merge stage.

  python -m hahi_bench.check_fast_merge --data vd_val --limit 548
"""
import argparse
from dataclasses import replace

import cv2
import numpy as np

from . import thesis_methods as TM
from .data import load_dataset
from .detector import Detector

WEIGHTS = {'vd_val': 'hahi_bench/train_runs/v8s_official/weights/best.pt',
           'vd_testdev': 'hahi_bench/train_runs/v8s_official/weights/best.pt',
           'dota_val': 'hahi_bench/train_runs/v8s_dota/weights/best.pt'}
CONFIGS = {
    'HAHI (standard)': TM.standard(TM.THESIS),
    'HAHI (thesis protocol, conf 0.3)': TM.THESIS,
    'HAHI tuned': replace(TM.standard(TM.THESIS), gcc=1.0, route_ratio=0.1),
    'HAHI logistic upscale': replace(TM.standard(TM.THESIS), upscale='logistic'),
    'SBSI+Packing (no SCE)': replace(TM.standard(TM.THESIS), sce=False),
    'HAHI global 1280': replace(TM.standard(TM.THESIS), global_imgsz=1280, gcc=1.0),
}


class Recorder:
    def __init__(self, det):
        self.det, self.log = det, []

    def __call__(self, images, **kw):
        out = self.det(images, **kw)
        self.log.append([(b.copy(), s.copy(), c.copy()) for b, s, c in out])
        return out


class Replayer:
    def __init__(self, log):
        self.log = list(log)

    def __call__(self, images, **kw):
        return self.log.pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='vd_val')
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()
    recs, _ = load_dataset(args.data)
    det = Detector(WEIGHTS[args.data], max_det=1000)
    det.warmup()
    timer = lambda: {k: 0.0 for k in ['global', 'plan', 'build', 'local', 'merge']}
    for name, cfg in CONFIGS.items():
        n_bad, n_det = 0, 0
        for i in range(min(args.limit, len(recs))):
            img = cv2.imread(recs[i].path)
            rec = Recorder(det)
            a = TM.run_hahi(rec, img, cfg, timer())
            b = TM.run_hahi(Replayer(rec.log), img, replace(cfg, fast_merge=True), timer())
            same = all(x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
            n_bad += not same
            n_det += len(a[0])
        print(f'{name:<34} images {min(args.limit, len(recs))}  detections {n_det}  mismatching images {n_bad}',
              flush=True)


if __name__ == '__main__':
    main()
