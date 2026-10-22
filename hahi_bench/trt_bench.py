"""TensorRT static-engine measurement on the local GPU (proxy for fixed-input accelerators).

Engines (batch 1, static shape): 640x640 for the global pass / crops / tiles, plus 736x1280 and 864x1536 (16:9)
for the high-resolution pass of multi-scale inference. FP16 and INT8 (calibrated on VisDrone-DET val).

  set PYTHONPATH=G:\\pylibs\\trt
  python -m hahi_bench.trt_bench export --prec fp16
  python -m hahi_bench.trt_bench export --prec int8
  python -m hahi_bench.trt_bench run --prec fp16 --protocol thesis      # deployment score floor 0.3 -> latency
  python -m hahi_bench.trt_bench run --prec int8 --protocol standard    # accuracy under INT8
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch

from .data import load_dataset
from .detector import Detector
from .run_eval import RUN_ROOT, run_one

HERE = os.path.dirname(__file__)
WORK = os.path.join(HERE, 'trt_work')
WEIGHTS = os.path.join(HERE, 'train_runs', 'v8s_official', 'weights', 'best.pt')
CALIB = os.path.join(os.environ.get('HAHI_VISDRONE_ROOT', 'F:/datasets/VisDrone2019'), 'visdrone_official.yaml')
SHAPES = {640: (640, 640), 1280: (736, 1280), 1536: (864, 1536)}
S = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv'
F = 'trunc=drop,vote=0.6'
SPECS = ['global@640', f'HAHI{{{S},{F}}}', f'HAHI{{{S},{F},route_ratio=0.1}}',
         f'MSTILE@1136{{{F}}}', f'MSTILE@1280{{{F}}}', f'MSTILE@1536{{{F}}}',
         'MS@640x1280{aspect=1.7778,vote=0.6}', 'MS@640x1536{aspect=1.7778,iou=0.7,vote=0.7}']


def engine_path(prec, size):
    return os.path.join(WORK, f'v8s_{prec}_{SHAPES[size][0]}x{SHAPES[size][1]}.engine')


def export(prec):
    from ultralytics import YOLO
    os.makedirs(WORK, exist_ok=True)
    for size, shape in SHAPES.items():
        dst = engine_path(prec, size)
        if os.path.exists(dst):
            continue
        src = os.path.join(WORK, f'v8s_{prec}_{size}.pt')
        shutil.copy(WEIGHTS, src)
        t = time.time()
        out = YOLO(src).export(format='engine', imgsz=list(shape), batch=1, dynamic=False, simplify=False,
                               half=prec == 'fp16', int8=prec == 'int8', data=CALIB if prec == 'int8' else None,
                               workspace=4, device=0, verbose=False)
        shutil.move(out, dst)
        print(f'{prec} {shape}: {dst} ({time.time() - t:.0f} s)', flush=True)


class EngineDetector(Detector):
    """Routes each call to the static engine of the requested input size (all calls batch 1)."""

    def __init__(self, prec, max_det=1000):
        from ultralytics import YOLO
        self.models = {s: YOLO(engine_path(prec, s), task='detect') for s in SHAPES}
        self.weights = f'tensorrt-{prec}'
        self.force_batch = 1
        self.square = False
        self.use_cuda = True
        self.imgsz = 640
        self.device = 0
        self.iou = 0.7
        self.max_det = max_det
        self.half = False
        self.gflops640 = float('nan')
        self.reset()

    def warmup(self, n=10):
        for s, (h, w) in SHAPES.items():
            dummy = np.full((h, w, 3), 114, np.uint8)
            for _ in range(n):
                self([dummy], conf=0.25, imgsz=s)
        self.reset()

    def __call__(self, images, conf, imgsz=None, batch=16, iou=None, max_det=None):
        if not images:
            return []
        size = imgsz or 640
        model, shape = self.models[size], list(SHAPES[size])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = []
        for im in images:
            r = model.predict(im, imgsz=shape, conf=conf, iou=iou or self.iou, max_det=max_det or self.max_det,
                              verbose=False, device=0)[0].boxes
            out.append((r.xyxy.cpu().numpy().astype(np.float32), r.conf.cpu().numpy().astype(np.float32),
                        r.cls.cpu().numpy().astype(np.int64)))
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.n_calls += 1
        self.n_images += len(images)
        self.gflops += sum(SHAPES[size][0] * SHAPES[size][1] for _ in images) / 640 ** 2   # 640x640 units
        return out


def run(prec, protocol, data, limit):
    recs, _ = load_dataset(data)
    det = EngineDetector(prec, max_det=1000 if protocol == 'standard' else 300)
    det.warmup()
    for spec in SPECS:
        from .run_eval import safe
        out = os.path.join(RUN_ROOT, data, 'v8s', f'{protocol}_trt_{prec}', safe(spec))
        run_one(det, recs, spec, protocol, out, limit)


def engines(n=300):
    """Single-pass accuracy of every static engine (FP16 vs INT8) on a fixed 300-image test-dev subset."""
    import json
    import cv2
    from .metrics import coco_eval
    recs, names = load_dataset('vd_testdev')
    idx = list(range(0, len(recs), 5))[:n]
    imgs = {i: cv2.imread(recs[i].path) for i in idx}
    out = {}
    for prec in ('fp16', 'int8'):
        det = EngineDetector(prec)
        for size in SHAPES:
            D = {i: det([imgs[i]], conf=0.001, imgsz=size)[0] for i in idx}
            m, _ = coco_eval(recs, D, idx, n_cls=len(names))
            out[f'{prec}_{size}'] = {k: m[k] for k in ('mAP', 'AP50', 'APs', 'APm', 'APl')}
            print(prec, size, f"{m['mAP']:.4f}", flush=True)
    with open(os.path.join(WORK, 'engine_single_pass.json'), 'w') as f:
        json.dump({'images': idx, 'results': out}, f, indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['export', 'run', 'engines'])
    ap.add_argument('--prec', default='fp16', choices=['fp16', 'int8'])
    ap.add_argument('--protocol', default='thesis')
    ap.add_argument('--data', default='vd_testdev')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if a.cmd == 'export':
        export(a.prec)
    elif a.cmd == 'engines':
        engines()
    else:
        run(a.prec, a.protocol, a.data, a.limit)
