import functools
import time

import numpy as np
import torch
import ultralytics.utils.nms as _nms
from ultralytics import YOLO

# Ultralytics aborts NMS after ~2 s (silently truncating detections), which can trigger under load or with
# very low score floors. Disable the time limit so every method's outputs are complete and deterministic.
if not getattr(_nms.non_max_suppression, '_no_time_limit', False):
    _orig_nms = _nms.non_max_suppression

    @functools.wraps(_orig_nms)
    def _nms_no_limit(*args, **kwargs):
        kwargs['max_time_img'] = 1e6
        return _orig_nms(*args, **kwargs)

    _nms_no_limit._no_time_limit = True
    _nms.non_max_suppression = _nms_no_limit


def letterbox_shape(h, w, imgsz, auto):
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    if auto:
        return nh + (imgsz - nh) % 32, nw + (imgsz - nw) % 32
    return imgsz, imgsz


class Detector:
    """Thin wrapper over an Ultralytics model with synchronized timing and cost accounting."""

    def __init__(self, weights, imgsz=640, device=0, iou=0.7, max_det=1000, half=False, force_batch=0, square=False):
        self.model = YOLO(weights)
        self.weights = weights
        self.force_batch = force_batch
        self.square = square
        self.use_cuda = device != 'cpu' and torch.cuda.is_available()
        self.imgsz = imgsz
        self.device = device
        self.iou = iou
        self.max_det = max_det
        self.half = half
        self.gflops640 = self._gflops640()
        self.reset()

    def _gflops640(self):
        try:
            from ultralytics.utils.torch_utils import get_flops
            return float(get_flops(self.model.model, imgsz=640))
        except Exception:
            return float('nan')

    def reset(self):
        self.n_images = 0
        self.n_calls = 0
        self.seconds = 0.0
        self.gflops = 0.0

    def warmup(self, n=5):
        dummy = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        for _ in range(n):
            self([dummy], conf=0.25)
        self.reset()

    def __call__(self, images, conf, imgsz=None, batch=16, iou=None, max_det=None):
        """Note: for in-memory image lists Ultralytics ignores `batch` and runs the whole list as ONE batch
        (LoadPilAndNumpy). This is also what the thesis code did. With force_batch, chunks are fed explicitly."""
        if not images:
            return []
        imgsz = imgsz or self.imgsz
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sizes = [im.shape[:2] for im in images]
        if self.square:
            # Emulates a static square input engine (NPU / fixed-shape TensorRT): pad bottom/right to a square so
            # the letterbox is exactly imgsz x imgsz at the same scale; box coordinates are unchanged.
            padded = []
            for im in images:
                h, w = im.shape[:2]
                if h == w:
                    padded.append(im)
                    continue
                s = max(h, w)
                p = np.full((s, s, 3), 114, np.uint8)
                p[:h, :w] = im
                padded.append(p)
            images = padded
        chunks = [images[i:i + self.force_batch] for i in range(0, len(images), self.force_batch)] \
            if self.force_batch else [images]
        out = []
        for ch in chunks:
            res = self.model.predict(ch, imgsz=imgsz, conf=conf, iou=iou or self.iou,
                                     max_det=max_det or self.max_det, batch=len(ch), verbose=False,
                                     device=self.device, half=self.half)
            for r in res:
                b = r.boxes
                out.append((b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                            b.cls.cpu().numpy().astype(np.int64)))
        if self.square:
            for k, (h, w) in enumerate(sizes):
                bb = out[k][0]
                bb[:, [0, 2]] = bb[:, [0, 2]].clip(0, w)
                bb[:, [1, 3]] = bb[:, [1, 3]].clip(0, h)
        if self.use_cuda:
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.n_calls += 1
        self.n_images += len(images)
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            same = len({im.shape for im in chunk}) == 1
            for im in chunk:
                hh, ww = letterbox_shape(im.shape[0], im.shape[1], imgsz, auto=same)
                self.gflops += self.gflops640 * (hh * ww) / (640 * 640)
        return out


class TorchvisionDetector:
    """Same interface as Detector for a torchvision Faster R-CNN checkpoint (train_frcnn.py).
    `imgsz` sets the longer side of the network input, as for the Ultralytics letterbox; the detector-internal NMS
    keeps torchvision's default (IoU 0.5)."""

    def __init__(self, weights, imgsz=640, device=0, max_det=1000, force_batch=0, **_):
        from .train_frcnn import build_model
        self.model = build_model(pretrained=False)
        sd = torch.load(weights, map_location='cpu', weights_only=False)
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.dev = torch.device(f'cuda:{device}' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        self.model.to(self.dev).eval()
        # inference: torchvision default RPN proposal counts (1000 / 1000) and FP16 autocast for speed
        self.model.rpn._pre_nms_top_n['testing'] = 1000
        self.model.rpn._post_nms_top_n['testing'] = 1000
        self.weights = weights
        self.force_batch = force_batch
        self.square = False
        self.use_cuda = self.dev.type == 'cuda'
        self.imgsz = imgsz
        self.max_det = max_det
        self.gflops640 = self._gflops640()
        self.reset()

    def _gflops640(self):
        try:
            from torch.utils.flop_counter import FlopCounterMode
            x = [torch.zeros(3, 640, 640, device=self.dev)]
            with torch.inference_mode(), FlopCounterMode(display=False) as fc:
                self.model(x)
            return fc.get_total_flops() / 1e9
        except Exception:
            return float('nan')

    reset = Detector.reset

    def warmup(self, n=3):
        dummy = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        for _ in range(n):
            self([dummy], conf=0.25)
        self.reset()

    def __call__(self, images, conf, imgsz=None, batch=16, iou=None, max_det=None):
        if not images:
            return []
        imgsz = imgsz or self.imgsz
        m = self.model
        m.transform.min_size, m.transform.max_size = (imgsz,), imgsz
        m.roi_heads.score_thresh = conf
        m.roi_heads.detections_per_img = max_det or self.max_det
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bs = self.force_batch or batch
        out = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.use_cuda):
            for i in range(0, len(images), bs):
                x = [torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1])).to(self.dev).permute(2, 0, 1).float() / 255.0
                     for im in images[i:i + bs]]
                for r in m(x):
                    out.append((r['boxes'].float().cpu().numpy().astype(np.float32), r['scores'].float().cpu().numpy().astype(np.float32),
                                (r['labels'] - 1).cpu().numpy().astype(np.int64)))
        if self.use_cuda:
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.n_calls += 1
        self.n_images += len(images)
        for im in images:
            r = imgsz / max(im.shape[:2])
            self.gflops += self.gflops640 * (round(im.shape[0] * r) * round(im.shape[1] * r)) / (640 * 640)
        return out


def make_detector(weights, **kw):
    if str(weights).endswith('.pth'):
        return TorchvisionDetector(weights, **kw)
    return Detector(weights, **kw)


def empty_dets():
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)


def concat_dets(parts):
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return empty_dets()
    return (np.concatenate([p[0] for p in parts]).astype(np.float32),
            np.concatenate([p[1] for p in parts]).astype(np.float32),
            np.concatenate([p[2] for p in parts]).astype(np.int64))
