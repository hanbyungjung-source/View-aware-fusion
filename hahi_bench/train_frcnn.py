"""Non-YOLO detector for the generality check: torchvision Faster R-CNN (ResNet-50-FPN v2, COCO weights) fine-tuned
on the official VisDrone2019-DET training split.

Training mirrors the YOLO protocol where possible: full images with the longer side resized to 640 pixels, the same
YOLO-format labels (classes 0-9; ignore regions and 'others' excluded), horizontal flips, no other augmentation.
Schedule: SGD, lr 0.005 (batch 4, linear scaling of 0.02 @ 16), momentum 0.9, weight decay 1e-4, 1000-iteration linear
warm-up, 12 epochs with x0.1 steps after epochs 8 and 11 (the standard '1x' schedule), mixed precision.
Anchors are halved relative to the COCO default (16-256 px) because VisDrone objects are small at a 640 input.

  python -m hahi_bench.train_frcnn --out G:/hahi_frcnn
"""
import argparse
import math
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.join(os.environ.get('HAHI_VISDRONE_ROOT', 'F:/datasets/VisDrone2019'), 'VisDrone2019-DET-train')
N_CLS = 10


def build_model(num_classes=N_CLS + 1, pretrained=True):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                   min_size=640, max_size=640, box_detections_per_img=1000, box_score_thresh=0.001,
                                   rpn_pre_nms_top_n_test=3000, rpn_post_nms_top_n_test=2000)
    m.rpn.anchor_generator = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return m


class VisDroneTrain(Dataset):
    def __init__(self, root=ROOT, long_side=640, train=True):
        self.img_dir, self.lbl_dir = os.path.join(root, 'images'), os.path.join(root, 'labels')
        self.names = sorted(os.listdir(self.img_dir))
        self.long_side, self.train = long_side, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(os.path.join(self.img_dir, name))[:, :, ::-1]
        h, w = img.shape[:2]
        r = self.long_side / max(h, w)
        nh, nw = round(h * r), round(w * r)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        boxes, labels = [], []
        p = os.path.join(self.lbl_dir, os.path.splitext(name)[0] + '.txt')
        if os.path.exists(p):
            for row in open(p):
                c, cx, cy, bw, bh = row.split()
                cx, cy, bw, bh = float(cx) * nw, float(cy) * nh, float(bw) * nw, float(bh) * nh
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                if x2 - x1 < 1:
                    x1, x2 = cx - 0.5, cx + 0.5
                if y2 - y1 < 1:
                    y1, y2 = cy - 0.5, cy + 0.5
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c) + 1)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            boxes[:, [0, 2]] = nw - boxes[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t, {'boxes': torch.from_numpy(boxes), 'labels': torch.as_tensor(labels, dtype=torch.int64)}


def collate(b):
    return tuple(zip(*b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.environ.get('HAHI_FRCNN_DIR', 'G:/hahi_frcnn'))
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device('cuda')
    model = build_model().to(dev)
    ds = VisDroneTrain()
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    persistent_workers=True, pin_memory=True, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    steps = (8, 11)
    scaler = torch.amp.GradScaler('cuda')
    start = 0
    ck = os.path.join(a.out, 'last.pth')
    if os.path.exists(ck):
        s = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['opt'])
        scaler.load_state_dict(s['scaler'])
        start = s['epoch'] + 1
        print('resumed from epoch', start, flush=True)
    it_per_epoch = len(dl)
    warm = 1000
    for ep in range(start, a.epochs):
        model.train()
        t0, run = time.time(), {}
        for k, (imgs, tgts) in enumerate(dl):
            it = ep * it_per_epoch + k
            lr = a.lr * (0.1 ** sum(ep >= s for s in steps))
            if it < warm:
                lr *= 0.001 + (1 - 0.001) * it / warm
            for g in opt.param_groups:
                g['lr'] = lr
            imgs = [i.to(dev, non_blocking=True) for i in imgs]
            tgts = [{kk: v.to(dev) for kk, v in t.items()} for t in tgts]
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model(imgs, tgts)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                print('non-finite loss, skipping batch', {kk: float(v) for kk, v in losses.items()}, flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            for kk, v in losses.items():
                run[kk] = run.get(kk, 0.0) * 0.98 + float(v) * 0.02
            if k % 200 == 0:
                el = time.time() - t0
                print(f'ep {ep} it {k}/{it_per_epoch} lr {lr:.5f} ' + ' '.join(f'{kk[5:]} {v:.3f}' for kk, v in run.items())
                      + f' | {el / max(k, 1):.3f}s/it', flush=True)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': ep}, ck)
        print(f'epoch {ep} done in {(time.time() - t0) / 60:.1f} min', flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'final.pth'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
