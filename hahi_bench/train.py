"""Convert official VisDrone2019-DET annotations to YOLO labels and train detectors on the official split.

Usage: python -m hahi_bench.train --model yolov8s.pt --name v8s_official
"""
import argparse
import os

from PIL import Image

ROOT = os.environ.get('HAHI_VISDRONE_ROOT', 'F:/datasets/VisDrone2019')
SPLITS = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
YAML = os.path.join(ROOT, 'visdrone_official.yaml')


def convert():
    for split in SPLITS:
        img_dir, ann_dir, lbl_dir = (os.path.join(ROOT, split, d) for d in ('images', 'annotations', 'labels'))
        os.makedirs(lbl_dir, exist_ok=True)
        for name in os.listdir(img_dir):
            stem = os.path.splitext(name)[0]
            out = os.path.join(lbl_dir, stem + '.txt')
            if os.path.exists(out):
                continue
            with Image.open(os.path.join(img_dir, name)) as im:
                w, h = im.size
            lines = []
            with open(os.path.join(ann_dir, stem + '.txt')) as f:
                for row in f:
                    p = [x for x in row.strip().split(',') if x != '']
                    if len(p) < 6:
                        continue
                    x, y, bw, bh, score, cat = map(int, p[:6])
                    if score == 0 or cat in (0, 11) or bw <= 0 or bh <= 0:
                        continue
                    lines.append(f'{cat - 1} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n')
            with open(out, 'w') as f:
                f.writelines(lines)
    names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    with open(YAML, 'w') as f:
        f.write(f'path: {ROOT}\ntrain: {SPLITS[0]}/images\nval: {SPLITS[1]}/images\ntest: {SPLITS[2]}/images\n')
        f.write('names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names)))


def convert_dota():
    """DOTAv1 (Ultralytics OBB format) -> axis-aligned YOLO labels; images are reused via a junction."""
    import subprocess
    src = os.environ.get('HAHI_DOTA_ROOT', 'F:/datasets/DOTA/DOTAv1')
    dst = src.rstrip('/\\') + '_hbb'
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f'{dst}/images'):
        subprocess.run(['cmd', '/c', 'mklink', '/J', os.path.normpath(f'{dst}/images'), os.path.normpath(f'{src}/images')],
                       check=True)
    for split in ('train', 'val'):
        os.makedirs(f'{dst}/labels/{split}', exist_ok=True)
        for name in os.listdir(f'{src}/labels/{split}'):
            out = f'{dst}/labels/{split}/{name}'
            if os.path.exists(out):
                continue
            lines = []
            with open(f'{src}/labels/{split}/{name}') as f:
                for row in f:
                    p = row.split()
                    if len(p) != 9:
                        continue
                    xs, ys = [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
                    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
                    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
                    if x2 > x1 and y2 > y1:
                        lines.append(f'{p[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n')
            with open(out, 'w') as f:
                f.writelines(lines)
    from .data import DOTA_NAMES
    yaml = f'{dst}/dota_hbb.yaml'
    with open(yaml, 'w') as f:
        f.write(f'path: {dst}\ntrain: images/train\nval: images/val\n')
        f.write('names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(DOTA_NAMES)))
    return yaml


def convert_uavdt():
    """UAVDT-DET train split (30 sequences = all minus the official 20 test sequences) -> YOLO format.
    25 sequences for training (every 5th frame), 5 held out for early stopping (every 10th frame); test sequences
    are never touched. Ignore regions are painted with the letterbox grey (114) so unlabeled objects are not
    learned as background."""
    import cv2
    import numpy as np
    from .data import UAVDT_ROOT, UAVDT_TEST, UAVDT_NAMES, read_uavdt_gt
    out = os.path.join(UAVDT_ROOT, 'yolo')
    yaml_path = os.path.join(out, 'uavdt.yaml')
    if os.path.exists(yaml_path):
        return yaml_path
    seqs = sorted(s for s in os.listdir(os.path.join(UAVDT_ROOT, 'UAV-benchmark-M')) if s not in UAVDT_TEST)
    assert len(seqs) == 30, len(seqs)
    val_seqs = seqs[::6]
    for split, ss, stride in (('train', [s for s in seqs if s not in val_seqs], 5), ('val', val_seqs, 10)):
        os.makedirs(os.path.join(out, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out, 'labels', split), exist_ok=True)
        for seq in ss:
            obj, ign = read_uavdt_gt(seq)
            img_dir = os.path.join(UAVDT_ROOT, 'UAV-benchmark-M', seq)
            for k, name in enumerate(sorted(os.listdir(img_dir))):
                if k % stride:
                    continue
                fr = int(''.join(ch for ch in name if ch.isdigit()))
                img = cv2.imread(os.path.join(img_dir, name))
                h, w = img.shape[:2]
                for x1, y1, x2, y2, _ in ign.get(fr, []):
                    img[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))] = 114
                stem = f'{seq}_{os.path.splitext(name)[0]}'
                cv2.imwrite(os.path.join(out, 'images', split, stem + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(out, 'labels', split, stem + '.txt'), 'w') as f:
                    for x1, y1, x2, y2, c in obj.get(fr, []):
                        if c not in (1, 2, 3):
                            continue
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        f.write(f'{c - 1} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} '
                                f'{(y2 - y1) / h:.6f}\n')
        print(split, len(ss), 'sequences', len(os.listdir(os.path.join(out, 'images', split))), 'images', flush=True)
    with open(yaml_path, 'w') as f:
        f.write(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n' +
                ''.join(f'  {i}: {n}\n' for i, n in enumerate(UAVDT_NAMES)))
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='visdrone', choices=['visdrone', 'dota', 'uavdt'])
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--name', default='v8s_official')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--cache', default='false')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    if args.data == 'dota':
        data_yaml = convert_dota()
    elif args.data == 'uavdt':
        data_yaml = convert_uavdt()
    else:
        convert()
        data_yaml = YAML
    from ultralytics import YOLO
    model = YOLO(args.model)
    # Note (Windows): cache='ram' is copied into every spawned dataloader worker -> paging-file exhaustion.
    cache = False if args.cache.lower() == 'false' else args.cache
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                deterministic=True, patience=args.patience, cache=cache, workers=args.workers,
                project=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_runs'), name=args.name, exist_ok=True)


if __name__ == '__main__':
    main()
