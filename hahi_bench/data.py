import os
import pickle
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

VISDRONE_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
                  'awning-tricycle', 'bus', 'motor']
ROBOFLOW_NAMES = ['Awning-tricycle', 'Bicycle', 'Bus', 'Car', 'Motor', 'Pedestrian', 'People',
                  'Tricycle', 'Truck', 'Van']

DOTA_NAMES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
              'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
              'roundabout', 'soccer ball field', 'swimming pool']

# Dataset roots: defaults are the paths used in the study; override with environment variables on other machines.
VISDRONE_ROOT = os.environ.get('HAHI_VISDRONE_ROOT', 'F:/datasets/VisDrone2019')
DOTA_ROOT = os.environ.get('HAHI_DOTA_ROOT', 'F:/datasets/DOTA/DOTAv1')
VID_ROOT = os.environ.get('HAHI_VID_ROOT', 'G:/datasets/VisDrone2019-VID')
UAVDT_ROOT = os.environ.get('HAHI_UAVDT_ROOT', 'G:/datasets/UAVDT')
UAVDT_NAMES = ['car', 'truck', 'bus']
UAVDT_TEST = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801', 'M0802',
              'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

DATASETS = {
    # Roboflow re-split used in the thesis (train/valid share video sequences -> leakage).
    'rf_valid': ('yolo', os.environ.get('HAHI_RF_ROOT', 'F:/notebook/detection/data') + '/valid', ROBOFLOW_NAMES),
    'rf_test': ('yolo', os.environ.get('HAHI_RF_ROOT', 'F:/notebook/detection/data') + '/test', ROBOFLOW_NAMES),
    # Official VisDrone2019-DET splits.
    'vd_train': ('visdrone', f'{VISDRONE_ROOT}/VisDrone2019-DET-train', VISDRONE_NAMES),
    'vd_val': ('visdrone', f'{VISDRONE_ROOT}/VisDrone2019-DET-val', VISDRONE_NAMES),
    'vd_testdev': ('visdrone', f'{VISDRONE_ROOT}/VisDrone2019-DET-test-dev', VISDRONE_NAMES),
    # DOTA v1.0 (large aerial scenes up to ~4000 px); OBB -> axis-aligned boxes. val is used as test set.
    'dota_val': ('dota', f'{DOTA_ROOT}/images/val', DOTA_NAMES),
    # Hold-out (never used for development): VisDrone2019-VID val + test-dev, every 10th frame of each sequence.
    'vid_holdout': ('visdrone_vid', [f'{VID_ROOT}/VisDrone2019-VID-val', f'{VID_ROOT}/VisDrone2019-VID-test-dev'],
                    VISDRONE_NAMES),
    # Second hold-out: UAVDT-DET official test split (20 sequences), every 10th frame; detector trained on its
    # train split (30 sequences).
    'uavdt_test': ('uavdt', UAVDT_TEST, UAVDT_NAMES),
}
VID_STRIDE = 10

HR_THRESHOLD = 1920 * 1080


@dataclass
class Record:
    name: str
    path: str
    w: int
    h: int
    boxes: np.ndarray            # (N, 4) xyxy float32, evaluated objects
    cls: np.ndarray              # (N,) int64
    ignore: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # (M, 4) xyxy

    @property
    def is_hr(self):
        return self.w * self.h >= HR_THRESHOLD


def _load_yolo(root):
    img_dir, lbl_dir = os.path.join(root, 'images'), os.path.join(root, 'labels')
    recs = []
    for name in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, name)
        with Image.open(path) as im:
            w, h = im.size
        boxes, cls = [], []
        lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + '.txt')
        if os.path.exists(lbl):
            with open(lbl) as f:
                for line in f:
                    p = line.split()
                    if len(p) != 5:
                        continue
                    c, xc, yc, bw, bh = map(float, p)
                    boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
                    cls.append(int(c))
        recs.append(Record(name, path, w, h, np.asarray(boxes, np.float32).reshape(-1, 4),
                           np.asarray(cls, np.int64)))
    return recs


def _load_visdrone(root):
    """Official VisDrone-DET annotation: x,y,w,h,score,category,truncation,occlusion.

    category 0 = ignored region, 11 = others; score 0 = not evaluated. All of these are treated as
    ignore regions following the official toolkit (objects/detections mostly inside them are dropped).
    """
    img_dir, ann_dir = os.path.join(root, 'images'), os.path.join(root, 'annotations')
    recs = []
    for name in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, name)
        with Image.open(path) as im:
            w, h = im.size
        boxes, cls, ign = [], [], []
        with open(os.path.join(ann_dir, os.path.splitext(name)[0] + '.txt')) as f:
            for line in f:
                p = [x for x in line.strip().split(',') if x != '']
                if len(p) < 6:
                    continue
                x, y, bw, bh, score, cat = map(int, p[:6])
                b = [x, y, x + bw, y + bh]
                if score == 0 or cat == 0 or cat == 11:
                    ign.append(b)
                else:
                    boxes.append(b)
                    cls.append(cat - 1)
        recs.append(Record(name, path, w, h, np.asarray(boxes, np.float32).reshape(-1, 4),
                           np.asarray(cls, np.int64), np.asarray(ign, np.float32).reshape(-1, 4)))
    return recs


def _load_dota(img_dir):
    """DOTA original labels (x1 y1 ... x4 y4 class difficult) -> axis-aligned xyxy boxes.

    Following the DOTA protocol, 'difficult' objects are not evaluated (treated as ignore regions).
    """
    lbl_dir = img_dir.replace('/images/', '/labels/') + '_original'
    name2id = {n.replace(' ', '-'): i for i, n in enumerate(DOTA_NAMES)}
    recs = []
    for name in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, name)
        with Image.open(path) as im:
            w, h = im.size
        boxes, cls, ign = [], [], []
        lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + '.txt')
        if os.path.exists(lbl):
            with open(lbl) as f:
                for line in f:
                    p = line.split()
                    if len(p) < 9 or p[8] not in name2id:
                        continue
                    pts = np.asarray(p[:8], np.float32).reshape(4, 2)
                    x1, y1 = pts.min(0)
                    x2, y2 = pts.max(0)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if len(p) > 9 and p[9] == '1':
                        ign.append([x1, y1, x2, y2])
                    else:
                        boxes.append([x1, y1, x2, y2])
                        cls.append(name2id[p[8]])
        recs.append(Record(name, path, w, h, np.asarray(boxes, np.float32).reshape(-1, 4),
                           np.asarray(cls, np.int64), np.asarray(ign, np.float32).reshape(-1, 4)))
    return recs


def _load_visdrone_vid(roots):
    """VisDrone-VID: sequences/<seq>/<frame>.jpg + annotations/<seq>.txt with
    frame,target_id,x,y,w,h,score,category,truncation,occlusion (same category / ignore semantics as DET).
    Every VID_STRIDE-th frame is used; record names are '<seq>/<frame>' (sequence = bootstrap cluster)."""
    recs = []
    for root in roots:
        seq_dir, ann_dir = os.path.join(root, 'sequences'), os.path.join(root, 'annotations')
        for seq in sorted(os.listdir(seq_dir)):
            ann = {}
            with open(os.path.join(ann_dir, seq + '.txt')) as f:
                for line in f:
                    p = [x for x in line.strip().split(',') if x != '']
                    if len(p) < 8:
                        continue
                    fr, _, x, y, bw, bh, score, cat = map(int, p[:8])
                    ann.setdefault(fr, []).append((x, y, bw, bh, score, cat))
            frames = sorted(os.listdir(os.path.join(seq_dir, seq)))
            for k, name in enumerate(frames):
                if k % VID_STRIDE:
                    continue
                fr = int(os.path.splitext(name)[0])
                path = os.path.join(seq_dir, seq, name)
                with Image.open(path) as im:
                    w, h = im.size
                boxes, cls, ign = [], [], []
                for x, y, bw, bh, score, cat in ann.get(fr, []):
                    b = [x, y, x + bw, y + bh]
                    if score == 0 or cat == 0 or cat == 11:
                        ign.append(b)
                    else:
                        boxes.append(b)
                        cls.append(cat - 1)
                recs.append(Record(f'{seq}/{name}', path, w, h, np.asarray(boxes, np.float32).reshape(-1, 4),
                                   np.asarray(cls, np.int64), np.asarray(ign, np.float32).reshape(-1, 4)))
    return recs


def read_uavdt_gt(seq):
    """UAVDT GT: frame,target_id,x,y,w,h,out_of_view,occlusion,category (1 car, 2 truck, 3 bus) in *_gt_whole.txt;
    ignore regions in *_gt_ignore.txt. Returns {frame: (boxes xyxy, cls 0..2)}, {frame: ignore boxes}."""
    gt_dir = os.path.join(UAVDT_ROOT, 'UAV-benchmark-MOTD_v1.0', 'GT')
    obj, ign = {}, {}
    for fn, dst in ((f'{seq}_gt_whole.txt', obj), (f'{seq}_gt_ignore.txt', ign)):
        p = os.path.join(gt_dir, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                v = [x for x in line.strip().split(',') if x != '']
                if len(v) < 9:
                    continue
                fr, x, y, bw, bh, cat = int(v[0]), float(v[2]), float(v[3]), float(v[4]), float(v[5]), int(v[8])
                dst.setdefault(fr, []).append((x, y, x + bw, y + bh, cat))
    return obj, ign


def _load_uavdt(seqs, stride=None):
    stride = stride or VID_STRIDE
    recs = []
    for seq in seqs:
        obj, ign = read_uavdt_gt(seq)
        img_dir = os.path.join(UAVDT_ROOT, 'UAV-benchmark-M', seq)
        frames = sorted(os.listdir(img_dir))
        for k, name in enumerate(frames):
            if k % stride:
                continue
            fr = int(''.join(ch for ch in name if ch.isdigit()))
            path = os.path.join(img_dir, name)
            with Image.open(path) as im:
                w, h = im.size
            b = [o for o in obj.get(fr, []) if o[4] in (1, 2, 3)]
            recs.append(Record(f'{seq}/{name}', path, w, h,
                               np.asarray([o[:4] for o in b], np.float32).reshape(-1, 4),
                               np.asarray([o[4] - 1 for o in b], np.int64),
                               np.asarray([g[:4] for g in ign.get(fr, [])], np.float32).reshape(-1, 4)))
    return recs


def load_dataset(key):
    kind, root, names = DATASETS[key]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{key}.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f), names
    loader = {'yolo': _load_yolo, 'visdrone': _load_visdrone, 'dota': _load_dota,
              'visdrone_vid': _load_visdrone_vid, 'uavdt': _load_uavdt}[kind]
    recs = loader(root)
    with open(cache, 'wb') as f:
        pickle.dump(recs, f)
    return recs, names
