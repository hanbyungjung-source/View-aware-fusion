"""COCO-style evaluation with VisDrone ignore-region handling and paired image-level bootstrap."""
import contextlib
import io

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = [1, 100, 500]          # VisDrone protocol uses up to 500 detections per image
AREA_NAMES = ['all', 'small', 'medium', 'large']


def max_dets_for(n_cls):
    """DOTA (15 classes, up to ~1.4k objects/img) is evaluated without an effective per-image cap."""
    return [1, 100, 2000] if n_cls == 15 else MAX_DETS


def _ignored_fraction(boxes, ignore, w, h):
    """Fraction of each box area covered by the union of ignore regions (as in the VisDrone toolkit)."""
    if len(boxes) == 0 or len(ignore) == 0:
        return np.zeros(len(boxes), np.float32)
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in ignore.round().astype(int):
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
    ii = cv2.integral(mask)  # (h+1, w+1)
    b = boxes.copy()
    b[:, [0, 2]] = np.clip(np.round(b[:, [0, 2]]), 0, w)
    b[:, [1, 3]] = np.clip(np.round(b[:, [1, 3]]), 0, h)
    b = b.astype(int)
    inside = ii[b[:, 3], b[:, 2]] - ii[b[:, 1], b[:, 2]] - ii[b[:, 3], b[:, 0]] + ii[b[:, 1], b[:, 0]]
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1)
    return inside / area


def build_coco(records, dets, indices, score_thr=0.0, use_ignore=True, n_cls=10):
    gt = {'images': [], 'annotations': [], 'categories': [{'id': i, 'name': str(i)} for i in range(n_cls)]}
    dt = []
    aid = 1
    for i in indices:
        r = records[i]
        gt['images'].append({'id': int(i), 'width': r.w, 'height': r.h})
        gb, gc = r.boxes, r.cls
        db, ds, dc = dets[i]
        keep = ds >= score_thr
        db, ds, dc = db[keep], ds[keep], dc[keep]
        if use_ignore and len(r.ignore):
            kg = _ignored_fraction(gb, r.ignore, r.w, r.h) <= 0.5
            gb, gc = gb[kg], gc[kg]
            kd = _ignored_fraction(db, r.ignore, r.w, r.h) <= 0.5
            db, ds, dc = db[kd], ds[kd], dc[kd]
        for (x1, y1, x2, y2), c in zip(gb.tolist(), gc.tolist()):
            gt['annotations'].append({'id': aid, 'image_id': int(i), 'category_id': int(c),
                                      'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
            aid += 1
        for (x1, y1, x2, y2), s, c in zip(db.tolist(), ds.tolist(), dc.tolist()):
            dt.append({'image_id': int(i), 'category_id': int(c), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s})
    return gt, dt


def coco_eval(records, dets, indices, score_thr=0.0, use_ignore=True, n_cls=10, keep_eval=False):
    indices = sorted(int(i) for i in indices)
    gt, dt = build_coco(records, dets, indices, score_thr, use_ignore, n_cls)
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        if not dt:
            return {k: 0.0 for k in ['mAP', 'mAP@100', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR500']}, None
        cd = cg.loadRes(dt)
        E = COCOeval(cg, cd, 'bbox')
        E.params.imgIds = indices
        E.params.maxDets = max_dets_for(n_cls)
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats
    # pycocotools' stats[0] always uses maxDets=100 regardless of params.maxDets, while AP50/APs/APm/APl use the
    # last maxDets entry. VisDrone evaluates AP with up to 500 detections, so mAP is recomputed at maxDets[-1].
    prec = E.eval['precision'][:, :, :, 0, -1]
    m_last = float(prec[prec > -1].mean()) if (prec > -1).any() else 0.0
    out = {'mAP': m_last, 'mAP@100': s[0], 'AP50': s[1], 'AP75': s[2], 'APs': s[3], 'APm': s[4], 'APl': s[5],
           'AR500': s[8]}
    out = {k: float(v) for k, v in out.items()}
    return out, (E if keep_eval else None)


class BootstrapTable:
    """Flattened per-image matching results of one COCOeval run, enabling fast weighted re-accumulation.

    Re-implements COCOeval.accumulate for maxDet=500 and arbitrary per-image weights (bootstrap counts).
    """

    def __init__(self, E, area='all', max_det=None):
        p = E.params
        max_det = max_det or p.maxDets[-1]
        a_idx = p.areaRngLbl.index(area)
        n_img = len(p.imgIds)
        n_area = len(p.areaRng)
        self.img_pos = {img: k for k, img in enumerate(p.imgIds)}
        self.rec_thrs = p.recThrs
        self.n_iou = len(p.iouThrs)
        self.cats = []
        for k, _ in enumerate(p.catIds):
            scores, img, tp, fp = [], [], [], []
            npig = np.zeros(n_img)
            for i in range(n_img):
                e = E.evalImgs[k * n_area * n_img + a_idx * n_img + i]
                if e is None:
                    continue
                ds = np.asarray(e['dtScores'][:max_det])
                dm = e['dtMatches'][:, :max_det]
                di = e['dtIgnore'][:, :max_det]
                gi = e['gtIgnore'].astype(bool)
                npig[i] = np.count_nonzero(~gi)
                if ds.size:
                    scores.append(ds)
                    img.append(np.full(ds.size, i))
                    tp.append(np.logical_and(dm, np.logical_not(di)))
                    fp.append(np.logical_and(np.logical_not(dm), np.logical_not(di)))
            if scores:
                sc = np.concatenate(scores)
                order = np.argsort(-sc, kind='mergesort')
                im_, tp_, fp_ = np.concatenate(img)[order], np.concatenate(tp, 1)[:, order], np.concatenate(fp, 1)[:, order]
                # Lossless pruning for fast re-accumulation: (a) detections ignored at every IoU add zero to both
                # cumulative sums, (b) detections after the last true positive (any IoU) are pure false positives
                # that cannot raise the interpolated precision at any reachable recall level.
                keep = (tp_ | fp_).any(0)
                anytp = np.where(tp_.any(0))[0]
                keep[(anytp[-1] + 1 if len(anytp) else 0):] = False
                self.cats.append((im_[keep], tp_[:, keep], fp_[:, keep], npig))
            else:
                self.cats.append((np.zeros(0, int), np.zeros((self.n_iou, 0), bool), np.zeros((self.n_iou, 0), bool), npig))

    def per_cat(self, weights=None, fast=False):
        """(n_valid_cats, n_iou) AP matrix for the given image weights (exact COCOeval.accumulate semantics).

        fast=True accumulates in float32 (integer bootstrap counts stay exact below 2^24) for resampling."""
        acc = np.float32 if fast else np.float64
        rows = []
        for img, tp, fp, npig in self.cats:
            w = np.ones_like(npig) if weights is None else weights
            n_pos = float(np.dot(w, npig))
            if n_pos == 0:
                continue
            q = np.zeros((self.n_iou, len(self.rec_thrs)))
            if tp.shape[1]:
                wd = w[img].astype(np.float32)
                tps = np.cumsum(tp * wd, axis=1, dtype=acc)
                fps = np.cumsum(fp * wd, axis=1, dtype=acc)
                rc = tps / n_pos
                pr = tps / np.maximum(tps + fps, np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                for t in range(self.n_iou):
                    inds = np.searchsorted(rc[t], self.rec_thrs, side='left')
                    valid = inds < pr.shape[1]
                    q[t, valid] = pr[t, inds[valid]]
            rows.append(q.mean(1))
        return np.asarray(rows).reshape(-1, self.n_iou)

    def per_cat_tp(self, weights=None):
        """Same result as per_cat (float64, pycocotools semantics) but evaluated only at true-positive columns.

        The interpolated precision max over any suffix starting at a recall threshold is attained at a weighted
        TP column (FP columns only lower precision, zero-weight columns leave it unchanged), and searchsorted
        on recall always lands on such a column, so restricting to TP columns is exact."""
        rows = []
        for img, tp, fp, npig in self.cats:
            w = np.ones_like(npig) if weights is None else weights
            n_pos = float(np.dot(w, npig))
            if n_pos == 0:
                continue
            q = np.zeros((self.n_iou, len(self.rec_thrs)))
            if tp.shape[1]:
                wd = w[img].astype(np.int64) if weights is not None else np.ones(len(img), np.int64)
                pos = wd > 0
                for t in range(self.n_iou):
                    ctp = np.cumsum(tp[t] * wd)
                    cfp = np.cumsum(fp[t] * wd)
                    P = np.flatnonzero(tp[t] & pos)
                    if not len(P):
                        continue
                    a, b = ctp[P].astype(np.float64), cfp[P].astype(np.float64)
                    rc = a / n_pos
                    pr = a / (a + b + np.spacing(1))
                    pr = np.maximum.accumulate(pr[::-1])[::-1]
                    inds = np.searchsorted(rc, self.rec_thrs, side='left')
                    valid = inds < len(P)
                    q[t, valid] = pr[inds[valid]]
            rows.append(q.mean(1))
        return np.asarray(rows).reshape(-1, self.n_iou)

    def ap(self, weights=None, iou_idx=None):
        """Mean AP over categories (and IoU thresholds unless iou_idx given) for image weights."""
        m = self.per_cat(weights)
        if not len(m):
            return 0.0
        return float(m[:, iou_idx].mean() if iou_idx is not None else m.mean())


def paired_bootstrap(table_a, table_b, n_boot=1000, seed=0):
    """Delta (B - A) for mAP and AP50 with 95% percentile CI and P(delta <= 0) over paired image resamples."""
    n = len(table_a.img_pos)
    rng = np.random.default_rng(seed)
    a0, b0 = table_a.per_cat_tp(), table_b.per_cat_tp()
    base = {'mAP': b0.mean() - a0.mean(), 'AP50': b0[:, 0].mean() - a0[:, 0].mean()}
    d = {'mAP': np.empty(n_boot), 'AP50': np.empty(n_boot)}
    for k in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        a, b = table_a.per_cat_tp(w), table_b.per_cat_tp(w)
        d['mAP'][k] = b.mean() - a.mean()
        d['AP50'][k] = b[:, 0].mean() - a[:, 0].mean()
    out = {}
    for key in d:
        lo, hi = np.percentile(d[key], [2.5, 97.5])
        out[key] = {'delta': float(base[key]), 'ci_lo': float(lo), 'ci_hi': float(hi),
                    'p_le_0': float(np.mean(d[key] <= 0))}
    return out
