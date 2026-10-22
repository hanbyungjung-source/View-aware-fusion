"""GPU version of the bootstrap replicate computation of stats._reps (same resamples, same cache files).

stats._reps re-accumulates COCO AP for every replicate on the CPU (BootstrapTable.per_cat_tp). This module does the same
arithmetic for a batch of replicates at once with torch: integer cumulative sums of weighted TP / FP flags, precision
and recall in float64 at TP columns, the suffix maximum of precision and the 101-point recall lookup. The image
weights are drawn with the same generator calls in the same order as stats._reps, so the replicates are identical and
the result is written to the same cache file name, which stats.py then loads.

  python -m hahi_bench.boot_gpu --data vid_holdout --tag v8n --n 10000 --cluster [--check 200]
"""
import argparse
import os

import numpy as np
import torch

from .data import load_dataset
from .stats import tables_for

EPS = np.spacing(1)


def weights(ta, recs, n, seed, cluster):
    """Replicate weights (n, n_img) drawn exactly as in stats._reps."""
    n_img = len(ta.img_pos)
    rng = np.random.default_rng(seed)
    W = np.empty((n, n_img), np.int64)
    if cluster:
        pos2img = sorted(ta.img_pos, key=ta.img_pos.get)
        groups = [recs[i].name.split('/')[0] for i in pos2img]
        uniq = sorted(set(groups))
        gidx = np.asarray([uniq.index(g) for g in groups])
    for k in range(n):
        if cluster:
            cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            W[k] = cnt[gidx]
        else:
            W[k] = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
    return W


def table_ap(table, W, dev, budget=3e7):
    """(R, n_iou) mean-over-categories AP for each replicate row of W (int64, R x n_img)."""
    R = W.shape[0]
    thr = torch.as_tensor(table.rec_thrs, dtype=torch.float64, device=dev)
    tot = torch.zeros(R, table.n_iou, dtype=torch.float64, device=dev)
    cnt = torch.zeros(R, dtype=torch.float64, device=dev)
    Wd = torch.as_tensor(W, device=dev)
    for img, tp, fp, npig in table.cats:
        n_pos = Wd.double() @ torch.as_tensor(npig, dtype=torch.float64, device=dev)
        inc = n_pos > 0
        cnt += inc.double()
        if not tp.shape[1]:
            continue
        img_t = torch.as_tensor(img, device=dev)
        N = tp.shape[1]
        step = max(1, int(budget // N))
        for r0 in range(0, R, step):
            r1 = min(R, r0 + step)
            wd = Wd[r0:r1][:, img_t]
            npos = n_pos[r0:r1, None]
            for t in range(table.n_iou):
                cols = np.flatnonzero(tp[t])
                if not len(cols):
                    continue
                c_t = torch.as_tensor(cols, device=dev)
                a = torch.cumsum(wd * torch.as_tensor(tp[t], device=dev), 1)[:, c_t].double()
                b = torch.cumsum(wd * torch.as_tensor(fp[t], device=dev), 1)[:, c_t].double()
                rc = a / npos
                pr = a / (a + b + EPS)
                pr = torch.flip(torch.cummax(torch.flip(pr, [1]), 1).values, [1])
                inds = torch.searchsorted(rc.contiguous(), thr.expand(r1 - r0, -1).contiguous(), right=False)
                valid = inds < len(cols)
                q = torch.where(valid, torch.gather(pr, 1, inds.clamp(max=len(cols) - 1)), torch.zeros((), dtype=pr.dtype, device=dev))
                tot[r0:r1, t] += torch.where(inc[r0:r1], q.mean(1), torch.zeros((), dtype=q.dtype, device=dev))
    return (tot / cnt[:, None]).cpu().numpy()


def reps(run_dir, data, subset='ALL', thr=0.0, n=1000, seed=0, cluster=False, check=0, batch=2000):
    cache = os.path.join(run_dir, f'boot_reps_{subset}_thr{thr:g}_n{n}_s{seed}{"_seq" if cluster else ""}.npy')
    if not check and os.path.exists(cache):
        return np.load(cache)
    recs, names = load_dataset(data)
    ta, ts = tables_for(run_dir, recs, subset, thr, len(names))
    W = weights(ta, recs, n, seed, cluster)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = check or n
    out = np.empty((m + 1, 3))
    a0, s0 = ta.per_cat_tp(), ts.per_cat_tp()
    out[0] = a0.mean(), a0[:, 0].mean(), s0.mean()
    for b0 in range(0, m, batch):
        b1 = min(m, b0 + batch)
        A, S = table_ap(ta, W[b0:b1], dev), table_ap(ts, W[b0:b1], dev)
        out[1 + b0:1 + b1, 0] = A.mean(1)
        out[1 + b0:1 + b1, 1] = A[:, 0]
        out[1 + b0:1 + b1, 2] = S.mean(1)
    if check:
        ref = np.load(cache)[:m + 1]
        print(f'{os.path.basename(run_dir)[:50]}: max |GPU - CPU| over {m} replicates = {np.abs(out - ref).max():.2e}', flush=True)
        return out
    if not os.path.exists(cache):
        np.save(cache, out)
    return out


def main():
    from .run_eval import RUN_ROOT
    from .holdout import PREREG, safe
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--cluster', action='store_true')
    ap.add_argument('--check', type=int, default=0, help='compare the first CHECK replicates with an existing CPU cache')
    ap.add_argument('--only', default='', help='semicolon-separated run dir names (default: all confirmatory runs)')
    a = ap.parse_args()
    M = PREREG['methods']
    pairs = PREREG['primary_families']['F1_plugin']['pairs'] + PREREG['primary_families']['F2_single_engine']['pairs']
    runs = sorted({safe(M[x]) for p in pairs for x in p})
    if a.only:
        runs = [r for r in runs if r in a.only.split(';')]
    root = os.path.join(RUN_ROOT, a.data, a.tag, 'standard')
    for r in runs:
        reps(os.path.join(root, r), a.data, n=a.n, cluster=a.cluster, check=a.check)
        if not a.check:
            print('done', r[:60], flush=True)


if __name__ == '__main__':
    main()
