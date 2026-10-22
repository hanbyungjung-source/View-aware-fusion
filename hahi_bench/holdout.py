"""Run the pre-registered hold-out evaluation (methods / pairs frozen in prereg.json).

  python -m hahi_bench.holdout run --tag v8s          # inference for every pre-registered method
  python -m hahi_bench.holdout eval --tag v8s         # COCO evaluation
  python -m hahi_bench.holdout boot --tag v8s         # sequence-level paired bootstrap for all pairs
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
PREREG = json.load(open(os.path.join(HERE, 'prereg.json'), encoding='utf-8'))
DATA = PREREG['dataset']['key']


def safe(spec):
    import re
    return re.sub(r'[^\w@.+\-=,]', '_', spec).replace('{', '(').replace('}', ')')


def run(tag, data):
    specs = ';'.join(PREREG['methods'].values())
    w = os.path.join(HERE, PREREG['detectors'][tag]['weights'])
    subprocess.run([sys.executable, '-m', 'hahi_bench.experiments', '--group', 'none', '--extra', specs, '--data',
                    data, '--weights', w, '--tag', tag], check=True)


def evaluate(tag, data):
    subprocess.run([sys.executable, '-m', 'hahi_bench.evaluate', '--data', data, '--tag', tag, '--subsets', 'ALL'],
                   check=True)


def boot(tag, data, workers):
    M = PREREG['methods']
    pairs = PREREG['primary_families']['F1_plugin']['pairs'] + PREREG['primary_families']['F2_single_engine']['pairs'] \
        + PREREG['descriptive_pairs']['pairs']
    arg = ';'.join(f'{safe(M[a])}|{safe(M[b])}' for a, b in pairs)
    subprocess.run([sys.executable, '-m', 'hahi_bench.stats', '--data', data, '--tag', tag, '--refs', 'x', '--pairs',
                    arg, '--n', '1000', '--workers', str(workers), '--cluster'], check=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'eval', 'boot'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    {'run': lambda: run(a.tag, a.data), 'eval': lambda: evaluate(a.tag, a.data),
     'boot': lambda: boot(a.tag, a.data, a.workers)}[a.cmd]()
