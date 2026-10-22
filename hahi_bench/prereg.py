"""Write the pre-registration of the hold-out evaluation (prereg.json) BEFORE any hold-out result exists.

Freezes: dataset definition, detectors (weights + sha256), method specifications, primary comparisons with
directional hypotheses, statistics, and sha256 of the method code. Re-running refuses to overwrite.

  python -m hahi_bench.prereg
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'prereg.json')
S = 'weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv'
F = 'trunc=drop,vote=0.6'

METHODS = {
    # reference single passes
    'global@640': 'global@640',
    'global@1280': 'global@1280',
    # multi-scale, configurations selected on VisDrone-DET val (YOLOv8s) in section 16 and frozen
    'MS dyn 1280 (A budget)': 'MS@640x1280{vote=0.6}',
    'MS dyn 1536 (B budget)': 'MS@640x1536{iou=0.7,vote=0.7}',
    'MS 16:9 engine 1280x736 (A budget)': 'MS@640x1280{aspect=1.7778,vote=0.6}',
    'MS 16:9 engine 1536x864 (B budget)': 'MS@640x1536{aspect=1.7778,iou=0.7,vote=0.7}',
    'MS 3-scale dyn': 'MS@640x1280x1536{vote=0.6}',
    # crop-based methods, base and with the view-aware fusion plug-in (fixed: TAF + voting IoU 0.6)
    'UC': 'UC2x2', 'UC + F': f'UC2x2{{{F}}}',
    'SAHI': 'SAHI@512{post=NMS,metric=IOU}', 'SAHI + F': f'SAHI@512{{post=NMS,metric=IOU,{F}}}',
    'DAHI': 'DAHI', 'DAHI + F': f'DAHI{{{F}}}',
    'DAHI N_ROI=2': 'DAHI{n_roi=2}', 'DAHI N_ROI=2 + F': f'DAHI{{n_roi=2,{F}}}',
    'PAN': 'PAN{bandwidth=256}', 'PAN + F': f'PAN{{bandwidth=256,{F}}}',
    'MS-tiled 960': 'MSTILE@960', 'MS-tiled 960 + F': f'MSTILE@960{{{F}}}',
    'MS-tiled 1136': 'MSTILE@1136', 'MS-tiled 1136 + F': f'MSTILE@1136{{{F}}}',
    'MS-tiled 1280': 'MSTILE@1280', 'MS-tiled 1280 + F': f'MSTILE@1280{{{F}}}',
    'MS-tiled 1536': 'MSTILE@1536', 'MS-tiled 1536 + F': f'MSTILE@1536{{{F}}}',
    'HAHI (thesis code)': 'HAHI',
    'HAHI-S': f'HAHI{{{S}}}', 'HAHI-S + F': f'HAHI{{{S},{F}}}',
    'HAHI-S route 0.1': f'HAHI{{{S},route_ratio=0.1}}', 'HAHI-S route 0.1 + F': f'HAHI{{{S},{F},route_ratio=0.1}}',
}
PLUGIN_PAIRS = [('UC + F', 'UC'), ('SAHI + F', 'SAHI'), ('DAHI + F', 'DAHI'), ('DAHI N_ROI=2 + F', 'DAHI N_ROI=2'),
                ('PAN + F', 'PAN'), ('MS-tiled 960 + F', 'MS-tiled 960'), ('MS-tiled 1136 + F', 'MS-tiled 1136'),
                ('MS-tiled 1280 + F', 'MS-tiled 1280'), ('MS-tiled 1536 + F', 'MS-tiled 1536'),
                ('HAHI-S + F', 'HAHI-S'), ('HAHI-S route 0.1 + F', 'HAHI-S route 0.1')]
ENGINE_PAIRS = [('HAHI-S + F', 'MS-tiled 1136 + F'), ('HAHI-S route 0.1 + F', 'MS-tiled 1280 + F')]
DESCRIPTIVE_PAIRS = [('HAHI-S + F', 'MS-tiled 1280 + F'), ('HAHI-S route 0.1 + F', 'MS-tiled 1536 + F'),
                     ('HAHI-S + F', 'MS dyn 1280 (A budget)'), ('HAHI-S route 0.1 + F', 'MS dyn 1536 (B budget)'),
                     ('HAHI-S + F', 'MS 16:9 engine 1280x736 (A budget)'),
                     ('HAHI-S route 0.1 + F', 'MS 16:9 engine 1536x864 (B budget)'),
                     ('HAHI-S route 0.1 + F', 'MS 3-scale dyn')]
WEIGHTS = {'v8s': 'train_runs/v8s_official/weights/best.pt', 'v8n': 'train_runs/v8n_official/weights/best.pt',
           'y11s': 'train_runs/y11s_official/weights/best.pt'}
CODE = ['thesis_methods.py', 'baselines.py', 'literature.py', 'detector.py', 'run_eval.py', 'metrics.py', 'data.py',
        'stats.py']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} exists; pre-registration is immutable')
    doc = {
        'created': datetime.datetime.now().astimezone().isoformat(),
        'status': 'registered before any hold-out inference',
        'dataset': {'key': 'vid_holdout', 'definition': 'VisDrone2019-VID val + test-dev (24 sequences), every 10th '
                    'frame; same category / ignore-region handling as VisDrone-DET; never used for development',
                    'second_holdout': 'UAVDT (if a detector trained on its train split is available); same methods, '
                    'pairs and hypotheses, single run per detector'},
        'detectors': {k: {'weights': v, 'sha256': sha(os.path.join(HERE, v))} for k, v in WEIGHTS.items()},
        'methods': METHODS,
        'protocol': 'standard (score floor 0.001, max 1000 det/img, mAP@[.5:.95] at maxDets 500)',
        'statistics': 'paired sequence-level (cluster) bootstrap, 1000 resamples, seed 0; one-sided P(delta<=0); '
                      'Holm correction within each primary family',
        'primary_families': {
            'F1_plugin': {'pairs': PLUGIN_PAIRS, 'detectors': list(WEIGHTS), 'hypothesis': 'mAP(method + F) > '
                          'mAP(method) for every crop-based method and detector'},
            'F2_single_engine': {'pairs': ENGINE_PAIRS, 'detectors': list(WEIGHTS), 'hypothesis': 'with only a '
                                 '640x640 engine, HAHI-S + F exceeds MS-tiled + F at similar or more passes'}},
        'descriptive_pairs': {'pairs': DESCRIPTIVE_PAIRS, 'expectation': 'no directional hypothesis (sections 16.2 and '
                              '16.6 predict ties against MS with dynamic / aspect-matched engines; HAHI-S + F ~ '
                              'MS-tiled with 30-40% more passes)'},
        'reporting': 'all pre-registered results are reported regardless of outcome; no configuration is changed '
                     'after hold-out inference starts',
        'code_sha256': {c: sha(os.path.join(HERE, c)) for c in CODE},
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print('written', OUT, doc['created'])


if __name__ == '__main__':
    main()
