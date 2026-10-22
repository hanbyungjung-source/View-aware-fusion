# View-aware fusion for crop-based small-object detection

Code for the manuscript *"View-aware fusion for crop-based small-object detection in high-resolution drone imagery:
a controlled and pre-registered evaluation"* (Byung Jung Han; manuscript in submission).

**View-aware fusion** is a training-free merging rule for crop-based inference (uniform tiles, SAHI-style slices,
density-guided windows, packed canvases, multi-scale tiles) that runs a global view next to the local crops:

1. **Truncation-aware filtering (TAF).** A local detection that touches a crop border lying *inside* the image is
   discarded; the object is assumed to be covered by the global view or by another crop.
2. **Score-weighted box voting.** All remaining global and local boxes are merged by class-wise NMS, and each kept box
   is replaced by the score-weighted mean of the same-class candidates that overlap it (IoU &ge; 0.6).

The detector is not retrained. In this code base the rule is the suffix `trunc=drop,vote=0.6` of a method
specification (written "+ F" in the manuscript), e.g. `SAHI@512{post=NMS,metric=IOU,trunc=drop,vote=0.6}`.

The repository also contains everything needed to reproduce the controlled comparison in the manuscript: faithful
re-implementations of the baselines, the evaluation and bootstrap code, the pre-registration files of the hold-out
evaluation, and the scripts that generate the figures and tables.

## Repository layout

| Path | Content |
|---|---|
| `hahi_bench/detector.py` | Ultralytics YOLO wrapper (batched, cost counters) and a torchvision Faster R-CNN wrapper |
| `hahi_bench/thesis_methods.py` | Uniform crops (UC) and HAHI from the author's earlier thesis; HAHI-S; the view-aware fusion merge (`trunc`, `vote`) |
| `hahi_bench/baselines.py` | Global inference, multi-scale (MS), multi-scale tiling (MS-tiled), SAHI (official `sahi` library) |
| `hahi_bench/literature.py` | Re-implementations of DAHI and of PAN's crop planning (PAN's learned modules are not reproduced) |
| `hahi_bench/dyntile.py` | Re-implementation of Dynamic Tiling (tiling + flag-and-re-crop + large&ndash;small filter) |
| `hahi_bench/run_eval.py` | Runs method specifications over a dataset and stores raw detections and per-image cost |
| `hahi_bench/evaluate.py`, `metrics.py` | COCO evaluation with VisDrone ignore regions; ALL / HR / LR subsets |
| `hahi_bench/stats.py` | Paired bootstrap of AP differences (image level, or sequence level with `--cluster`) |
| `hahi_bench/prereg.py`, `holdout.py`, `prereg*.json` | Pre-registration and the frozen hold-out protocol |
| `hahi_bench/train.py`, `train_frcnn.py` | Detector training (YOLOv8n/s, YOLO11s, UAVDT, DOTA; Faster R-CNN R50-FPN v2) |
| `hahi_bench/trt_bench.py` | TensorRT FP16/INT8 export and latency measurement (optional) |
| `hahi_bench/headroom.py`, `fusion_rules.py`, `diag_*.py`, `gating.py`, `adaptive.py`, `fairness.py`, `directions.py` | Analyses reported in the manuscript and its Supporting Information |
| `hahi_bench/paper_figs.py`, `paper_rows.py`, `paper_si.py` | Figures, table rows and Supporting Information of the manuscript |
| `preregistration/` | Frozen pre-registration bundle (files, SHA-256 manifest, timeline); see its README |

## Installation

Tested with Python 3.10, CUDA 12.8 and an NVIDIA RTX 5050 (8 GB) on Windows 11.

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run every command from the repository root (`python -m hahi_bench.<module>`).

## Datasets

| Key | Dataset | Default root (override with) |
|---|---|---|
| `vd_train`, `vd_val`, `vd_testdev` | VisDrone2019-DET (official splits) | `HAHI_VISDRONE_ROOT` |
| `vid_holdout` | VisDrone2019-VID val + test-dev, every 10th frame (hold-out) | `HAHI_VID_ROOT` |
| `uavdt_test` | UAVDT-DET, 20 official test sequences, every 10th frame (hold-out) | `HAHI_UAVDT_ROOT` |
| `dota_val` | DOTA v1.0 val, oriented boxes converted to axis-aligned boxes | `HAHI_DOTA_ROOT` |
| `rf_valid`, `rf_test` | Roboflow re-split used in the thesis (only for reproducing the thesis numbers) | `HAHI_RF_ROOT` |

Expected layouts:

```
$HAHI_VISDRONE_ROOT/VisDrone2019-DET-{train,val,test-dev}/{images,annotations}
$HAHI_VID_ROOT/VisDrone2019-VID-{val,test-dev}/{sequences,annotations}
$HAHI_UAVDT_ROOT/UAV-benchmark-M/<sequence>/  and  $HAHI_UAVDT_ROOT/UAV-benchmark-MOTD_v1.0/GT/
$HAHI_DOTA_ROOT/images/val  and  $HAHI_DOTA_ROOT/labels/val_original
```

## Training the detectors

```bash
python -m hahi_bench.train --model yolov8s.pt --name v8s_official    # also yolov8n.pt / yolo11s.pt
python -m hahi_bench.train_frcnn --out <dir>                          # Faster R-CNN R50-FPN v2, 12 epochs
```

YOLO weights are written to `hahi_bench/train_runs/<name>/weights/best.pt`. Several analysis scripts expect this
location.

## Running methods

```bash
python -m hahi_bench.run_eval --data vd_testdev --weights hahi_bench/train_runs/v8s_official/weights/best.pt \
    --tag v8s --protocol standard \
    --methods "UC2x2;UC2x2{trunc=drop,vote=0.6};SAHI@512{post=NMS,metric=IOU};SAHI@512{post=NMS,metric=IOU,trunc=drop,vote=0.6}"
python -m hahi_bench.evaluate --data vd_testdev --tag v8s --protocol standard
python -m hahi_bench.stats --data vd_testdev --tag v8s --refs "<run dir of method + F>" --others "<run dir of method>"
```

A `.pth` file passed to `--weights` selects the Faster R-CNN wrapper. Raw detections are stored in
`hahi_bench/runs/<data>/<tag>/<protocol>/<method>/dets.pkl`. The `standard` protocol uses a score floor of 0.001,
at most 1000 detections per image and COCO mAP@[.5:.95] with `maxDets` = 500.

Method specifications used in the manuscript (append `trunc=drop,vote=0.6` inside the braces for "+ F"):

| Manuscript name | Specification |
|---|---|
| Global 640 / 1280 | `global@640`, `global@1280` |
| MS 640 + 1280 (+ voting) | `MS@640x1280`, `MS@640x1280{vote=0.6}` |
| MS 640 + 1280&times;736 engine + voting | `MS@640x1280{aspect=1.7778,vote=0.6}` |
| MS-tiled 960 / 1136 / 1280 / 1536 | `MSTILE@960` ... `MSTILE@1536` |
| UC (2&times;2 crops + global) | `UC2x2` |
| SAHI (official library) | `SAHI@512{post=NMS,metric=IOU}` |
| DAHI | `DAHI`, `DAHI{n_roi=2}` |
| PAN | `PAN{bandwidth=256}` |
| HAHI (thesis code) | `HAHI` |
| HAHI-S | `HAHI{weight=uniform,snap=False,upscale=none,gcc=1.0,nms_local=0.6,nms_final=0.6,fast_merge=True,nms_impl=tv}` |
| Dynamic Tiling, as published | `DYNTILE{alpha=0.002,iou=0.7,plan_conf=0.1}` |
| Dynamic Tiling, best val configuration | `DYNTILE{filt=False,plan_conf=0.1}` |
| 3&times;2 tiles + FI + F | `DYNTILE{trunc=drop,filt=False,vote=0.6}` |

The analysis scripts read results of earlier runs. In particular, `fusion_rules.py`, `diag_views.py` and the
scale-response allocation in `thesis_methods.py` need the detector's scale-response profile
`hahi_bench/runs/vd_val/v8s/standard/diag_scale.json`, written by `python -m hahi_bench.diag_scale --data vd_val`.

## Pre-registered hold-out evaluation

The protocol for VisDrone2019-VID and UAVDT (30 method configurations, confirmatory families, statistics) was
frozen in `hahi_bench/prereg.json` and `hahi_bench/prereg_amendment1.json` before any hold-out inference.

```bash
python -m hahi_bench.holdout run  --tag v8s    # inference for every pre-registered method
python -m hahi_bench.holdout eval --tag v8s    # COCO evaluation
python -m hahi_bench.holdout boot --tag v8s    # sequence-level paired bootstrap, Holm correction
```

The code was extended after the hold-out evaluation with switches that are off by default (alternative fusion
rules, the Dynamic Tiling baseline, the Faster R-CNN wrapper, environment-variable dataset roots), so the current
files no longer match the code hashes in `prereg.json`. See `preregistration/README.md` for the timeline and the
limitations of this internal registration.

## Figures, tables and Supporting Information

```bash
python -m hahi_bench.paper_figs              # Figures 1-6 -> paper/figures (PNG 300 dpi + PDF); Figure 2 runs the detector
python -m hahi_bench.paper_rows dt           # Table 4(b) rows
python -m hahi_bench.paper_rows frcnn        # Faster R-CNN rows of Table 3
python -m hahi_bench.paper_si                # Supporting Information -> paper/supplementary_en.html
```

Numbers are read from the stored runs and bootstrap files, except for the six bars of Figure 3, which are copied
from the validation analyses named in the code. `paper_si.py` additionally
reads a local results report (`hahi_bench/REPORT.md`, written by `make_tables.py` and the analysis scripts) and the
manuscript style sheet (`paper/paper.css`); neither is part of this repository.

## Weights and raw detections

Trained detector weights and the raw detections of all reported runs (including the hold-out runs) are too large for
this repository and will be deposited in a public archive; the DOI will be added here.

## Related work and attribution

- HAHI and the thesis protocol: B. J. Han, *Efficient high-resolution object detection via hybrid-aided hyper
  inference: utilizing density and dynamic packing*, Master's thesis, Hanyang University, 2026
  ([code](https://github.com/hanbyungjung-source/hahi-small-object-detection)).
- SAHI is run through the official [`sahi`](https://github.com/obss/sahi) library (v0.11.36).
- DAHI, PAN and Dynamic Tiling are independent re-implementations from the respective papers; any deviation is
  documented in the code and in the Supporting Information of the manuscript.
- Box voting follows Gidaris and Komodakis (ICCV 2015).

## Citation

If you use this code, please cite the manuscript (reference to be added on publication) and the thesis above.
