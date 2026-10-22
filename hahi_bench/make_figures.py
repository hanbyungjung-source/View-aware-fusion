"""Generate report figures."""
import os

from .plots import pareto, sensitivity

FIG = os.path.join(os.path.dirname(__file__), 'figures')

ITEMS = [
    ('Global640', 'global@640', 'o', '#7f7f7f'),
    ('Global1280', 'global@1280', 'o', '#1f77b4'),
    ('MS640+1280', 'MS@640x1280', 's', '#1f77b4'),
    ('MS640+1536', 'MS@640x1536', 's', '#17becf'),
    ('MS640+1280+1536', 'MS@640x1280x1536', 's', '#0b3d91'),
    ('UC2x2', 'UC2x2', 'v', '#8c564b'),
    ('SAHI512', 'SAHI@512_post=NMS,metric=IOU_', 'v', '#e377c2'),
    ('DAHI', 'DAHI', '^', '#bcbd22'),
    ('DAHI N_ROI=2', 'DAHI_n_roi=2_', '^', '#dbdb8d'),
    ('PAN', 'PAN_bandwidth=256_', 'D', '#9467bd'),
    ('HAHI', 'HAHI', '*', '#d62728'),
    ('HAHI tuned', 'HAHI_gcc=1.0,route_ratio=0.1_', '*', '#ff7f0e'),
    ('HAHI-ext', 'HAHI_global_imgsz=1280,gcc=1.0_', 'P', '#2ca02c'),
]


def main():
    os.makedirs(FIG, exist_ok=True)
    pareto('vd_testdev', 'v8s', 'standard', ITEMS, os.path.join(FIG, 'pareto_testdev.png'),
           title='VisDrone test-dev (YOLOv8s): accuracy vs. cost',
           offsets={'MS640+1280': (-8, 4, 'right'), 'MS640+1536': (0, 9, 'center'), 'HAHI tuned': (6, -12, 'left'),
                    'Global1280': (6, -12, 'left')})
    pareto('dota_val', 'v8s_dota', 'standard', ITEMS, os.path.join(FIG, 'pareto_dota.png'),
           title='DOTA v1.0 val (YOLOv8s): accuracy vs. cost (latency/VRAM: methods with dedicated timing runs)',
           offsets={'MS640+1280': (-8, 4, 'right'), 'HAHI tuned': (-8, 6, 'right'), 'HAHI': (-8, -12, 'right'),
                    'PAN': (8, -4, 'left'), 'DAHI N_ROI=2': (6, 6, 'left'), 'Global640': (6, -12, 'left'),
                    'UC2x2': (6, -12, 'left')})
    groups = {
        'route_ratio': [(0.1, 'HAHI_route_ratio=0.1_'), (0.2, 'HAHI_route_ratio=0.2_'), (0.3, 'HAHI'),
                        (0.4, 'HAHI_route_ratio=0.4_'), (0.5, 'HAHI_route_ratio=0.5_')],
        'window (px)': [(384, 'HAHI_window=384_'), (512, 'HAHI'), (640, 'HAHI_window=640_'), (768, 'HAHI_window=768_')],
        'sigma = window / k': [(2, 'HAHI_sigma_div=2.0_'), (4, 'HAHI'), (8, 'HAHI_sigma_div=8.0_')],
        'upscale ratio': [(1.0, 'HAHI_upscale=none_'), (1.25, 'HAHI_upscale_ratio=1.25_'), (1.5, 'HAHI'),
                          (2.0, 'HAHI_upscale_ratio=2.0_')],
        'ROI conf filter': [(0.05, 'HAHI_conf_filter=0.05_'), (0.1, 'HAHI'), (0.2, 'HAHI_conf_filter=0.2_')],
    }
    sensitivity('vd_val', 'v8s', 'standard', groups, os.path.join(FIG, 'sensitivity_val.png'))
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
