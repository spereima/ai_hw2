"""
Additional experiment with Faster R-CNN v2 without scale augmentation.
The goal is to compare v1, v2, and v2 with scale augmentation across all 5 folds to separately evaluate the effects of the architecture change and scale augmentation.
"""

import json
import pandas as pd
import faster_rcnn_v2
import train_all_folds

if __name__ == '__main__':
    experiment = next(experiment for experiment in faster_rcnn_v2.EXPERIMENTS if experiment['name'] == 'v2_lr_scheduler')
    rows = []
    for fold in range(5):
        result = faster_rcnn_v2.run_experiment(experiment, fold)
        official_metrics = train_all_folds.official(experiment['name'], fold)['metrics']
        v1_map = json.loads((faster_rcnn_v2.ROOT / f'faster_rcnn/lr_scheduler/fold{fold}/result.json')
                            .read_text())['metrics']['mAP']
        rows.append(dict(fold=fold, cv_mAP=result['metrics']['mAP'], v1_cv_mAP=v1_map, official_mAP=official_metrics['mAP'],
                         best_epoch=result['best_epoch']))
        print(f"v2 no scale aug fold {fold}: CV mAP {result['metrics']['mAP']:.4f} (v1 {v1_map:.4f}) | "
              f"official {official_metrics['mAP']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(faster_rcnn_v2.HERE / 'v2_lr_scheduler_5fold.csv', index=False)
