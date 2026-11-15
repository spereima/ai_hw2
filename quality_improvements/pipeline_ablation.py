"""
Analysis of how different improvements affect Faster R-CNN performance on the official validation set.
The comparison includes Faster R-CNN v1 and v2 with scale augmentation, followed by the gradual addition of:
1. An ensemble of 5 checkpoints using WBF.
2. TTA with 6 image views.
3. Temporal post-processing.
4. Fusion with FCOS.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parent
for import_dir in (ROOT / 'shared', ROOT / 'data_temporality', ROOT / 'fcos/inference',
                   ROOT / 'faster_rcnn/faster_rcnn_v2/inference'):
    sys.path.insert(0, str(import_dir))
import experiment_utils
import temporal_postprocessing
import detector_fusion

FASTER_RCNN_V1_CACHE = ROOT / 'faster_rcnn/official_validation/cache'
V2_VIEWS = ROOT / 'quality_improvements/faster_rcnn_v2_ensemble/cache/views/val'


def native_sets(model, ids):
    """Cached native-resize (none@1.0) predictions of the 5 fold checkpoints of `model` ('v1' or 'v2'), in the order of `ids`."""
    if model == 'v1':
        prediction_sets = []
        for fold in range(5):
            cached_npz = np.load(FASTER_RCNN_V1_CACHE / f'fold{fold}_none@1.0.npz')
            assert cached_npz['ids'].astype(str).tolist() == ids
            prediction_sets.append(experiment_utils.unpack(cached_npz, 'full_'))
        return prediction_sets
    return [detector_fusion._chunks(V2_VIEWS / f'fold{fold}', 'none@1.0', ids) for fold in range(5)]


if __name__ == '__main__':
    meta = experiment_utils.frames_meta('val')
    ids = meta.image_id.astype(str).tolist()
    gt = temporal_postprocessing.coco_gt(experiment_utils.annotations_for('val'), meta.image_id.tolist())
    groups = {'all': meta.image_id.tolist(),
              **{f'video_{video}': meta.image_id[meta.video == video].tolist() for video in meta.video.unique()}}
    rows = []
    v1_raw = json.loads((ROOT / 'fcos/inference/official_validation_raw.json').read_text())['results']
    v2_raw = json.loads((ROOT / 'quality_improvements/faster_rcnn_v2_ensemble/official_validation_raw.json').read_text())['results']
    cached = {('v1', '5 checkpoints x 6 TTA views, WBF'): v1_raw['Faster R-CNN 5 checkpoints TTA (0.2587 pipeline)'],
              ('v1', '+ temporal processing'): v1_raw['Faster R-CNN + temporal (0.2810 pipeline)'],
              ('v1', '+ FCOS fusion + temporal'): v1_raw['Faster R-CNN + FCOS fusion + temporal (tuned)'],
              ('v2', '5 checkpoints x 6 TTA views, WBF'): v2_raw['v2 5 checkpoints TTA'],
              ('v2', '+ temporal processing'): v2_raw['A: v2 + temporal (tuned)'],
              ('v2', '+ FCOS fusion + temporal'): v2_raw['B: v2 + FCOS fusion + temporal (tuned)']}
    for model in ('v1', 'v2'):
        prediction_sets = native_sets(model, ids)
        single_checkpoint_metrics = [temporal_postprocessing.coco_metrics(gt, meta.image_id.tolist(), prediction_set, groups)
                                     for prediction_set in prediction_sets]
        rows.append(dict(model=model, step='1 checkpoint (mean of 5)',
                         **{metric: float(np.mean([metrics['all'][metric] for metrics in single_checkpoint_metrics]))
                            for metric in single_checkpoint_metrics[0]['all']},
                         **{f'mAP_{group}': float(np.mean([metrics[group]['mAP'] for metrics in single_checkpoint_metrics]))
                            for group in groups if group != 'all'}))
        ensemble = experiment_utils.fuse_sets(prediction_sets, meta)
        metrics = temporal_postprocessing.coco_metrics(gt, meta.image_id.tolist(), ensemble, groups)
        rows.append(dict(model=model, step='5 checkpoints, WBF, no TTA', **metrics['all'],
                         **{f'mAP_{group}': metrics[group]['mAP'] for group in groups if group != 'all'}))
        for step in ['5 checkpoints x 6 TTA views, WBF', '+ temporal processing', '+ FCOS fusion + temporal']:
            metrics = cached[(model, step)]['metrics']
            rows.append(dict(model=model, step=step, **metrics['all'],
                             **{f'mAP_{group}': metrics[group]['mAP'] for group in groups if group != 'all'}))
        print(model, 'done', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / 'pipeline_ablation_official_val.csv', index=False)
    map_by_step = table.pivot_table(index='step', columns='model', values='mAP', sort=False)
    map_by_step['model effect (v2 - v1)'] = map_by_step['v2'] - map_by_step['v1']
    print(table[['model', 'step', 'mAP', 'AP50', 'AP75', 'APs', 'AR']].round(4).to_string(index=False))
    print(map_by_step.round(4).to_string())
