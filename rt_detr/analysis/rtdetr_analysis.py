"""
RT-DETR analysis on the official validation set using saved predictions, without running inference again.
1. Small objects and dense scenes
RT-DETR, Faster R-CNN v2, and RetinaNet are compared based on detection performance for objects of different sizes and
scenes with different numbers of objects (maxDets = 100 and 500).
2. NMS and duplicate analysis
The effect of NMS (IoU = 0.5 and 0.7) on RT-DETR predictions is evaluated, and the number of duplicate detections
is compared with CNN models.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
TASK_DIR = ANALYSIS_DIR.parent
ROOT = TASK_DIR.parent
sys.path.insert(0, str(ROOT / 'shared'))

import experiment_utils
import temporal_postprocessing

FOLDS = range(5)
CHUNK = 400
VIEW = 'none@1.0'
DENSITY_BINS = [0, 20, 40, 60, 90, 1000]
CONFIDENT = 0.3


def model_folders():
    chosen = json.loads((TASK_DIR / 'inference' / 'chosen_config.json').read_text())['config']
    return {f'RT-DETR ({chosen})': TASK_DIR / 'inference' / 'cache' / 'views' / 'val',
            'Faster R-CNN v2 + scale aug': ROOT / 'final_pipeline/ensemble/cache/views/val',
            'RetinaNet (baseline)': ROOT / 'retinanet/inference/cache/views/val'}


def load_view(folder, image_ids):
    """Rows (boxes, scores, labels) of one fold checkpoint, aligned with image_ids."""
    rows, loaded_ids = [], []
    for start in range(0, len(image_ids), CHUNK):
        data = np.load(folder / f'{VIEW}_{start:05d}.npz')
        loaded_ids += data['ids'].tolist()
        rows += experiment_utils.unpack(data)
    assert loaded_ids == image_ids, folder
    return rows


def classwise_nms(rows, iou_threshold):
    import torch
    from torchvision.ops import batched_nms
    out = []
    for boxes, scores, labels in rows:
        keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores), torch.as_tensor(labels), iou_threshold)
        keep = keep.numpy()
        out.append((boxes[keep], scores[keep], labels[keep]))
    return out


def duplicate_pairs(rows):
    """Mean number per frame of pairs of confident boxes of the same class with IoU >= 0.5."""
    counts = []
    for boxes, scores, labels in rows:
        confident = scores >= CONFIDENT
        boxes, labels = boxes[confident].astype(np.float64), labels[confident]
        if len(boxes) < 2:
            counts.append(0)
            continue
        iou = temporal_postprocessing.iou_matrix(boxes, boxes)
        same_class = labels[:, None] == labels[None, :]
        upper = np.triu(np.ones_like(iou, dtype=bool), 1)
        counts.append(int((upper & same_class & (iou >= 0.5)).sum()))
    return float(np.mean(counts))


def evaluate_fold(job):
    """One model x fold: metrics for every density group (maxDets 100 and 500), plus the NMS check for RT-DETR."""
    model, folder, fold, image_ids, groups = job
    gt = temporal_postprocessing.coco_gt(experiment_utils.annotations_for('val'), image_ids)
    rows = load_view(folder / f'fold{fold}', image_ids)
    result = dict(model=model, fold=fold, duplicates_per_frame=duplicate_pairs(rows), metrics={})
    for max_dets in (100, 500):
        result['metrics'][max_dets] = temporal_postprocessing.coco_metrics(gt, image_ids, rows, groups=groups, max_dets=max_dets)
    if model.startswith('RT-DETR'):
        result['nms'] = {}
        for iou_threshold in (0.5, 0.7):
            nms_rows = classwise_nms(rows, iou_threshold)
            result['nms'][iou_threshold] = dict(
                metrics=temporal_postprocessing.coco_metrics(gt, image_ids, nms_rows)['all'],
                duplicates_per_frame=duplicate_pairs(nms_rows))
    return result


def density_groups(frames):
    annotations = experiment_utils.annotations_for('val')['annotations']
    objects = pd.Series(0, index=frames.image_id.astype(int))
    for annotation in annotations:
        if not annotation.get('iscrowd', 0):
            objects[int(annotation['image_id'])] += 1
    bins = pd.cut(objects, DENSITY_BINS, right=False)
    groups = {'all frames': objects.index.tolist()}
    for interval, ids in objects.groupby(bins, observed=True):
        groups[f'{int(interval.left)}-{int(interval.right) - 1} objects'] = ids.index.tolist()
    return groups, objects


if __name__ == '__main__':
    frames = experiment_utils.frames_meta('val')
    image_ids = frames.image_id.astype(str).tolist()
    groups, objects = density_groups(frames)
    print({name: len(ids) for name, ids in groups.items()}, flush=True)
    jobs = [(model, folder, fold, image_ids, groups) for model, folder in model_folders().items() for fold in FOLDS]
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(evaluate_fold, jobs))

    rows = []
    for result in results:
        for max_dets, by_group in result['metrics'].items():
            for group, metrics in by_group.items():
                rows.append(dict(model=result['model'], fold=result['fold'], max_dets=max_dets, group=group,
                                 frames=len(groups[group]), **metrics))
    table = pd.DataFrame(rows)
    table.to_csv(ANALYSIS_DIR / 'density_and_size.csv', index=False)

    nms_rows = []
    for result in results:
        if 'nms' in result:
            nms_rows.append(dict(model=result['model'], fold=result['fold'], variant='as predicted (no NMS)',
                                 duplicates_per_frame=result['duplicates_per_frame'], **result['metrics'][100]['all frames']))
            for iou_threshold, entry in result['nms'].items():
                nms_rows.append(dict(model=result['model'], fold=result['fold'], variant=f'+ class-wise NMS {iou_threshold}',
                                     duplicates_per_frame=entry['duplicates_per_frame'], **entry['metrics']))
    pd.DataFrame(nms_rows).to_csv(ANALYSIS_DIR / 'nms_check.csv', index=False)
    pd.DataFrame([dict(model=result['model'], fold=result['fold'], duplicates_per_frame=result['duplicates_per_frame']) for result in results]) \
        .to_csv(ANALYSIS_DIR / 'duplicates.csv', index=False)
    print(table[(table.max_dets == 100)].groupby(['group', 'model']).mAP.mean().unstack().round(4).to_string(), flush=True)
