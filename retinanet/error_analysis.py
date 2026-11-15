"""
Error analysis of RetinaNet and Faster R-CNN on the official validation set.
The goal is to investigate how removing the proposal stage (RPN) affects object detection performance.

Three models are compared:
- Faster R-CNN v1 (lr_scheduler).
- Faster R-CNN v2 with scale augmentation.
- RetinaNet baseline.

The analysis includes:
- AP at different IoU thresholds from 0.50 to 0.95 and for each class.
- Classification of detections with score ≥ 0.3 into true positives, duplicates,
  detections in ignore regions, localization errors, classification errors, and background errors.
- Analysis of missed objects by size, considering all predictions
  and separately those with score ≥ 0.3.
- Visualizations of frames where RetinaNet misses the most objects
  that Faster R-CNN v1 correctly detects.
"""

import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import retina
import detector_fusion
import experiment_utils
from torchvision.ops import box_iou

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'fcos/inference'))

OUT = HERE / 'analysis'
CONFIDENT_SCORE = 0.3
FRCNN_V1_CACHE = ROOT / 'faster_rcnn/official_validation/cache'
FRCNN_V2_VIEWS = ROOT / 'final_pipeline/ensemble/cache/views/val'
RETINA_VIEWS = HERE / 'inference/cache/views/val'
SIZE_RANGES = {'small': (0, 32 ** 2), 'medium': (32 ** 2, 96 ** 2), 'large': (96 ** 2, 1e12)}


def load_model_preds(model, fold, ids):
    """Cached native single-view predictions (boxes, scores, labels) of one fold checkpoint, in the order of ids."""
    if model == 'Faster R-CNN v1':
        cached = np.load(FRCNN_V1_CACHE / f'fold{fold}_none@1.0.npz')
        assert cached['ids'].astype(str).tolist() == ids
        return experiment_utils.unpack(cached, 'full_')
    folder = (FRCNN_V2_VIEWS if model.startswith('Faster R-CNN v2') else RETINA_VIEWS) / f'fold{fold}'
    return detector_fusion._chunks(folder, 'none@1.0', ids)


def gt_by_image():
    """Validation ground truth per image id as [x1, y1, x2, y2, category, area]: normal boxes and ignore (crowd) regions."""
    boxes, ignored = {}, {}
    for ann in retina.coco_val['annotations']:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            continue
        (ignored if ann['iscrowd'] else boxes).setdefault(ann['image_id'], []).append(
            [x, y, x + w, y + h, ann['category_id'], w * h])
    return boxes, ignored


def classify(pred, gt, ignored):
    """Per confident detection: category; per GT: detected at all scores / among confident detections."""
    boxes, scores, labels = pred
    gt_array = np.array(gt, dtype=np.float64).reshape(-1, 6)
    gt_boxes, gt_labels = torch.from_numpy(gt_array[:, :4]).float(), gt_array[:, 4].astype(int)
    order = np.argsort(-scores, kind='stable')
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    iou = (box_iou(torch.from_numpy(boxes).float(), gt_boxes).numpy() if len(boxes) and len(gt_array)
           else np.zeros((len(boxes), len(gt_array))))
    same_class_iou = iou * (labels[:, None] == gt_labels[None, :])
    hit_any = (same_class_iou >= 0.5).any(0) if len(boxes) else np.zeros(len(gt_array), bool)
    confident = scores >= CONFIDENT_SCORE
    hit_conf = (same_class_iou[confident] >= 0.5).any(0) if confident.any() else np.zeros(len(gt_array), bool)
    ignored_array = np.array(ignored, dtype=np.float64).reshape(-1, 6)
    inside = (retina.inside_ignored(torch.from_numpy(boxes).float(), torch.from_numpy(ignored_array[:, :4]).float(), 0.5).numpy()
              if len(ignored_array) and len(boxes)
              else np.zeros(len(boxes), bool))
    matched = np.zeros(len(gt_array), bool)
    categories = []
    for i in np.where(confident)[0]:
        candidates = np.where(same_class_iou[i] >= 0.5)[0]
        unmatched = candidates[~matched[candidates]]
        if len(unmatched):
            j = unmatched[np.argmax(same_class_iou[i, unmatched])]
            matched[j] = True
            categories.append('true positive')
            continue
        if len(candidates):
            categories.append('duplicate')
            continue
        if inside[i]:
            categories.append('inside ignore region')
            continue
        if len(gt_array) and same_class_iou[i].max() >= 0.1:
            categories.append('localisation (same class, IoU 0.1-0.5)')
            continue
        if len(gt_array) and iou[i].max() >= 0.5:
            categories.append('class confusion (IoU >= 0.5)')
            continue
        categories.append('background (IoU < 0.1)')
    return categories, hit_any, hit_conf, gt_array[:, 5], gt_labels


def ap_by_iou(preds, ids):
    """AP for every IoU threshold of COCO (Faster R-CNN COCO setup)."""
    import io
    from contextlib import redirect_stdout
    from pycocotools.cocoeval import COCOeval
    gt = retina.training_utils.make_gt(retina.coco_val, ids)
    detections = [dict(image_id=int(image_id), category_id=int(label),
                       bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], score=float(score))
                  for image_id, (boxes, scores, labels) in zip(ids, preds)
                  for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels)
                  if x2 > x1 and y2 > y1]
    with redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(gt, gt.loadRes(detections), 'bbox')
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
    precision = coco_eval.eval['precision'][:, :, :, 0, -1]
    result = {f'AP@{threshold:.2f}': float(precision[i][precision[i] > -1].mean())
              for i, threshold in enumerate(coco_eval.params.iouThrs)}
    class_names = {category['id']: category['name'] for category in retina.coco_val['categories']}
    for k, class_id in enumerate(coco_eval.params.catIds):
        class_precision = precision[:, :, k]
        result[f'class {class_names[class_id]}'] = (float(class_precision[class_precision > -1].mean())
                                                    if (class_precision > -1).any() else float('nan'))
    return result


def failure_figures(per_model, ids, gt_boxes_by_image, ignored_by_image, n=4):
    """Save the n frames (from different videos) where Faster R-CNN v1 finds the most objects that RetinaNet misses."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    (OUT / 'failure_cases').mkdir(parents=True, exist_ok=True)
    retina_hits, frcnn_hits = per_model['RetinaNet baseline'], per_model['Faster R-CNN v1']
    gap = [(int(frcnn_found.sum()) - int(retina_found.sum()), k)
           for k, (retina_found, frcnn_found) in enumerate(zip(retina_hits['hit_conf'], frcnn_hits['hit_conf']))]
    image_info = {image['id']: image for image in retina.coco_val['images']}
    chosen, videos = [], set()
    for _, k in sorted(gap, reverse=True):
        video = image_info[int(ids[k])]['video']
        if video not in videos:
            chosen.append(k)
            videos.add(video)
        if len(chosen) == n:
            break
    for k in chosen:
        image = image_info[int(ids[k])]
        frame = Image.open(retina.DATA / 'val' / image['file_name']).convert('RGB')
        fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
        for ax, (title, hits) in zip(axes, [('RetinaNet baseline', retina_hits), ('Faster R-CNN v1', frcnn_hits)]):
            ax.imshow(frame)
            ax.axis('off')
            gt = np.array(gt_boxes_by_image.get(int(ids[k]), []), dtype=float).reshape(-1, 6)
            for (x1, y1, x2, y2, category, _), hit in zip(gt, hits['hit_conf'][k]):
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1, ec='lime' if hit else 'red'))
            for x1, y1, x2, y2, *_ in ignored_by_image.get(int(ids[k]), []):
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1, ls='--', ec='yellow'))
            ax.set_title(f"{title}: {int(hits['hit_conf'][k].sum())}/{len(gt)} objects found (score >= {CONFIDENT_SCORE}); "
                         f"green found, red missed, yellow dashed ignore")
        fig.tight_layout()
        fig.savefig(OUT / 'failure_cases' / f"{image['video']}_{image['frame_id']:05d}.png", dpi=80)
        plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    ids = [str(image['id']) for image in retina.coco_val['images']]
    meta_ids = experiment_utils.frames_meta('val').image_id.astype(str).tolist()
    assert ids == meta_ids
    gt_boxes_by_image, ignored_by_image = gt_by_image()
    models = ['Faster R-CNN v1', 'Faster R-CNN v2 + scale aug', 'RetinaNet baseline']
    ap_rows, error_rows, miss_rows, fold0_hits = [], [], [], {}
    for model in models:
        for fold in range(5):
            preds = load_model_preds(model, fold, ids)
            ap_rows.append(dict(model=model, fold=fold, **ap_by_iou(preds, ids)))
            categories, hit_any, hit_conf, area, classes = [], [], [], [], []
            for image_id, pred in zip(ids, preds):
                frame_categories, frame_hit_any, frame_hit_conf, frame_area, frame_classes = classify(
                    pred, gt_boxes_by_image.get(int(image_id), []), ignored_by_image.get(int(image_id), []))
                categories += frame_categories
                hit_any.append(frame_hit_any)
                hit_conf.append(frame_hit_conf)
                area.append(frame_area)
                classes.append(frame_classes)
            category_counts = pd.Series(categories).value_counts()
            error_rows.append(dict(model=model, fold=fold, confident_detections=len(categories),
                                   **(category_counts / len(categories)).to_dict()))
            all_hit_any, all_hit_conf, all_area = np.concatenate(hit_any), np.concatenate(hit_conf), np.concatenate(area)
            for group_name, (low, high) in {'all': (0, 1e12), **SIZE_RANGES}.items():
                selected = (all_area >= low) & (all_area < high)
                miss_rows.append(dict(model=model, fold=fold, objects=group_name, n=int(selected.sum()),
                                      missed_all_scores=float(1 - all_hit_any[selected].mean()),
                                      missed_score_ge_0_3=float(1 - all_hit_conf[selected].mean())))
            if fold == 0:
                fold0_hits[model] = dict(hit_conf=hit_conf)
            print(model, 'fold', fold, 'done', flush=True)
    for name, rows in [('ap_by_iou', ap_rows), ('error_breakdown', error_rows), ('misses', miss_rows)]:
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f'{name}.csv', index=False)
        group_columns = ['model', 'objects'] if name == 'misses' else ['model']
        print(table.drop(columns='fold').groupby(group_columns).mean(numeric_only=True).round(4).to_string(), '\n')
    failure_figures(fold0_hits, ids, gt_boxes_by_image, ignored_by_image)
