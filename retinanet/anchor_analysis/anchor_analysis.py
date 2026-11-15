"""
Anchor analysis for RetinaNet.

Examines object sizes and aspect ratios after image resizing, as well as how well
different anchor configurations cover these objects.

Additionally, anchor scales from 0.35 to 1.0 of the default sizes are tested
with three aspect-ratio sets to select a configuration for retinanet/run_retinanet.py.
"""

import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
DATA = ROOT / 'datasets/multiple-object-video-detection-hw-2-ucu-ai-2026'
OUT = Path(__file__).resolve().parent
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_SIZE, MAX_SIZE = 800, 1333
STRIDES = [8, 16, 32, 64, 128]
DEFAULT_BASE_SIZES = [32, 64, 128, 256, 512]
DEFAULT_RATIOS = (0.5, 1.0, 2.0)


def anchor_sizes(mult, base=DEFAULT_BASE_SIZES):
    """torchvision rule: per level (x, x*2^(1/3), x*2^(2/3)), here with x = mult * default base size."""
    return tuple((int(base_size * mult), int(base_size * mult * 2 ** (1 / 3)), int(base_size * mult * 2 ** (2 / 3)))
                 for base_size in base)


TRAINED_CONFIGS = {
    'default (baseline, tuned_focal_loss)':
        (((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812)), DEFAULT_RATIOS),
    'tuned_anchors (0.5x)':
        (((16, 20, 25), (32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406)), DEFAULT_RATIOS),
    'anchor_intermediate_075 (0.75x)':
        (((24, 30, 38), (48, 60, 76), (96, 121, 152), (192, 242, 305), (384, 484, 610)), DEFAULT_RATIOS),
}


def rebuild_folds(coco):
    """The fold code of the RetinaNet notebooks: sort by video + frame_id, every 5th frame per video, GroupKFold(5) by video."""
    images = pd.DataFrame(coco['images']).sort_values(['video', 'frame_id']).reset_index(drop=True)
    images['pos'] = images.groupby('video').cumcount()
    cv_frames = images[images.pos % 5 == 0].copy().reset_index(drop=True)
    cv_frames['Fold'] = -1
    for k, (_, val_index) in enumerate(GroupKFold(n_splits=5).split(cv_frames, groups=cv_frames['video'])):
        cv_frames.loc[val_index, 'Fold'] = k
    return cv_frames


def resize_scale(w, h):
    """Scale factor of torchvision's GeneralizedRCNNTransform resize (min side 800, max side 1333)."""
    return min(MIN_SIZE / min(w, h), MAX_SIZE / max(w, h))


def anchors_for(size_cfg, ratios, w, h):
    """All anchors (in resized-image coordinates) of one frame size, and the resize scale of that frame size."""
    scale = resize_scale(w, h)
    resized_h, resized_w = int(math.floor(h * scale + 0.5)), int(math.floor(w * scale + 0.5))
    padded_h, padded_w = int(math.ceil(resized_h / 32) * 32), int(math.ceil(resized_w / 32) * 32)
    feature_maps = [torch.empty(1, 1, math.ceil(padded_h / stride), math.ceil(padded_w / stride), device=DEV)
                    for stride in STRIDES]
    generator = AnchorGenerator(sizes=size_cfg, aspect_ratios=(tuple(ratios),) * 5)
    image_list = ImageList(torch.empty(1, 3, padded_h, padded_w, device=DEV), [(resized_h, resized_w)])
    return generator(image_list, feature_maps)[0], scale


def gt_boxes(coco, ids):
    """Non-crowd ground-truth boxes (x1, y1, x2, y2, category) clipped to the frame, per image id; and the image records."""
    boxes_by_image = {}
    image_info = {image['id']: image for image in coco['images']}
    for ann in coco['annotations']:
        if ann['image_id'] in ids and not ann['iscrowd']:
            image = image_info[ann['image_id']]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = max(0, x), max(0, y), min(image['width'], x + w), min(image['height'], y + h)
            if x2 > x1 and y2 > y1:
                boxes_by_image.setdefault(ann['image_id'], []).append((x1, y1, x2, y2, ann['category_id']))
    return boxes_by_image, image_info


def coverage(boxes_by_image, image_info, size_cfg, ratios):
    """Per object: best anchor IoU and number of anchors with IoU >= 0.5 (RetinaNet foreground threshold)."""
    images_by_size = {}
    for image_id in boxes_by_image:
        images_by_size.setdefault((image_info[image_id]['width'], image_info[image_id]['height']), []).append(image_id)
    best_iou, n_pos, area, classes = [], [], [], []
    for (w, h), image_ids in images_by_size.items():
        anchors, scale = anchors_for(size_cfg, ratios, w, h)
        for image_id in image_ids:
            gt = torch.tensor([box[:4] for box in boxes_by_image[image_id]], device=DEV, dtype=torch.float32) * scale
            iou = box_iou(gt, anchors)
            best_iou.append(iou.max(1).values.cpu())
            n_pos.append((iou >= 0.5).sum(1).cpu())
            area.append(((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).cpu())
            classes += [box[4] for box in boxes_by_image[image_id]]
    return pd.DataFrame(dict(best_iou=torch.cat(best_iou).numpy(), n_pos=torch.cat(n_pos).numpy(),
                             area=torch.cat(area).numpy(), cls=classes))


def size_group(area):
    """COCO size group of each box area: small < 32^2 <= medium < 96^2 <= large."""
    return np.where(area < 32 ** 2, 'small', np.where(area < 96 ** 2, 'medium', 'large'))


def summarize(objects, name):
    """Coverage summary rows of one anchor configuration: all objects and per COCO size group."""
    rows = []
    for group_name, subset in [('all', objects),
                               *[(size, objects[size_group(objects.area) == size]) for size in ('small', 'medium', 'large')]]:
        rows.append(dict(config=name, objects=group_name, n=len(subset),
                         covered_iou50=float((subset.best_iou >= 0.5).mean()),
                         mean_best_iou=float(subset.best_iou.mean()), mean_pos_anchors=float(subset.n_pos.mean()),
                         zero_pos_share=float((subset.n_pos == 0).mean())))
    return rows


if __name__ == '__main__':
    coco = json.loads((DATA / 'train/annotations_coco.json').read_text())
    import sklearn
    rebuilt = rebuild_folds(coco)
    cv_frames = pd.read_csv(ROOT / 'shared/sequence_folds.csv')
    same_frames = set(rebuilt.id) == set(cv_frames.id)
    same_folds = (rebuilt[['id', 'Fold']].sort_values('id').reset_index(drop=True)
                  .equals(cv_frames[['id', 'Fold']].sort_values('id').reset_index(drop=True)))
    (OUT / 'folds_check.json').write_text(json.dumps(dict(
        same_frames_as_faster_rcnn=bool(same_frames), same_fold_assignment_when_rebuilt_locally=bool(same_folds),
        sklearn_local=sklearn.__version__, frames=len(cv_frames), per_fold=cv_frames.groupby('Fold').size().to_dict(),
        note='GroupKFold of the local scikit-learn assigns some videos to other folds than the version used in Colab; the new runs '
             'therefore read shared/sequence_folds.csv (the Faster R-CNN folds) instead of rebuilding the split'), indent=2))
    print('same frames:', same_frames, '| same folds when rebuilt with sklearn', sklearn.__version__, ':', same_folds)

    boxes_by_image, image_info = gt_boxes(coco, set(cv_frames.id))
    class_names = {category['id']: category['name'] for category in coco['categories']}
    rows = []
    for image_id, boxes in boxes_by_image.items():
        scale = resize_scale(image_info[image_id]['width'], image_info[image_id]['height'])
        for x1, y1, x2, y2, category_id in boxes:
            w, h = (x2 - x1) * scale, (y2 - y1) * scale
            rows.append((class_names[category_id], w, h))
    box_table = pd.DataFrame(rows, columns=['cls', 'w', 'h'])
    box_table['side'] = np.sqrt(box_table.w * box_table.h)
    box_table['ratio_h_w'] = box_table.h / box_table.w
    box_table['size'] = size_group(box_table.w * box_table.h)
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    stats = []
    for group_name, subset in [('all', box_table),
                               *[(f'class {cls}', part) for cls, part in box_table.groupby('cls')],
                               *[(f'size {size}', part) for size, part in box_table.groupby('size')]]:
        stats.append(dict(group=group_name, n=len(subset), share=len(subset) / len(box_table),
                          **{f'side_p{int(quantile * 100)}': float(subset.side.quantile(quantile)) for quantile in quantiles},
                          **{f'ratio_p{int(quantile * 100)}': float(subset.ratio_h_w.quantile(quantile)) for quantile in quantiles}))
    pd.DataFrame(stats).to_csv(OUT / 'box_sizes.csv', index=False)
    print(pd.DataFrame(stats)[['group', 'n', 'share', 'side_p5', 'side_p25', 'side_p50', 'side_p75', 'side_p95',
                               'ratio_p10', 'ratio_p50', 'ratio_p90']].round(2).to_string(index=False))

    coverage_rows = []
    for name, (size_cfg, ratios) in TRAINED_CONFIGS.items():
        coverage_rows += summarize(coverage(boxes_by_image, image_info, size_cfg, ratios), name)
    pd.DataFrame(coverage_rows).to_csv(OUT / 'anchor_coverage.csv', index=False)
    print(pd.DataFrame(coverage_rows).round(3).to_string(index=False))

    scan_rows = []
    for mult in (0.35, 0.4, 0.5, 0.6, 0.65, 0.75, 0.85, 1.0):
        for ratio_name, ratios in [('0.5,1,2', DEFAULT_RATIOS), ('0.5,1,2,3', (0.5, 1.0, 2.0, 3.0)),
                                   ('0.33,0.5,1,2,3', (1 / 3, 0.5, 1.0, 2.0, 3.0))]:
            scan_rows += [dict(mult=mult, ratios=ratio_name, **{key: value for key, value in row.items() if key != 'config'})
                          for row in summarize(coverage(boxes_by_image, image_info, anchor_sizes(mult), ratios),
                                               f'{mult} {ratio_name}')]
            print(mult, ratio_name, 'done', flush=True)
    pd.DataFrame(scan_rows).to_csv(OUT / 'anchor_scan.csv', index=False)
    scan_table = pd.DataFrame(scan_rows)
    print(scan_table.pivot_table(index=['mult', 'ratios'], columns='objects', values='covered_iou50').round(3).to_string())
