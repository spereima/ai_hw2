"""
Data, cross-validation folds and COCO evaluation shared by Faster R-CNN v2, RetinaNet and RT-DETR.
"""
import io
import json
import os
import random
import shutil
import sys
import hashlib
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
DATA = ROOT / 'datasets/multiple-object-video-detection-hw-2-ucu-ai-2026'
FOLDS_CSV = ROOT / 'shared/sequence_folds.csv'
MAX_DETS = 100

coco_train = json.loads((DATA / 'train/annotations_coco.json').read_text())
coco_val = json.loads((DATA / 'val/annotations_coco.json').read_text())
folds = pd.read_csv(FOLDS_CSV)
CATS = {int(category['id']): category['name'] for category in coco_train['categories']}
CAT_IDS = sorted(CATS)
CAT_TO_LABEL = {category_id: i + 1 for i, category_id in enumerate(CAT_IDS)}
LABEL_TO_CAT = {v: k for k, v in CAT_TO_LABEL.items()}
N_CLASSES = len(CATS) + 1
FOLDS_SHA = hashlib.sha256(FOLDS_CSV.read_bytes()).hexdigest()[:12]
assert folds.groupby('video').Fold.nunique().max() == 1, 'a video is split across folds'
assert not set(folds.video) & {image_info['video'] for image_info in coco_val['images']}, \
    'official validation videos in the CV folds'
FOLDS = pd.read_csv(FOLDS_CSV)


def make_gt(coco, ids):
    """COCO ground-truth object for the images `ids` (boxes with zero width or height are dropped)."""
    ids = set(map(int, ids))
    annotations = []
    for annotation in coco['annotations']:
        if int(annotation['image_id']) not in ids:
            continue
        x, y, w, h = map(float, annotation['bbox'])
        if w <= 0 or h <= 0:
            continue
        item = dict(annotation)
        item['area'] = float(annotation.get('area', w * h))
        item['iscrowd'] = int(annotation.get('iscrowd', 0))
        annotations.append(item)
    for j, annotation in enumerate(annotations):
        annotation['id'] = j + 1
    gt = COCO()
    gt.dataset = dict(
        info={}, licenses=[], images=[image_info for image_info in coco['images'] if int(image_info['id']) in ids],
        annotations=annotations, categories=coco['categories'])
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def coco_metrics(preds, gt):
    """COCO bbox metrics (mAP, AP50, AP75, APs, APm, APl, AR, per-class AP) of predictions against `gt`."""
    detections = []
    for image_id, prediction in preds:
        for box, score, label in zip(
            prediction['boxes'].tolist(),
            prediction['scores'].tolist(),
            prediction['labels'].tolist()):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(dict(image_id=image_id, category_id=LABEL_TO_CAT[int(label)],
                                   bbox=[x1, y1, x2 - x1, y2 - y1], score=float(score)))
    keys = ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']
    if not detections:
        return {**{k: 0. for k in keys}, 'per_class': {v: 0. for v in CATS.values()}}
    with redirect_stdout(io.StringIO()):
        evaluation = COCOeval(gt, gt.loadRes(detections), 'bbox')
        evaluation.params.imgIds = sorted(gt.getImgIds())
        evaluation.params.maxDets = [1, 10, MAX_DETS]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    precision = evaluation.eval['precision'][:, :, :, 0, -1]
    per_class = {}
    for k, category_id in enumerate(evaluation.params.catIds):
        values = precision[:, :, k]
        per_class[CATS[category_id]] = float(values[values > -1].mean()) if (values > -1).any() else float('nan')
    stats = evaluation.stats
    return dict(
        mAP=float(precision[precision > -1].mean()), AP50=float(stats[1]), AP75=float(stats[2]), APs=float(stats[3]),
        APm=float(stats[4]), APl=float(stats[5]), AR=float(stats[8]), per_class=per_class)


def save_atomic(obj, path, keep_backup=True):
    """Write a json or torch file through a temporary file, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix=path.suffix, dir=path.parent)
    os.close(file_descriptor)
    try:
        if path.suffix == '.json':
            with open(tmp_path, 'w') as file:
                json.dump(obj, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
        else:
            torch.save(obj, tmp_path)
        if keep_backup and path.exists():
            backup_tmp = str(path) + '.bak.tmp'
            shutil.copy2(path, backup_tmp)
            os.replace(backup_tmp, str(path) + '.bak')
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def seed_all(seed):
    """Seed python, numpy and torch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class VisDroneDataset(Dataset):
    """VisDrone frames as (uint8 image tensor, target dict: boxes, labels, image_id, ignored_boxes); optional horizontal flip."""
    def __init__(self, dataset_path, split, coco_data, image_ids=None, augment=False, hflip=0.5):
        self.dataset_path = dataset_path
        self.split = split
        self.augment = augment
        self.hflip = hflip
        self.images = coco_data['images']
        if image_ids is not None:
            selected_ids = set(image_ids)
            self.images = [image for image in self.images if image['id'] in selected_ids]
        self.annotations = defaultdict(list)
        self.ignore_annotations = defaultdict(set)
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if annotation['iscrowd'] == 0:
                self.annotations[image_id].append(annotation)
            else:
                self.ignore_annotations[image_id].add(tuple(annotation['bbox']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_path = os.path.join(self.dataset_path, self.split, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        boxes, labels = [], []
        for annotation in self.annotations[image_info['id']]:
            x, y, width, height = annotation['bbox']
            xmin, ymin = max(0, x), max(0, y)
            xmax, ymax = min(image_width, x + width), min(image_height, y + height)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
        ignored_boxes = []
        for x, y, width, height in self.ignore_annotations[image_info['id']]:
            xmin, ymin = max(0, x), max(0, y)
            xmax, ymax = min(image_width, x + width), min(image_height, y + height)
            if xmax > xmin and ymax > ymin:
                ignored_boxes.append([xmin, ymin, xmax, ymax])
        ignored_boxes = torch.tensor(ignored_boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and torch.rand(1).item() < self.hflip:
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                old_xmin, old_xmax = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0] = image_width - old_xmax
                boxes[:, 2] = image_width - old_xmin
            if len(ignored_boxes) > 0:
                old_xmin, old_xmax = ignored_boxes[:, 0].clone(), ignored_boxes[:, 2].clone()
                ignored_boxes[:, 0] = image_width - old_xmax
                ignored_boxes[:, 2] = image_width - old_xmin
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor(image_info['id']), 'ignored_boxes': ignored_boxes}
        return image, target

def fold_ids(fold):
    """(training frame ids, held-out frame ids) of a fold."""
    return FOLDS.loc[FOLDS.Fold != fold, 'id'].tolist(), FOLDS.loc[FOLDS.Fold == fold, 'id'].tolist()

def inside_ignored(boxes, ignored, threshold):
    """Share of each box inside its most-overlapping ignore region (intersection / box area) >= threshold (Faster R-CNN rule)."""
    is_inside = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    for start in range(0, len(boxes), 65536):
        chunk = boxes[start:start + 65536]
        top_left = torch.maximum(chunk[:, None, :2], ignored[None, :, :2])
        bottom_right = torch.minimum(chunk[:, None, 2:], ignored[None, :, 2:])
        overlap_size = (bottom_right - top_left).clamp(min=0)
        intersection = overlap_size[..., 0] * overlap_size[..., 1]
        box_area = ((chunk[:, 2] - chunk[:, 0]).clamp(min=1e-6) * (chunk[:, 3] - chunk[:, 1]).clamp(min=1e-6))
        is_inside[start:start + 65536] = (intersection / box_area[:, None]).amax(dim=1) >= threshold
    return is_inside
