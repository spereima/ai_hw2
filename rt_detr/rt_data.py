"""
Data preparation module for RT-DETR that processes images and annotations into the required format, including ignore regions.
"""
import os, random
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrImageProcessor

MODEL_ID = 'PekingU/rtdetr_r50vd'
CATEGORY_TO_INDEX = {category_id: index for index, category_id in enumerate(range(1, 9))}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def worker_init(_):
    info = torch.utils.data.get_worker_info()
    seed_all(info.seed % 2 ** 31)

def subset_coco(coco, ids):
    """Only the images / annotations of the given ids (the dataset then holds a small copy; outputs for these ids are unchanged)."""
    ids = set(int(i) for i in ids)
    return dict(images=[image for image in coco['images'] if int(image['id']) in ids],
                annotations=[annotation for annotation in coco['annotations'] if int(annotation['image_id']) in ids], categories=coco['categories'])

class VisDroneDataset(Dataset):
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

_PROC = {}
def processor(cfg):
    key = (cfg['height'], cfg['width'])
    if key not in _PROC:
        processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
        processor.size = {'height': cfg['height'], 'width': cfg['width']}
        _PROC[key] = processor
    return _PROC[key]

class Collate:
    """The notebook's rtdetr_collate_fn: COCO-style annotations (category index 0..7) through the RT-DETR image processor; plus the
    ignore regions as normalised (cx, cy, w, h) for CHANGED 3."""
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, batch):
        proc = processor(self.cfg)
        images, annotations, originals, ignored = [], [], [], []
        for image, target in batch:
            arr = image.permute(1, 2, 0).numpy()
            h, w = arr.shape[:2]
            images.append(arr)
            coco = []
            for (x1, y1, x2, y2), label in zip(target['boxes'].tolist(), target['labels'].tolist()):
                coco.append(dict(id=len(coco), image_id=int(target['image_id']), category_id=CATEGORY_TO_INDEX[int(label)],
                                 bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0))
            annotations.append(dict(image_id=int(target['image_id']), annotations=coco))
            originals.append(target)
            ignored_xyxy = target['ignored_boxes']
            ignored.append(torch.stack([(ignored_xyxy[:, 0] + ignored_xyxy[:, 2]) / 2 / w, (ignored_xyxy[:, 1] + ignored_xyxy[:, 3]) / 2 / h,
                                        (ignored_xyxy[:, 2] - ignored_xyxy[:, 0]) / w, (ignored_xyxy[:, 3] - ignored_xyxy[:, 1]) / h], 1) if len(ignored_xyxy) else torch.zeros(0, 4))
        enc = proc(images=images, annotations=annotations, return_tensors='pt')
        for lab, ign in zip(enc['labels'], ignored):
            lab['ignored_boxes'] = ign
        enc['original_targets'] = originals
        return enc

