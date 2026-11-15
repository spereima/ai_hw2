import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class VisDrone(Dataset):
    def __init__(self, root, split, coco, image_ids, train=False, seed=42):
        ids = set(map(int, image_ids))
        self.root, self.split = Path(root), split
        self.train, self.seed = train, seed
        self.epoch = 0
        self.images = [image_info for image_info in coco['images'] if int(image_info['id']) in ids]
        cat_ids = sorted(int(category['id']) for category in coco['categories'])
        cat_to_label = {category_id: i + 1 for i, category_id in enumerate(cat_ids)}
        self.objs, self.ign = defaultdict(list), defaultdict(set)
        for annotation in coco['annotations']:
            image_id = int(annotation['image_id'])
            if image_id not in ids:
                continue
            x, y, w, h = map(float, annotation['bbox'])
            if annotation.get('iscrowd', 0):
                self.ign[image_id].add((x, y, x + w, y + h))
            else:
                self.objs[image_id].append((x, y, x + w, y + h, cat_to_label[int(annotation['category_id'])]))

    def __len__(self):
        return len(self.images)

    def get_target(self, i):
        image_info = self.images[i]
        width, height = image_info['width'], image_info['height']
        objects = np.asarray(self.objs[int(image_info['id'])], dtype=np.float32).reshape(-1, 5)
        boxes = objects[:, :4].copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        ignored_boxes = np.asarray(sorted(self.ign[int(image_info['id'])]), dtype=np.float32).reshape(-1, 4)
        ignored_boxes[:, [0, 2]] = ignored_boxes[:, [0, 2]].clip(0, width)
        ignored_boxes[:, [1, 3]] = ignored_boxes[:, [1, 3]].clip(0, height)
        ignored_boxes = ignored_boxes[(ignored_boxes[:, 2] > ignored_boxes[:, 0]) &
                                      (ignored_boxes[:, 3] > ignored_boxes[:, 1])]
        return dict(boxes=torch.from_numpy(boxes[ok]), labels=torch.from_numpy(objects[ok, 4]).long(),
                    ignored=torch.from_numpy(ignored_boxes), image_id=torch.tensor(int(image_info['id'])))

    def __getitem__(self, i):
        image_info = self.images[i]
        target = self.get_target(i)
        with Image.open(self.root / self.split / image_info['file_name']) as picture:
            img = TF.pil_to_tensor(picture.convert('RGB')).float().div_(255)
        if self.train:
            rng = random.Random(self.seed * 1000003 + self.epoch * 100003 + i)
            if rng.random() < .5:
                img = img.flip(-1)
                w = img.shape[-1]
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = w - target['boxes'][:, [2, 0]]
                target['boxes'] = boxes
                ignored = target['ignored'].clone()
                ignored[:, [0, 2]] = w - target['ignored'][:, [2, 0]]
                target['ignored'] = ignored
            if rng.random() < .5:
                img = TF.adjust_brightness(img, rng.uniform(.8, 1.2))
                img = TF.adjust_contrast(img, rng.uniform(.8, 1.2))
                img = TF.adjust_saturation(img, rng.uniform(.8, 1.2))
        return img, target
def collate(batch):
    return tuple(zip(*batch))
