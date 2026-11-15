import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class VisDroneDataset(Dataset):
    def __init__(self, dataset_path, split, coco_data, image_ids=None, augment=False):
      self.dataset_path = dataset_path
      self.split = split
      self.augment = augment

      self.images = coco_data["images"]

      if image_ids is not None:
          selected_ids = set(image_ids)
          self.images = [image for image in self.images if image["id"] in selected_ids]

      self.annotations = defaultdict(list)
      self.ignore_annotations = defaultdict(set)
      own_ids = {image["id"] for image in self.images}

      for annotation in coco_data["annotations"]:
          image_id = annotation["image_id"]
          if image_id not in own_ids:
              continue

          if annotation["iscrowd"] == 0:
              self.annotations[image_id].append(annotation)
          else:
              self.ignore_annotations[image_id].add(tuple(annotation["bbox"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]

        image_path = os.path.join(self.dataset_path, self.split, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        boxes = []
        labels = []

        for annotation in self.annotations[image_info["id"]]:
            x, y, width, height = annotation["bbox"]

            xmin = max(0, x)
            ymin = max(0, y)
            xmax = min(image_width, x + width)
            ymax = min(image_height, y + height)

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        labels = torch.tensor(labels, dtype=torch.int64)

        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.float() / 255.0

        ignored_boxes = []

        for x, y, width, height in self.ignore_annotations[image_info["id"]]:
            xmin = max(0, x)
            ymin = max(0, y)
            xmax = min(image_width, x + width)
            ymax = min(image_height, y + height)

            if xmax > xmin and ymax > ymin:
                ignored_boxes.append([xmin, ymin, xmax, ymax])

        ignored_boxes = torch.tensor(ignored_boxes, dtype=torch.float32).reshape(-1, 4)

        if self.augment and torch.rand(1).item() < 0.5:
          image = torch.flip(image, dims=[2])

          if len(boxes) > 0:
              old_xmin = boxes[:, 0].clone()
              old_xmax = boxes[:, 2].clone()

              boxes[:, 0] = image_width - old_xmax
              boxes[:, 2] = image_width - old_xmin

          if len(ignored_boxes) > 0:
              old_xmin = ignored_boxes[:, 0].clone()
              old_xmax = ignored_boxes[:, 2].clone()

              ignored_boxes[:, 0] = image_width - old_xmax
              ignored_boxes[:, 2] = image_width - old_xmin

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(image_info["id"]), "ignored_boxes": ignored_boxes}

        return image, target


class FCOSDataset(VisDroneDataset):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        if self.augment:
            for x1, y1, x2, y2 in target['ignored_boxes'].int().tolist():
                image[:, max(0,y1):min(image.shape[1],y2),
                      max(0,x1):min(image.shape[2],x2)] = mean
        return image, target


def collate(batch):
    return tuple(zip(*batch))
