"""
RetinaNet is a one-stage anchor-based detector built on torchvision ResNet-50 FPN.

Changes made to RetinaNet:
- COCO-pretrained weights instead of an ImageNet-pretrained backbone.
- Ignore-region handling (IoA ≥ 0.5).
- Best epoch selection based on held-out fold mAP.
- Inference: score ≥ 0.01, up to 500 detections, NMS = 0.5.
- The same 5 folds as Faster R-CNN.
- Configurable anchors, focal loss, and IoU thresholds.
"""

import os
import sys
import json
import math
import time
import random
import hashlib
import types
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
import training_utils
from training_utils import (DATA, coco_train, coco_val, FOLDS_CSV, FOLDS, FOLDS_SHA, seed_all, VisDroneDataset, fold_ids,
                            inside_ignored)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import sigmoid_focal_loss, box_iou

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs'
sys.path.insert(0, str(ROOT / 'shared'))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 8
NUM_WORKERS = 3

DEFAULT_ANCHORS = ((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812))
BASE = dict(name='baseline', seed=42, epochs=6, bs=1, lr=1e-4, wd=1e-4, step_size=5, step_gamma=0.1, clip=1.0,
            min_size=800, max_size=1333, trainable_layers=3,
            anchor_sizes=DEFAULT_ANCHORS, ratios=(0.5, 1.0, 2.0), fg_iou=0.5, bg_iou=0.4, focal_alpha=0.25, focal_gamma=2.0,
            ignore_ioa=0.5, score_thr=0.01, nms_thr=0.5, det_per_img=500, topk=1000, hflip=0.5,
            init='coco', save_every=1000)

def detection_collate_fn(batch):
    images, targets = zip(*batch)
    retinanet_targets = []
    for target in targets:
        new_target = target.copy()
        new_target['labels'] = target['labels'] - 1
        retinanet_targets.append(new_target)
    return list(images), retinanet_targets

def modified_focal_loss(self, targets, head_outputs, matched_idxs):
    """Notebook cell 39 (alpha / gamma from the configuration; with 0.25 / 2.0 identical to torchvision's loss)."""
    losses = []
    cls_logits = head_outputs['cls_logits']
    for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
        foreground_idxs = matched_idxs_per_image >= 0
        num_foreground = foreground_idxs.sum()
        gt_classes_target = torch.zeros_like(cls_logits_per_image)
        gt_classes_target[foreground_idxs, targets_per_image['labels'][matched_idxs_per_image[foreground_idxs]]] = 1.0
        valid_idxs = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
        loss = sigmoid_focal_loss(cls_logits_per_image[valid_idxs], gt_classes_target[valid_idxs],
                                  alpha=self.experiment_alpha, gamma=self.experiment_gamma, reduction='sum')
        losses.append(loss / max(1, int(num_foreground.item())))
    return torch.stack(losses).mean()

def compute_loss_ignore_aware(self, targets, head_outputs, anchors):
    """torchvision RetinaNet.compute_loss + CHANGED 2: background anchors inside ignore regions -> BETWEEN_THRESHOLDS (no loss)."""
    matched_idxs = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image['boxes'].numel() == 0:
            matched_idxs_per_image = torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
        else:
            matched_idxs_per_image = self.proposal_matcher(box_iou(targets_per_image['boxes'], anchors_per_image))
        ignored_boxes = targets_per_image.get('ignored_boxes')
        if self.ignore_ioa and ignored_boxes is not None and len(ignored_boxes):
            background_idxs = torch.where(matched_idxs_per_image == det_utils.Matcher.BELOW_LOW_THRESHOLD)[0]
            if len(background_idxs):
                in_ignored_region = inside_ignored(anchors_per_image[background_idxs], ignored_boxes, self.ignore_ioa)
                matched_idxs_per_image[background_idxs[in_ignored_region]] = det_utils.Matcher.BETWEEN_THRESHOLDS
        matched_idxs.append(matched_idxs_per_image)
    return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

def add_ignore_rescaling(model):
    """The resize inside the model only rescales 'boxes'; the ignore regions are rescaled the same way (as in Faster R-CNN)."""
    standard_forward = model.transform.forward
    def forward(self, images, targets=None):
        original_sizes = [image.shape[-2:] for image in images]
        image_list, targets = standard_forward(images, targets)
        if targets is not None:
            for target, (old_height, old_width), (new_height, new_width) in zip(targets, original_sizes, image_list.image_sizes):
                ignored_boxes = target['ignored_boxes'].clone()
                ignored_boxes[:, [0, 2]] *= new_width / old_width
                ignored_boxes[:, [1, 3]] *= new_height / old_height
                target['ignored_boxes'] = ignored_boxes
        return image_list, targets
    model.transform.forward = types.MethodType(forward, model.transform)

def build_model(cfg):
    """torchvision RetinaNet R50-FPN (COCO weights) with new class / box output layers, custom anchors, ignore-aware focal loss."""
    seed_all(cfg['seed'])
    model_kwargs = dict(trainable_backbone_layers=cfg['trainable_layers'], min_size=cfg['min_size'], max_size=cfg['max_size'],
                        score_thresh=cfg['score_thr'], nms_thresh=cfg['nms_thr'], detections_per_img=cfg['det_per_img'],
                        topk_candidates=cfg['topk'], fg_iou_thresh=cfg['fg_iou'], bg_iou_thresh=cfg['bg_iou'])
    if cfg.get('init', 'coco') == 'imagenet':
        from torchvision.models import ResNet50_Weights
        model = retinanet_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.DEFAULT, num_classes=NUM_CLASSES, **model_kwargs)
    else:
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1, **model_kwargs)
    anchor_sizes = tuple(tuple(int(value) for value in level_sizes) for level_sizes in cfg['anchor_sizes'])
    aspect_ratios = tuple(float(ratio) for ratio in cfg['ratios'])
    model.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=(aspect_ratios,) * 5)
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    classification_head = model.head.classification_head
    classification_head.cls_logits = torch.nn.Conv2d(256, num_anchors * NUM_CLASSES, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(classification_head.cls_logits.weight, std=0.01)
    torch.nn.init.constant_(classification_head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    classification_head.num_classes, classification_head.num_anchors = NUM_CLASSES, num_anchors
    regression_head = model.head.regression_head
    if num_anchors != regression_head.bbox_reg.out_channels // 4:
        regression_head.bbox_reg = torch.nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(regression_head.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(regression_head.bbox_reg.bias)
    classification_head.experiment_alpha, classification_head.experiment_gamma = cfg['focal_alpha'], cfg['focal_gamma']
    classification_head.compute_loss = types.MethodType(modified_focal_loss, classification_head)
    model.ignore_ioa = cfg['ignore_ioa']
    model.compute_loss = types.MethodType(compute_loss_ignore_aware, model)
    add_ignore_rescaling(model)
    return model

def to_float(images):
    """uint8 image tensors -> float images in [0, 1] on the device."""
    return [image.to(DEV, non_blocking=True).float().div_(255) for image in images]

@torch.inference_mode()
def predict(model, dataset):
    """Detections for every frame of the dataset as (image_id, dict(boxes, scores, labels 1..8)) rows."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detection_collate_fn)
    rows = []
    for images, targets in tqdm(loader, desc='Evaluate', leave=False):
        with torch.autocast(device_type=DEV.type, enabled=DEV.type == 'cuda'):
            prediction = model(to_float(images))[0]
        rows.append((int(targets[0]['image_id']), dict(boxes=prediction['boxes'].float().cpu(), scores=prediction['scores'].float().cpu(),
                                                        labels=prediction['labels'].cpu() + 1)))
    return rows

def evaluate(model, ids, coco=None):
    """COCO metrics (Faster R-CNN code) of the model on the frames `ids` of the train (default) or val annotations."""
    coco = coco or coco_train
    dataset = VisDroneDataset(str(DATA), 'train' if coco is coco_train else 'val', coco, ids, augment=False)
    return training_utils.coco_metrics(predict(model, dataset), training_utils.make_gt(coco, ids))

@torch.inference_mode()
def measure_speed(model, num_frames=40, warmup_frames=8):
    """ms per official-validation frame, batch 1, AMP, model only (decoding excluded) — the Faster R-CNN measurement."""
    model.eval()
    dataset = VisDroneDataset(str(DATA), 'val', coco_val, augment=False)
    images = [dataset[(i * 7) % len(dataset)][0] for i in range(num_frames + warmup_frames)]
    seconds_per_frame = []
    for image in images:
        float_images = to_float([image])
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.autocast(device_type=DEV.type, enabled=DEV.type == 'cuda'):
            model(float_images)
        torch.cuda.synchronize()
        seconds_per_frame.append(time.perf_counter() - start_time)
    return float(np.mean(seconds_per_frame[warmup_frames:]) * 1000)

def cfg_id(cfg):
    """Hash of the configuration (without save_every); a checkpoint / result belongs to the configuration with the same id."""
    stable = {key: value for key, value in cfg.items() if key != 'save_every'}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=list).encode()).hexdigest()

def make_cfg(overrides, fold):
    """BASE configuration with the experiment overrides, the fold number and the hash of the folds file."""
    cfg = {**BASE, **overrides, 'fold': fold, 'folds': FOLDS_SHA}
    cfg['anchor_sizes'] = [list(level_sizes) for level_sizes in cfg['anchor_sizes']]
    cfg['ratios'] = list(cfg['ratios'])
    return cfg

def worker_init(worker_id):
    """Seed every DataLoader worker differently (derived from the worker seed)."""
    worker_info = torch.utils.data.get_worker_info()
    seed_all(worker_info.seed % 2 ** 31)

def run_experiment(overrides, fold):
    """Train one configuration on one fold (resumable), keep the epoch with the best held-out mAP, save result.json and return it."""
    cfg = make_cfg(overrides, fold)
    folder = OUT / cfg['name'] / f'fold{fold}'
    result_path = folder / 'result.json'
    if result_path.exists():
        saved_result = json.loads(result_path.read_text())
        assert saved_result['cfg_id'] == cfg_id(cfg), f'{folder}: result.json belongs to another configuration'
        return saved_result
    folder.mkdir(parents=True, exist_ok=True)
    training_utils.save_atomic(cfg, folder / 'config.json', keep_backup=False)
    train_ids, val_ids = fold_ids(fold)
    train_dataset = VisDroneDataset(str(DATA), 'train', coco_train, train_ids, augment=True, hflip=cfg['hflip'])
    model = build_model(cfg).to(DEV)
    num_params_millions = sum(param.numel() for param in model.parameters()) / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['step_gamma'])
    scaler = torch.amp.GradScaler('cuda', enabled=DEV.type == 'cuda')
    state = dict(epoch=0, batch=0, running=0.0, history=[], best=-1.0, best_epoch=0, seconds=0.0)
    last_path = folder / 'last.pth'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
        assert checkpoint['cfg_id'] == cfg_id(cfg)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        state = checkpoint['state']
        torch.set_rng_state(checkpoint['rng_torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        print(f"{cfg['name']}/fold{fold}: resume at epoch {state['epoch'] + 1}, batch {state['batch']}", flush=True)

    def save_checkpoint(path, full=True):
        checkpoint_data = dict(cfg_id=cfg_id(cfg), model=model.state_dict(), epoch=state['epoch'])
        if full:
            checkpoint_data.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), state=state,
                                   rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all())
        training_utils.save_atomic(checkpoint_data, path, keep_backup=False)

    print(f"{cfg['name']}/fold{fold}: {len(train_dataset)} training frames, {len(val_ids)} held-out frames, {num_params_millions:.1f}M parameters", flush=True)
    while state['epoch'] < cfg['epochs']:
        epoch = state['epoch']
        model.train()
        order = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(cfg['seed'] + epoch)).tolist()
        loader = DataLoader(Subset(train_dataset, order[state['batch']:]), batch_size=cfg['bs'], shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=detection_collate_fn, worker_init_fn=worker_init, pin_memory=True)
        interval_start = time.time()
        progress_bar = tqdm(loader, desc=f"{cfg['name']}/fold{fold} {epoch + 1}/{cfg['epochs']}", mininterval=60)
        for images, targets in progress_bar:
            images = to_float(images)
            targets = [{key: value.to(DEV, non_blocking=True) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type, enabled=DEV.type == 'cuda'):
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())
            if not torch.isfinite(total_loss).item():
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}, batch {state['batch'] + 1}")
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['clip'])
            scaler.step(optimizer)
            scaler.update()
            state['running'] += total_loss.item()
            state['batch'] += 1
            if state['batch'] % cfg['save_every'] == 0:
                state['seconds'] += time.time() - interval_start
                interval_start = time.time()
                save_checkpoint(last_path)
        scheduler.step()
        state['seconds'] += time.time() - interval_start
        train_loss = state['running'] / len(train_dataset)
        eval_start = time.time()
        metrics = evaluate(model, val_ids)
        state['history'].append(dict(epoch=epoch + 1, train_loss=train_loss, lr=optimizer.param_groups[0]['lr'],
                                     epoch_seconds=state['seconds'], eval_seconds=time.time() - eval_start,
                                     **{key: value for key, value in metrics.items() if key != 'per_class'}))
        print(f"{cfg['name']}/fold{fold} epoch {epoch + 1}: loss {train_loss:.4f} | mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} "
              f"APs {metrics['APs']:.4f} | {state['seconds'] / 60:.1f} min", flush=True)
        if metrics['mAP'] > state['best']:
            state['best'], state['best_epoch'] = metrics['mAP'], epoch + 1
            state['best_metrics'] = metrics
            save_checkpoint(folder / 'best.pth', full=False)
        state.update(epoch=epoch + 1, batch=0, running=0.0, seconds=0.0)
        save_checkpoint(last_path)
    model.load_state_dict(torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)['model'])
    model.to(DEV)
    result = dict(cfg=cfg, cfg_id=cfg_id(cfg), fold=fold, best_epoch=state['best_epoch'], metrics=state['best_metrics'],
                  history=state['history'], n_params_M=num_params_millions, ms_per_img=measure_speed(model))
    training_utils.save_atomic(result, result_path, keep_backup=False)
    last_path.unlink(missing_ok=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return result

def load_model(name, fold):
    """Best checkpoint of run `name` on `fold` as (model in eval mode on the device, its configuration)."""
    folder = OUT / name / f'fold{fold}'
    cfg = json.loads((folder / 'config.json').read_text())
    checkpoint = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
    assert checkpoint['cfg_id'] == cfg_id(cfg)
    model = build_model(cfg)
    model.load_state_dict(checkpoint['model'])
    return model.to(DEV).eval(), cfg

def official_validation(name, fold):
    """One evaluation of a selected checkpoint on the official validation set (threaded uint8 decoding of Faster R-CNN's evaluator)."""
    path = OUT / name / f'fold{fold}' / 'official_val.json'
    if path.exists():
        return json.loads(path.read_text())
    import final_evaluation
    model, _ = load_model(name, fold)
    ids = [int(image_info['id']) for image_info in coco_val['images']]
    start_time = time.time()
    predictions = final_evaluation.predict(model, 'val', ids, 1)
    rows = [(image_id, dict(boxes=torch.from_numpy(predictions[image_id]['boxes']), scores=torch.from_numpy(predictions[image_id]['scores']),
                            labels=torch.from_numpy(predictions[image_id]['labels']) + 1)) for image_id in ids]
    evaluation = dict(metrics=training_utils.coco_metrics(rows, training_utils.make_gt(coco_val, ids)), seconds=time.time() - start_time)
    training_utils.save_atomic(evaluation, path, keep_backup=False)
    del model
    torch.cuda.empty_cache()
    return evaluation
