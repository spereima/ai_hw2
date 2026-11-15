"""
Faster R-CNN v2 with COCO-pretrained weights and the same training settings as Faster R-CNN
(lr_scheduler, SGD, 6 epochs, batch size 4, 720/1280 resize).

Two experiments:
- v2_lr_scheduler: Faster R-CNN v2 without scale augmentation.
- v2_lr_scheduler_scale_aug: Faster R-CNN v2 with scale augmentation to improve object detection at different image sizes.
"""

import os
import io
import json
import time
import random
import math
import hashlib
import shutil
import tempfile
import types
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from torchvision.ops import generalized_box_iou_loss
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models.detection.roi_heads as roi_heads_module
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'shared'))
from visdrone_data import VisDrone, collate
from training_utils import (DATA, FOLDS_CSV, MAX_DETS, coco_train, coco_val, folds, CATS, CAT_IDS, CAT_TO_LABEL,
                            LABEL_TO_CAT, N_CLASSES, FOLDS_SHA, make_gt, coco_metrics, save_atomic)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEV.type == 'cuda'
NUM_WORKERS = 2


def seed_all(seed):
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def make_loader(dataset, batch_size, shuffle=False, workers=NUM_WORKERS):
    """DataLoader for evaluation (no sampler, no persistent workers)."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=AMP, collate_fn=collate,
        persistent_workers=False)


class EpochBatches(Sampler):
    def __init__(self, length, batch_size, seed, epoch, start=0):
        indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        self.batches = [indices[i:i + batch_size] for i in range(0, length, batch_size)]
        self.start = start

    def __iter__(self):
        return iter(self.batches[self.start:])

    def __len__(self):
        return len(self.batches) - self.start


def train_loader(dataset, cfg, epoch, batch_start):
    """Training DataLoader of one epoch that can resume at batch `batch_start`."""
    dataset.epoch = epoch
    batches = EpochBatches(len(dataset), cfg['bs'], cfg['seed'], epoch, batch_start)
    return DataLoader(dataset, batch_sampler=batches, num_workers=NUM_WORKERS, pin_memory=AMP, collate_fn=collate,
                      persistent_workers=False, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))


def fold_ids(fold):
    """Image ids (train, validation) of a cross-validation fold; videos never cross the split."""
    train_ids = folds.loc[folds.Fold != fold, 'id'].astype(int)
    val_ids = folds.loc[folds.Fold == fold, 'id'].astype(int)
    assert len(train_ids) and len(val_ids) and not (set(train_ids) & set(val_ids))
    assert not set(folds.loc[folds.Fold != fold, 'video']) & set(folds.loc[folds.Fold == fold, 'video'])
    return train_ids, val_ids


class VisDroneScaleAug(VisDrone):
    """V2_MS only: random downscaling of a training frame. The test set contains 382- and 540-pixel-high frames, lower than any
    training frame; this lets the detector see such resolutions during training. Boxes and ignore regions are scaled with it."""
    def __init__(self, *args, p_down=0.0, down_range=(0.35, 0.7), **kwargs):
        super().__init__(*args, **kwargs)
        self.p_down, self.down_range = p_down, down_range

    def __getitem__(self, i):
        img, target = super().__getitem__(i)
        if self.train and self.p_down > 0:
            rng = random.Random(self.seed * 7919 + self.epoch * 104729 + i)
            if rng.random() < self.p_down:
                scale = rng.uniform(*self.down_range)
                h, w = img.shape[-2:]
                new_h, new_w = max(32, round(h * scale)), max(32, round(w * scale))
                img = F.interpolate(img[None], size=(new_h, new_w), mode='bilinear', antialias=True,
                                    align_corners=False)[0]
                scale_x, scale_y = new_w / w, new_h / h
                for key in ('boxes', 'ignored'):
                    scaled_boxes = target[key].clone()
                    scaled_boxes[:, [0, 2]] *= scale_x
                    scaled_boxes[:, [1, 3]] *= scale_y
                    target[key] = scaled_boxes
                keep = (target['boxes'][:, 2] - target['boxes'][:, 0] >= 1) & (
                    target['boxes'][:, 3] - target['boxes'][:, 1] >= 1)
                target['boxes'], target['labels'] = target['boxes'][keep], target['labels'][keep]
        return img, target

BASE = dict(name='baseline', seed=42, epochs=6, bs=4, lr=.01, wd=1e-4, warmup=300,
            scheduler='cosine', min_size=720, max_size=1280,
            rpn_pre_train=2000, rpn_post_train=2000, rpn_pre_test=1000, rpn_post_test=1000,
            rpn_batch=256, roi_batch=512, roi_pos_frac=.25,
            det_per_img=500, score_thr=.01, nms_thr=.5,
            box_loss='smooth_l1', giou_w=2., ignore_ioa=.5, trainable_layers=3,
            save_every=100)
LR_SCHEDULER = dict(scheduler='multistep')

_ORIG_LOSS = roi_heads_module.fastrcnn_loss
_BOX_WEIGHTS = (10., 10., 5., 5.)


def decode_unit(deltas):
    """Turn box regression deltas into boxes around the origin, so a GIoU loss can compare them."""
    deltas = deltas.float() / deltas.new_tensor(_BOX_WEIGHTS)
    dw = deltas[:, 2].clamp(max=math.log(1000 / 16))
    dh = deltas[:, 3].clamp(max=math.log(1000 / 16))
    cx, cy = deltas[:, 0], deltas[:, 1]
    w, h = dw.exp(), dh.exp()
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def make_giou_loss(weight):
    """Replacement for torchvision's fastrcnn_loss with a GIoU box loss scaled by `weight`."""
    def fastrcnn_giou(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        cls_loss = F.cross_entropy(class_logits, labels)
        positive = torch.where(labels > 0)[0]
        num_rois = class_logits.shape[0]
        pred_deltas = box_regression.reshape(num_rois, -1, 4)[positive, labels[positive]]
        if len(positive):
            reg_loss = generalized_box_iou_loss(
                decode_unit(pred_deltas),
                decode_unit(regression_targets[positive]),
                reduction='sum')
            reg_loss = reg_loss / max(labels.numel(), 1)
        else:
            reg_loss = box_regression.sum() * 0
        return cls_loss, weight * reg_loss
    return fastrcnn_giou


def ignore_negatives(candidates, labels, ignored, threshold):
    """Set negative labels to -1 (ignored) for candidates lying mostly inside an ignore region."""
    if not len(ignored):
        return labels
    negative = torch.where(labels == 0)[0]
    if not len(negative):
        return labels
    labels = labels.clone()
    for chunk in negative.split(2048):
        boxes = candidates[chunk]
        left_top = torch.maximum(boxes[:, None, :2], ignored[None, :, :2])
        right_bottom = torch.minimum(boxes[:, None, 2:], ignored[None, :, 2:])
        overlap_size = (right_bottom - left_top).clamp(min=0)
        intersection = overlap_size[..., 0] * overlap_size[..., 1]
        area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6))
        inside = (intersection / area[:, None]).amax(dim=1) >= threshold
        labels[chunk[inside]] = -1
    return labels


def add_ignore_aware_training(model, threshold):
    """Patch the model so ignore regions are rescaled with the images and excluded from negative sampling."""
    standard_transform = model.transform.forward

    def transform_with_ignore(self, images, targets=None):
        original_sizes = [image.shape[-2:] for image in images]
        transformed, targets = standard_transform(images, targets)
        if targets is not None:
            for target, (old_h, old_w), (new_h, new_w) in zip(targets, original_sizes, transformed.image_sizes):
                boxes = target['ignored'].clone()
                boxes[:, [0, 2]] *= new_w / old_w
                boxes[:, [1, 3]] *= new_h / old_h
                target['ignored'] = boxes
        return transformed, targets
    model.transform.forward = types.MethodType(transform_with_ignore, model.transform)
    standard_rpn_assign = model.rpn.assign_targets_to_anchors

    def rpn_assign(self, anchors, targets):
        labels, matched_boxes = standard_rpn_assign(anchors, targets)
        for i, target in enumerate(targets):
            labels[i] = ignore_negatives(anchors[i], labels[i], target['ignored'], threshold)
        return labels, matched_boxes
    model.rpn.assign_targets_to_anchors = types.MethodType(rpn_assign, model.rpn)
    standard_roi_assign = model.roi_heads.assign_targets_to_proposals
    standard_roi_sample = model.roi_heads.select_training_samples

    def roi_assign(self, proposals, gt_boxes, gt_labels):
        matched_ids, labels = standard_roi_assign(proposals, gt_boxes, gt_labels)
        for i, ignored in enumerate(self._ignored_for_sampling):
            labels[i] = ignore_negatives(proposals[i], labels[i], ignored, threshold)
        return matched_ids, labels

    def roi_sample(self, proposals, targets):
        self._ignored_for_sampling = [target['ignored'] for target in targets]
        try:
            return standard_roi_sample(proposals, targets)
        finally:
            del self._ignored_for_sampling
    model.roi_heads.assign_targets_to_proposals = types.MethodType(roi_assign, model.roi_heads)
    model.roi_heads.select_training_samples = types.MethodType(roi_sample, model.roi_heads)
    return model


def add_train_scales(model, cfg):
    """CHANGED (V2_MS only): in training mode the shorter side is drawn from cfg['train_min_sizes'] with max side
    cfg['train_max_size']; in eval mode exactly the original single size (min_size / max_size) is used."""
    sizes = tuple(cfg['train_min_sizes'])
    inner = model.transform.forward

    def forward(self, images, targets=None):
        if self.training:
            self.min_size, self.max_size = sizes, cfg['train_max_size']
        else:
            self.min_size, self.max_size = (cfg['min_size'],), cfg['max_size']
        return inner(images, targets)
    model.transform.forward = types.MethodType(forward, model.transform)
    return model


def build_model(cfg):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=cfg['trainable_layers'],
        min_size=cfg['min_size'], max_size=cfg['max_size'],
        rpn_pre_nms_top_n_train=cfg['rpn_pre_train'], rpn_post_nms_top_n_train=cfg['rpn_post_train'],
        rpn_pre_nms_top_n_test=cfg['rpn_pre_test'], rpn_post_nms_top_n_test=cfg['rpn_post_test'],
        rpn_batch_size_per_image=cfg['rpn_batch'],
        box_batch_size_per_image=cfg['roi_batch'], box_positive_fraction=cfg['roi_pos_frac'],
        box_detections_per_img=cfg['det_per_img'], box_score_thresh=cfg['score_thr'],
        box_nms_thresh=cfg['nms_thr'])
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, N_CLASSES)
    roi_heads_module.fastrcnn_loss = (make_giou_loss(cfg['giou_w']) if cfg['box_loss'] == 'giou' else _ORIG_LOSS)
    add_ignore_aware_training(model, cfg['ignore_ioa'])
    if cfg.get('train_min_sizes'):
        add_train_scales(model, cfg)
    return model


@torch.inference_mode()
def predict(model, loader, verbose=False):
    """Run the model on a loader; returns a list of (image id, prediction dict on CPU)."""
    model.eval()
    rows = []
    for images, targets in tqdm(loader, desc='Predict', leave=False, disable=not verbose):
        with torch.autocast(device_type=DEV.type, enabled=AMP):
            preds = model([image.to(DEV, non_blocking=True) for image in images])
        for prediction, target in zip(preds, targets):
            rows.append((int(target['image_id']), {k: v.detach().cpu() for k, v in prediction.items()}))
    return rows


@torch.inference_mode()
def measure_speed(model, dataset, num_images=40, warmup_steps=8):
    """Mean milliseconds per image (batch size 1) after `warmup_steps` warm-up images."""
    model.eval()
    images = [dataset[(i * 7) % len(dataset)][0].to(DEV) for i in range(min(num_images + warmup_steps, len(dataset)))]
    times = []
    for image in images:
        if DEV.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type=DEV.type, enabled=AMP):
            model([image])
        if DEV.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.mean(times[min(warmup_steps, max(0, len(times) - 1)):]) * 1000)


def cfg_id(cfg):
    """Hash of a training config (without `save_every`); stored in checkpoints to detect a changed config."""
    stable = {k: v for k, v in cfg.items() if k != 'save_every'}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def lr_schedule(cfg, total_steps):
    """Learning-rate factor per step: linear warm-up, then the cosine or multistep schedule of the config."""
    warmup_steps = max(1, min(cfg['warmup'], total_steps // 4))

    def factor(step):
        step = min(step + 1, total_steps)
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        if cfg['scheduler'] == 'cosine':
            return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))
        if cfg['scheduler'] == 'multistep':
            return 1. if progress < .6 else (.1 if progress < .85 else .01)
        raise ValueError('Unknown scheduler: ' + cfg['scheduler'])
    return factor


def read_checkpoint(folder, identity):
    """Load last.pth (or its backup) if it belongs to this config; otherwise None."""
    for path in [folder / 'last.pth', folder / 'last.pth.bak']:
        if not path.exists():
            continue
        try:
            checkpoint_data = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint_data.get('cfg_id') != identity:
                raise ValueError(f'Checkpoint config mismatch: {path}')
            return checkpoint_data
        except (EOFError, RuntimeError, OSError, ValueError, KeyError) as error:
            print('Cannot use checkpoint:', path, '|', error)
    return None


def make_train_ds(cfg, ids):
    """Training dataset of the given images, with scale augmentation when the config asks for it."""
    if cfg.get('p_down', 0) > 0:
        return VisDroneScaleAug(DATA, 'train', coco_train, ids, train=True, seed=cfg['seed'], p_down=cfg['p_down'],
                                down_range=tuple(cfg['down_range']))
    return VisDrone(DATA, 'train', coco_train, ids, train=True, seed=cfg['seed'])


def run_experiment(overrides, fold):
    """Train one configuration on one fold (resumable), evaluate every epoch and return the result dict."""
    cfg = {**BASE, **overrides, 'fold': int(fold), 'folds': FOLDS_SHA}
    run = f"{cfg['name']}/fold{fold}"
    folder = OUT / cfg['name'] / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    identity = cfg_id(cfg)
    config_file, result_file = folder / 'config.json', folder / 'result.json'
    if config_file.exists():
        if cfg_id(json.loads(config_file.read_text())) != identity:
            raise ValueError(f'{run}: different configuration already exists. Use a NEW run name.')
    else:
        if any(folder.iterdir()):
            raise FileExistsError(f'{folder} has files but no config: refusing to overwrite')
        save_atomic(cfg, config_file, keep_backup=False)
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if cfg_id(result['cfg']) != identity:
            raise ValueError('Existing result does not match the requested configuration')
        print(run, ': result already exists - skipping training')
        return result
    seed_all(cfg['seed'])
    if cfg.get('train_min_sizes') and DEV.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.9)
    model = build_model(cfg).to(DEV)
    fold_train_ids, fold_val_ids = fold_ids(fold)
    dataset = make_train_ds(cfg, fold_train_ids)
    fold_val_dataset = VisDrone(DATA, 'train', coco_train, fold_val_ids)
    fold_loader, fold_gt = make_loader(fold_val_dataset, 4), make_gt(coco_train, fold_val_ids)
    batches_per_epoch = math.ceil(len(dataset) / cfg['bs'])
    print(f"{run}: {len(dataset)} training frames, {len(fold_val_dataset)} validation frames, "
          f"{batches_per_epoch} batches per epoch", flush=True)
    total_steps = batches_per_epoch * cfg['epochs']
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=.9, weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule(cfg, total_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    history, epoch, batch_start, sums, seen = [], 0, 0, {}, 0
    checkpoint_data = read_checkpoint(folder, identity)
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data['model'])
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        scaler.load_state_dict(checkpoint_data['scaler'])
        history = checkpoint_data['history']
        epoch, batch_start = checkpoint_data['epoch'], checkpoint_data['batch_start']
        sums, seen = checkpoint_data['sums'], checkpoint_data['seen']
        if 'torch_rng' in checkpoint_data:
            torch.set_rng_state(checkpoint_data['torch_rng'])
            np.random.set_state(checkpoint_data['numpy_rng'])
            random.setstate(checkpoint_data['python_rng'])
            if DEV.type == 'cuda' and checkpoint_data.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(checkpoint_data['cuda_rng'])
        print(f"{run}: resume at epoch {epoch + 1}, batch {batch_start + 1}/{batches_per_epoch}", flush=True)

    def checkpoint(current_epoch, batch, losses, count):
        state = dict(cfg_id=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     history=history, epoch=current_epoch, batch_start=batch, sums=dict(losses), seen=count,
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None)
        save_atomic(state, folder / 'last.pth')

    started = time.time()
    try:
        for current_epoch in range(epoch, cfg['epochs']):
            t_epoch = time.time()
            start_batch = batch_start if current_epoch == epoch else 0
            epoch_loss_sums = defaultdict(float, sums if current_epoch == epoch else {})
            count = seen if current_epoch == epoch else 0
            model.train()
            loader = train_loader(dataset, cfg, current_epoch, start_batch)
            for batch_index, (images, targets) in enumerate(
                tqdm(loader, desc=f"{run} {current_epoch + 1}/{cfg['epochs']}", mininterval=60),
                start=start_batch):
                images = [image.to(DEV, non_blocking=True) for image in images]
                targets = [
                    {
                        'boxes': target['boxes'].to(DEV),
                        'labels': target['labels'].to(DEV),
                        'ignored': target['ignored'].to(DEV)}
                    for target in targets]
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.autocast(device_type=DEV.type, enabled=AMP):
                        losses = model(images, targets)
                        loss = sum(losses.values())
                except torch.OutOfMemoryError:
                    if not cfg.get('train_min_sizes'):
                        raise
                    losses = loss = None
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(
                        f'{run}: CUDA OOM at epoch {current_epoch + 1} batch {batch_index + 1}, retrying once',
                        flush=True)
                    with torch.autocast(device_type=DEV.type, enabled=AMP):
                        losses = model(images, targets)
                        loss = sum(losses.values())
                if not torch.isfinite(loss):
                    checkpoint(current_epoch, batch_index, epoch_loss_sums, count)
                    raise RuntimeError(f"Non-finite loss in {run} epoch {current_epoch + 1} batch {batch_index + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 10.)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not AMP or scaler.get_scale() >= before:
                    scheduler.step()
                for k, v in losses.items():
                    epoch_loss_sums[k] += float(v.detach())
                count += 1
                if (batch_index + 1) % cfg['save_every'] == 0:
                    checkpoint(current_epoch, batch_index + 1, epoch_loss_sums, count)
                if cfg.get('train_min_sizes') and (batch_index + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            checkpoint(current_epoch, batches_per_epoch, epoch_loss_sums, count)
            train_seconds = time.time() - t_epoch
            metrics = coco_metrics(predict(model, fold_loader), fold_gt)
            history.append(dict(epoch=current_epoch + 1,
                                **{k: v / max(count, 1) for k, v in epoch_loss_sums.items()},
                                loss=sum(epoch_loss_sums.values()) / max(count, 1),
                                **{k: metrics[k] for k in ['mAP', 'AP50', 'APs', 'APm', 'APl']},
                                lr=optimizer.param_groups[0]['lr'],
                                train_seconds=train_seconds, epoch_seconds=time.time() - t_epoch))
            if metrics['mAP'] > max((epoch_record['mAP'] for epoch_record in history[:-1]), default=-1.):
                save_atomic(
                    dict(cfg_id=identity, model=model.state_dict(), epoch=current_epoch + 1, metrics=metrics),
                    folder / 'best.pth',
                    keep_backup=False)
            best_epoch = max(history, key=lambda epoch_record: epoch_record['mAP'])['epoch']
            print(f"{run} epoch {current_epoch + 1}: mAP={metrics['mAP']:.4f} AP50={metrics['AP50']:.4f} "
                  f"AP_small={metrics['APs']:.4f} "
                  f"| best epoch {best_epoch} | {(time.time() - t_epoch) / 60:.1f} min", flush=True)
            checkpoint(current_epoch + 1, 0, {}, 0)
            batch_start, sums, seen = 0, {}, 0
        best = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(best['model'])
        result = dict(
            name=cfg['name'], fold=int(fold), cfg=cfg, history=history, best_epoch=best['epoch'],
            metrics=best['metrics'], train_frames=len(dataset), val_frames=len(fold_val_dataset),
            ms_per_img=measure_speed(model, fold_val_dataset),
            n_params_M=sum(param.numel() for param in model.parameters()) / 1e6)
        save_atomic(result, result_file, keep_backup=False)
        return result
    finally:
        del model, optimizer
        if DEV.type == 'cuda':
            torch.cuda.empty_cache()


def load_model(name, fold=0):
    """Rebuild the model of a finished run and load its best.pth (eval mode); returns (model, config)."""
    folder = OUT / name / f'fold{fold}'
    cfg = json.loads((folder / 'config.json').read_text())
    checkpoint_data = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
    if checkpoint_data['cfg_id'] != cfg_id(cfg):
        raise ValueError(f'best.pth does not match the configuration of {name}/fold{fold}')
    model = build_model(cfg)
    model.load_state_dict(checkpoint_data['model'])
    return model.to(DEV).eval(), cfg

EXPERIMENTS = [
    dict(name='v2_lr_scheduler', **LR_SCHEDULER),
    dict(name='v2_lr_scheduler_scale_aug', **LR_SCHEDULER,
         train_min_sizes=[576, 640, 704, 768, 832, 896, 960, 1024, 720], train_max_size=1824,
         p_down=0.2, down_range=[0.35, 0.7]),
]
PILOT_FOLD = 0
