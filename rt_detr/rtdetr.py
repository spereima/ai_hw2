"""
RT-DETR R50-vd is based on a COCO-pretrained model from HuggingFace, adapted to detect 8 object classes.
"""
import os
import sys
import json
import math
import time
import copy
import hashlib
import types
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrConfig
import transformers.loss.loss_rt_detr as rtdetr_loss
from transformers.image_transforms import center_to_corners_format
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs'
sys.path.insert(0, str(ROOT / 'shared'))

import training_utils

DATA, coco_train, coco_val = training_utils.DATA, training_utils.coco_train, training_utils.coco_val
FOLDS, FOLDS_SHA = training_utils.FOLDS, training_utils.FOLDS_SHA
DEV = torch.device('cuda')
MODEL_ID = 'PekingU/rtdetr_r50vd'
CATEGORY_IDS = sorted(int(category['id']) for category in coco_train['categories'])
CATEGORY_TO_INDEX = {category_id: index for index, category_id in enumerate(CATEGORY_IDS)}
NUM_CLASSES = len(CATEGORY_IDS)
ID2LABEL = {index: next(category['name'] for category in coco_train['categories'] if category['id'] == category_id)
            for index, category_id in enumerate(CATEGORY_IDS)}
NUM_WORKERS = 4

BASE = dict(name='baseline', seed=42, epochs=6, batch_size=4, lr_backbone=1e-5, lr_transformer=1e-4, wd=1e-4, step_size=4, step_gamma=0.1,
            clip=1.0, height=640, width=640, num_queries=300, matcher_class_cost=2.0, matcher_bbox_cost=5.0, matcher_giou_cost=2.0,
            weight_loss_bbox=5.0, weight_loss_giou=2.0, num_denoising=100, warmup_iters=0, ema=False, ema_decay=0.9999,
            ema_ramp=2000, ignore_ioa=0.5, score_thr=0.01, hflip=0.5, save_every=1000)

_ORIG_VFL = rtdetr_loss.RTDetrLoss.loss_labels_vfl

def loss_labels_vfl_ignore(self, outputs, targets, indices, num_boxes, log=True):
    """transformers' RTDetrLoss.loss_labels_vfl, with weight 0 for unmatched queries inside ignore regions (pred box IoA >= ignore_ioa)."""
    source_idx = self._get_source_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][source_idx]
    target_boxes = torch.cat([target_per_image['boxes'][target_idx] for target_per_image, (_, target_idx) in zip(targets, indices)], dim=0)
    ious, _ = rtdetr_loss.box_iou(center_to_corners_format(src_boxes.detach()), center_to_corners_format(target_boxes))
    ious = torch.diag(ious)
    src_logits = outputs['logits']
    target_classes_original = torch.cat([target_per_image['class_labels'][target_idx] for target_per_image, (_, target_idx) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
    target_classes[source_idx] = target_classes_original
    target = torch.nn.functional.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
    target_score_original = torch.zeros_like(target_classes, dtype=src_logits.dtype)
    target_score_original[source_idx] = ious.to(target_score_original.dtype)
    target_score = target_score_original.unsqueeze(-1) * target
    pred_score = torch.sigmoid(src_logits.detach())
    weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
    ignore_ioa = getattr(self, 'ignore_ioa', 0)
    if ignore_ioa:
        is_matched = torch.zeros(src_logits.shape[:2], dtype=torch.bool, device=src_logits.device)
        is_matched[source_idx] = True
        pred_boxes_xyxy = center_to_corners_format(outputs['pred_boxes'].detach().float())
        for batch_idx, target_per_image in enumerate(targets):
            ignored_boxes = target_per_image.get('ignored_boxes')
            if ignored_boxes is None or not len(ignored_boxes):
                continue
            in_ignored_region = training_utils.inside_ignored(pred_boxes_xyxy[batch_idx], center_to_corners_format(ignored_boxes.float()), ignore_ioa)
            ignored_queries = in_ignored_region & ~is_matched[batch_idx]
            if ignored_queries.any():
                weight[batch_idx, ignored_queries] = 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
    loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
    return {'loss_vfl': loss}

rtdetr_loss.RTDetrLoss.loss_labels_vfl = loss_labels_vfl_ignore
_ORIG_INIT = rtdetr_loss.RTDetrLoss.__init__
def _init_with_ignore(self, config):
    _ORIG_INIT(self, config)
    self.ignore_ioa = getattr(config, 'ignore_ioa', 0)
rtdetr_loss.RTDetrLoss.__init__ = _init_with_ignore

_PROCESSORS = {}
def processor(cfg):
    """RT-DETR image processor for the configured input size (cached per size)."""
    size_key = (cfg['height'], cfg['width'])
    if size_key not in _PROCESSORS:
        image_processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
        image_processor.size = {'height': cfg['height'], 'width': cfg['width']}
        _PROCESSORS[size_key] = image_processor
    return _PROCESSORS[size_key]

class Collate:
    """The notebook's rtdetr_collate_fn: COCO-style annotations (category index 0..7) through the RT-DETR image processor; plus the
    ignore regions as normalised (cx, cy, w, h) for CHANGED 3."""
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, batch):
        image_processor = processor(self.cfg)
        images, annotations, original_targets, ignored_per_image = [], [], [], []
        for image, target in batch:
            image_array = image.permute(1, 2, 0).numpy()
            image_height, image_width = image_array.shape[:2]
            images.append(image_array)
            coco_boxes = []
            for (x1, y1, x2, y2), label in zip(target['boxes'].tolist(), target['labels'].tolist()):
                coco_boxes.append(dict(id=len(coco_boxes), image_id=int(target['image_id']), category_id=CATEGORY_TO_INDEX[int(label)],
                                       bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0))
            annotations.append(dict(image_id=int(target['image_id']), annotations=coco_boxes))
            original_targets.append(target)
            ignored_xyxy = target['ignored_boxes']
            ignored_per_image.append(torch.stack([(ignored_xyxy[:, 0] + ignored_xyxy[:, 2]) / 2 / image_width,
                                                  (ignored_xyxy[:, 1] + ignored_xyxy[:, 3]) / 2 / image_height,
                                                  (ignored_xyxy[:, 2] - ignored_xyxy[:, 0]) / image_width,
                                                  (ignored_xyxy[:, 3] - ignored_xyxy[:, 1]) / image_height], 1)
                                     if len(ignored_xyxy) else torch.zeros(0, 4))
        encoding = image_processor(images=images, annotations=annotations, return_tensors='pt')
        for label_dict, ignored_boxes in zip(encoding['labels'], ignored_per_image):
            label_dict['ignored_boxes'] = ignored_boxes
        encoding['original_targets'] = original_targets
        return encoding

def fold_ids(fold):
    """(training frame ids, held-out frame ids) of a fold."""
    return training_utils.fold_ids(fold)

def as_list(batch):
    return batch

def build_model(cfg):
    """RT-DETR R50-vd (COCO weights) with a new 8-class head and the query / matching settings of the configuration."""
    training_utils.seed_all(cfg['seed'] + cfg['fold'])
    model_config = RTDetrConfig.from_pretrained(MODEL_ID)
    model_config.num_labels = NUM_CLASSES
    model_config.id2label = ID2LABEL
    model_config.label2id = {name: index for index, name in ID2LABEL.items()}
    for config_key in ('num_queries', 'matcher_class_cost', 'matcher_bbox_cost', 'matcher_giou_cost', 'weight_loss_bbox', 'weight_loss_giou'):
        setattr(model_config, config_key, cfg[config_key])
    model_config.num_denoising = cfg['num_denoising']
    model_config.ignore_ioa = cfg['ignore_ioa']
    return RTDetrForObjectDetection.from_pretrained(MODEL_ID, config=model_config, ignore_mismatched_sizes=True)

class EMA:
    """Exponential moving average of the weights with the warm-up ramp of the official RT-DETR (decay * (1 - exp(-updates / ramp)))."""
    def __init__(self, model, decay, ramp):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay, self.ramp, self.updates = decay, ramp, 0
    @torch.no_grad()
    def update(self, model):
        """One EMA step. transformers ties the decoder's class / box heads to the top-level heads, so state_dict() lists the same
        tensor under two names; each tensor is updated once. CHANGED after the runs: the loop used for every RT-DETR run in this
        folder updated those tied head tensors twice per step (decay d**2 for them) - identical in all runs."""
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp))
        model_state = model.state_dict()
        already_updated = set()
        for name, ema_tensor in self.module.state_dict().items():
            if not ema_tensor.dtype.is_floating_point or ema_tensor.data_ptr() in already_updated:
                continue
            already_updated.add(ema_tensor.data_ptr())
            ema_tensor.mul_(decay).add_(model_state[name].detach(), alpha=1 - decay)

def move(batch):
    """Processor batch -> dict of model inputs (pixel_values, labels, optional pixel_mask) on the device."""
    labels = [{key: (value.to(DEV) if torch.is_tensor(value) else value) for key, value in target.items()} for target in batch['labels']]
    model_inputs = dict(pixel_values=batch['pixel_values'].to(DEV, non_blocking=True), labels=labels)
    if 'pixel_mask' in batch:
        model_inputs['pixel_mask'] = batch['pixel_mask'].to(DEV, non_blocking=True)
    return model_inputs

@torch.inference_mode()
def predict_arrays(model, cfg, arrays, flip=False):
    """Detections in original-frame coordinates for a list of uint8 HxWx3 arrays: boxes, scores, labels 1..8 (score >= score_thr)."""
    image_processor = processor(cfg)
    if flip:
        arrays = [array[:, ::-1].copy() for array in arrays]
    encoding = image_processor(images=arrays, return_tensors='pt')
    with torch.autocast('cuda'):
        model_output = model(pixel_values=encoding['pixel_values'].to(DEV))
    target_sizes = torch.tensor([array.shape[:2] for array in arrays], device=DEV)
    detections = image_processor.post_process_object_detection(model_output, threshold=cfg['score_thr'], target_sizes=target_sizes)
    rows = []
    for array, detection in zip(arrays, detections):
        boxes = detection['boxes'].float().cpu()
        if flip:
            width = array.shape[1]
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], 1)
        rows.append(dict(boxes=boxes, scores=detection['scores'].float().cpu(), labels=detection['labels'].cpu() + 1))
    return rows

def evaluate(model, cfg, ids, coco=None, split='train', bs=8, loader=None):
    """Faster R-CNN COCO evaluation. Frames are decoded in 4 threads of this process (uint8; no worker processes, so no second copy of
    the data in memory and no start-up per call), then go through the RT-DETR processor in batches of `bs`."""
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    from PIL import Image
    coco = coco or coco_train
    image_infos = {int(image_info['id']): image_info for image_info in coco['images']}
    ids = [int(image_id) for image_id in ids]
    def load(image_id):
        with Image.open(DATA / split / image_infos[image_id]['file_name']) as image:
            return np.array(image.convert('RGB'))
    model.eval()
    rows = []
    with ThreadPoolExecutor(4) as pool:
        queue = deque(pool.submit(load, image_id) for image_id in ids[:2 * bs])
        next_to_submit = len(queue)
        for start in tqdm(range(0, len(ids), bs), desc='Evaluate', leave=False, mininterval=30):
            batch = ids[start:start + bs]
            arrays = []
            for _ in batch:
                arrays.append(queue.popleft().result())
                if next_to_submit < len(ids):
                    queue.append(pool.submit(load, ids[next_to_submit]))
                    next_to_submit += 1
            predictions = predict_arrays(model, cfg, arrays)
            rows += [(image_id, prediction) for image_id, prediction in zip(batch, predictions)]
    return training_utils.coco_metrics(rows, training_utils.make_gt(coco, ids))

@torch.inference_mode()
def measure_speed(model, cfg, num_frames=40, warmup_frames=8):
    """ms per official-validation frame, batch 1, AMP: preprocessing on the GPU tensor excluded, model forward only (Faster R-CNN method)."""
    dataset = training_utils.VisDroneDataset(str(DATA), 'val', coco_val, augment=False)
    image_processor = processor(cfg)
    pixel_values_list = [image_processor(images=[dataset[(i * 7) % len(dataset)][0].permute(1, 2, 0).numpy()],
                                         return_tensors='pt')['pixel_values'].to(DEV)
                         for i in range(num_frames + warmup_frames)]
    model.eval()
    seconds_per_frame = []
    for pixel_values in pixel_values_list:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.autocast('cuda'):
            model(pixel_values=pixel_values)
        torch.cuda.synchronize()
        seconds_per_frame.append(time.perf_counter() - start_time)
    return float(np.mean(seconds_per_frame[warmup_frames:]) * 1000)

def cfg_id(cfg):
    """Hash of the configuration (without save_every); a checkpoint / result belongs to the configuration with the same id."""
    stable = {key: value for key, value in cfg.items() if key != 'save_every'}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()

def make_cfg(overrides, fold):
    """BASE configuration with the experiment overrides, the fold number and the hash of the folds file."""
    return {**BASE, **overrides, 'fold': fold, 'folds': FOLDS_SHA}

class EpochSampler(torch.utils.data.Sampler):
    """Yields the index order set for the current epoch; lets one DataLoader with persistent workers serve every epoch
    (the workers are started once per run instead of once per epoch; the order and the data are unchanged)."""
    def __init__(self):
        self.indices = []
    def set(self, indices):
        self.indices = list(indices)
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def worker_init(_):
    """Seed every DataLoader worker differently (derived from the worker seed)."""
    worker_info = torch.utils.data.get_worker_info()
    training_utils.seed_all(worker_info.seed % 2 ** 31)

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
    train_dataset = training_utils.VisDroneDataset(str(DATA), 'train', coco_train, train_ids, augment=True, hflip=cfg['hflip'])
    model = build_model(cfg).to(DEV)
    backbone_params = [param for param_name, param in model.named_parameters() if param.requires_grad and 'backbone' in param_name]
    other_params = [param for param_name, param in model.named_parameters() if param.requires_grad and 'backbone' not in param_name]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': cfg['lr_backbone']}, {'params': other_params, 'lr': cfg['lr_transformer']}],
                                  weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['step_gamma'])
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(model, cfg['ema_decay'], cfg['ema_ramp']) if cfg['ema'] else None
    base_lrs = [cfg['lr_backbone'], cfg['lr_transformer']]
    state = dict(epoch=0, batch=0, step=0, running=0.0, n=0, history=[], best=-1.0, best_epoch=0, seconds=0.0)
    last_path = folder / 'last.pth'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
        assert checkpoint['cfg_id'] == cfg_id(cfg)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        if ema is not None:
            ema.module.load_state_dict(checkpoint['ema'])
            ema.updates = checkpoint['ema_updates']
        state = checkpoint['state']
        torch.set_rng_state(checkpoint['rng_torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        print(f"{cfg['name']}/fold{fold}: resume at epoch {state['epoch'] + 1}, batch {state['batch']}", flush=True)

    def save_checkpoint(path, full=True):
        eval_model = ema.module if ema is not None else model
        checkpoint_data = dict(cfg_id=cfg_id(cfg), model=eval_model.state_dict(), epoch=state['epoch'])
        if full:
            checkpoint_data.update(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                   scaler=scaler.state_dict(), state=state, rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                                   ema=None if ema is None else ema.module.state_dict(), ema_updates=None if ema is None else ema.updates)
        training_utils.save_atomic(checkpoint_data, path, keep_backup=False)

    collate = Collate(cfg)
    sampler = EpochSampler()
    loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], sampler=sampler, num_workers=NUM_WORKERS, collate_fn=collate,
                        worker_init_fn=worker_init, pin_memory=True, persistent_workers=True)
    print(f"{cfg['name']}/fold{fold}: {len(train_dataset)} training frames, {len(val_ids)} held-out frames", flush=True)
    while state['epoch'] < cfg['epochs']:
        epoch = state['epoch']
        model.train()
        order = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(cfg['seed'] + fold * 100 + epoch)).tolist()
        sampler.set(order[state['batch'] * cfg['batch_size']:])
        interval_start = time.time()
        for batch in tqdm(loader, desc=f"{cfg['name']}/fold{fold} {epoch + 1}/{cfg['epochs']}", mininterval=60):
            if cfg['warmup_iters'] and state['step'] < cfg['warmup_iters']:
                warmup_factor = (state['step'] + 1) / cfg['warmup_iters']
                for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                    param_group['lr'] = base_lr * warmup_factor * (cfg['step_gamma'] ** (epoch // cfg['step_size']))
            inputs = move(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss = model(**inputs).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}, batch {state['batch'] + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            state['running'] += loss.item()
            state['n'] += 1
            state['batch'] += 1
            state['step'] += 1
            if state['batch'] % cfg['save_every'] == 0:
                state['seconds'] += time.time() - interval_start
                interval_start = time.time()
                save_checkpoint(last_path)
        scheduler.step()
        state['seconds'] += time.time() - interval_start
        eval_start = time.time()
        eval_model = ema.module if ema is not None else model
        metrics = evaluate(eval_model, cfg, val_ids)
        state['history'].append(dict(epoch=epoch + 1, train_loss=state['running'] / max(1, state['n']), lr=optimizer.param_groups[1]['lr'],
                                     epoch_seconds=state['seconds'], eval_seconds=time.time() - eval_start,
                                     **{key: value for key, value in metrics.items() if key != 'per_class'}))
        print(f"{cfg['name']}/fold{fold} epoch {epoch + 1}: loss {state['history'][-1]['train_loss']:.4f} | mAP {metrics['mAP']:.4f} "
              f"AP50 {metrics['AP50']:.4f} APs {metrics['APs']:.4f} | {state['seconds'] / 60:.1f} min", flush=True)
        if metrics['mAP'] > state['best']:
            state['best'], state['best_epoch'], state['best_metrics'] = metrics['mAP'], epoch + 1, metrics
            save_checkpoint(folder / 'best.pth', full=False)
        state.update(epoch=epoch + 1, batch=0, running=0.0, n=0, seconds=0.0)
        save_checkpoint(last_path)
    model.load_state_dict(torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)['model'])
    result = dict(cfg=cfg, cfg_id=cfg_id(cfg), fold=fold, best_epoch=state['best_epoch'], metrics=state['best_metrics'],
                  history=state['history'], n_params_M=sum(param.numel() for param in model.parameters()) / 1e6,
                  train_minutes=sum(epoch_record['epoch_seconds'] for epoch_record in state['history']) / 60, ms_per_img=measure_speed(model, cfg))
    training_utils.save_atomic(result, result_path, keep_backup=False)
    last_path.unlink(missing_ok=True)
    del model, optimizer, ema, loader
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
    """Evaluate the best checkpoint of run `name` on the official validation set once (result cached in official_val.json)."""
    path = OUT / name / f'fold{fold}' / 'official_val.json'
    if path.exists():
        return json.loads(path.read_text())
    model, cfg = load_model(name, fold)
    ids = [int(image_info['id']) for image_info in coco_val['images']]
    start_time = time.time()
    evaluation = dict(metrics=evaluate(model, cfg, ids, coco_val, 'val', bs=4), seconds=time.time() - start_time)
    training_utils.save_atomic(evaluation, path, keep_backup=False)
    del model
    torch.cuda.empty_cache()
    return evaluation
