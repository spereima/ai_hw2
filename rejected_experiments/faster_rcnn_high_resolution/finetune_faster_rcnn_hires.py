"""
Higher-resolution, multi-scale fine-tuning of a trained Faster R-CNN lr_scheduler checkpoint.
"""
import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import types
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from visdrone_data import VisDrone, collate


ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'shared'))
DATA_DIR = ROOT / 'datasets/multiple-object-video-detection-hw-2-ucu-ai-2026'
FOLDS_CSV = ROOT / 'shared/sequence_folds.csv'
BASE_RUNS = ROOT / 'faster_rcnn/lr_scheduler'
OUTPUT_DIR = ROOT / 'rejected_experiments/faster_rcnn_high_resolution'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
MAX_DETS = 100
NUM_WORKERS = 2
KEYS = ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']
REF_MIN, REF_MAX = 720, 1280
EVAL_SCALES = {1.0: 4, 1.5: 2}

OVERRIDES = dict(name='hires_ms_ft', epochs=3, bs=2, lr=0.002, min_size=[720, 810, 900, 990, 1080], max_size=1920)


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sha256(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            hasher.update(block)
    return hasher.hexdigest()

def ignore_negatives(candidates, labels, ignored, threshold):
    """Mark negative candidates that lie mostly inside an ignore region with label -1 (excluded from the loss)."""
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
        size = (right_bottom - left_top).clamp(min=0)
        intersection = size[..., 0] * size[..., 1]
        area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6) *
                (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6))
        inside = (intersection / area[:, None]).amax(dim=1) >= threshold
        labels[chunk[inside]] = -1
    return labels

def add_ignore_aware_training(model, threshold):
    """Patch the torchvision model so that anchors and proposals inside ignore regions are not used as negatives."""
    standard_transform = model.transform.forward

    def transform_with_ignore(self, images, targets=None):
        original_sizes = [image.shape[-2:] for image in images]
        transformed, targets = standard_transform(images, targets)
        if targets is not None:
            for target, (old_h, old_w), (new_h, new_w) in zip(
                    targets, original_sizes, transformed.image_sizes):
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

def build_model(cfg, n_classes):
    """Build the Faster R-CNN ResNet-50 FPN from the config, with a new box predictor and ignore-aware training."""
    assert cfg['box_loss'] == 'smooth_l1'
    min_size = cfg['min_size']
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=cfg['trainable_layers'],
        min_size=tuple(min_size) if isinstance(min_size, (list, tuple)) else min_size, max_size=cfg['max_size'],
        rpn_pre_nms_top_n_train=cfg['rpn_pre_train'], rpn_post_nms_top_n_train=cfg['rpn_post_train'],
        rpn_pre_nms_top_n_test=cfg['rpn_pre_test'], rpn_post_nms_top_n_test=cfg['rpn_post_test'],
        rpn_batch_size_per_image=cfg['rpn_batch'],
        box_batch_size_per_image=cfg['roi_batch'], box_positive_fraction=cfg['roi_pos_frac'],
        box_detections_per_img=cfg['det_per_img'], box_score_thresh=cfg['score_thr'],
        box_nms_thresh=cfg['nms_thr'])
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, n_classes)
    add_ignore_aware_training(model, cfg['ignore_ioa'])
    return model

class EpochBatches(Sampler):
    """Batch sampler: a seeded random permutation per epoch, cut into batches."""
    def __init__(self, length, batch_size, seed, epoch, start=0):
        indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        self.batches = [indices[i:i + batch_size] for i in range(0, length, batch_size)]
        self.start = start
    def __iter__(self):
        return iter(self.batches[self.start:])
    def __len__(self):
        return len(self.batches) - self.start

def train_loader(dataset, cfg, epoch, batch_start):
    """Training DataLoader for one epoch, starting at batch_start (non-zero when resuming)."""
    dataset.epoch = epoch
    batches = EpochBatches(len(dataset), cfg['bs'], cfg['seed'], epoch, batch_start)
    return DataLoader(dataset, batch_sampler=batches, num_workers=NUM_WORKERS, pin_memory=USE_AMP, collate_fn=collate,
                      persistent_workers=False, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))

def make_loader(dataset, batch_size):
    """Sequential (non-shuffled) DataLoader used for evaluation."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_AMP,
                      collate_fn=collate, persistent_workers=False)

def make_gt(coco, ids):
    """Build a pycocotools ground-truth object restricted to the given image ids."""
    ids = set(map(int, ids))
    annotations = []
    for annotation in coco['annotations']:
        if int(annotation['image_id']) not in ids:
            continue
        x, y, w, h = map(float, annotation['bbox'])
        if w <= 0 or h <= 0:
            continue
        item = dict(annotation)
        item['area'] = float(annotation.get('area', w*h))
        item['iscrowd'] = int(annotation.get('iscrowd', 0))
        annotations.append(item)
    for j, annotation in enumerate(annotations):
        annotation['id'] = j + 1
    gt = COCO()
    gt.dataset = dict(info={}, licenses=[], images=[image for image in coco['images'] if int(image['id']) in ids],
                      annotations=annotations, categories=coco['categories'])
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt

@torch.inference_mode()
def predict(model, loader):
    """Run the model over a loader; returns a list of (image_id, prediction dict on CPU)."""
    model.eval()
    rows = []
    for images, targets in loader:
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            predictions = model([image.to(DEVICE, non_blocking=True) for image in images])
        for prediction, target in zip(predictions, targets):
            rows.append((int(target['image_id']), {key: value.detach().cpu() for key, value in prediction.items()}))
    return rows

def coco_metrics(predictions, gt, label_to_cat):
    """COCO bbox evaluation; returns mAP, AP50, AP75, APs, APm, APl and AR (at 100 detections)."""
    detections = []
    for image_id, prediction in predictions:
        for box, score, label in zip(prediction['boxes'].tolist(), prediction['scores'].tolist(), prediction['labels'].tolist()):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(dict(image_id=image_id, category_id=label_to_cat[int(label)], bbox=[x1, y1, x2-x1, y2-y1],
                                   score=float(score)))
    with redirect_stdout(io.StringIO()):
        evaluation = COCOeval(gt, gt.loadRes(detections), 'bbox')
        evaluation.params.imgIds = sorted(gt.getImgIds())
        evaluation.params.maxDets = [1, 10, MAX_DETS]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    precision = evaluation.eval['precision'][:, :, :, 0, -1]
    stats = evaluation.stats
    return dict(mAP=float(precision[precision > -1].mean()), AP50=float(stats[1]), AP75=float(stats[2]), APs=float(stats[3]),
                APm=float(stats[4]), APl=float(stats[5]), AR=float(stats[8]))

def save_atomic(obj, path, keep_backup=True):
    """Write a json/torch file through a temporary file so that an interruption never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, tmp = tempfile.mkstemp(prefix='.tmp_', suffix=path.suffix, dir=path.parent)
    os.close(file_descriptor)
    try:
        if path.suffix == '.json':
            with open(tmp, 'w') as file:
                json.dump(obj, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
        else:
            torch.save(obj, tmp)
        if keep_backup and path.exists():
            backup_tmp = str(path) + '.bak.tmp'
            shutil.copy2(path, backup_tmp)
            os.replace(backup_tmp, str(path) + '.bak')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def cfg_id(cfg):
    """Hash identifying a configuration (ignores the checkpoint frequency)."""
    stable = {key: value for key, value in cfg.items() if key != 'save_every'}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()

def lr_schedule(cfg, total_steps):
    """Return the LR multiplier function: linear warm-up, then cosine or multistep decay."""
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
    """Load last.pth (or its backup) if it belongs to this configuration; otherwise return None."""
    for path in [folder / 'last.pth', folder / 'last.pth.bak']:
        if not path.exists():
            continue
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('cfg_id') != identity:
                raise ValueError(f'Checkpoint config mismatch: {path}')
            if path.suffix == '.bak':
                print('Recovered from backup:', path)
            return checkpoint
        except (EOFError, RuntimeError, OSError, ValueError, KeyError) as error:
            print('Cannot use checkpoint:', path, '|', error)
    return None

@contextmanager
def at_scale(model, scale):
    """Temporarily set the model's evaluation resolution."""
    original_sizes = model.transform.min_size, model.transform.max_size
    model.transform.min_size, model.transform.max_size = (round(REF_MIN * scale),), round(REF_MAX * scale)
    try:
        yield
    finally:
        model.transform.min_size, model.transform.max_size = original_sizes

class Data:
    """Annotations, fold split and class mapping for one cross-validation fold."""
    def __init__(self, fold):
        with open(DATA_DIR / 'train/annotations_coco.json') as file:
            self.coco = json.load(file)
        self.folds = pd.read_csv(FOLDS_CSV)
        self.folds_sha = hashlib.sha256(FOLDS_CSV.read_bytes()).hexdigest()[:12]
        cat_ids = sorted(int(category['id']) for category in self.coco['categories'])
        self.label_to_cat = {i + 1: category_id for i, category_id in enumerate(cat_ids)}
        self.n_classes = len(cat_ids) + 1
        self.fold = fold
        self.train_ids = self.folds.loc[self.folds.Fold != fold, 'id'].astype(int).tolist()
        self.val_ids = self.folds.loc[self.folds.Fold == fold, 'id'].astype(int).tolist()

def make_config(fold, data):
    """Fine-tuning config = the fold's original config + OVERRIDES + the init checkpoint and its hash."""
    original = json.loads((BASE_RUNS / f'fold{fold}/config.json').read_text())
    init = BASE_RUNS / f'fold{fold}/best.pth'
    cfg = {**original, **OVERRIDES, 'init_checkpoint': init.relative_to(ROOT).as_posix(), 'init_sha256': sha256(init)}
    assert cfg['fold'] == fold and cfg['folds'] == data.folds_sha
    return cfg, original

def load_init(cfg, original, data):
    """Build the model and load the fold's original best.pth; returns (model on device, checkpoint)."""
    checkpoint = torch.load(ROOT / cfg['init_checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['cfg_id'] == cfg_id(original), 'best.pth does not belong to the original configuration'
    model = build_model(cfg, data.n_classes)
    missing = model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(DEVICE), checkpoint

def evaluate(model, data, scale, batch_size, ids=None):
    """COCO metrics of the model on the held-out fold (or the given ids) at the given evaluation scale."""
    ids = ids or data.val_ids
    dataset = VisDrone(DATA_DIR, 'train', data.coco, ids)
    with at_scale(model, scale):
        predictions = predict(model, make_loader(dataset, batch_size))
    return coco_metrics(predictions, make_gt(data.coco, ids), data.label_to_cat)

def train(cfg, data, folder, train_ids=None, eval_ids=None, stop_after=None, crash=False, trace=None, init_model=None):
    """run_experiment of faster_rcnn.ipynb, starting from the fold's best.pth. Returns 'done' or 'interrupted'.
    stop_after / crash / trace / train_ids / eval_ids exist only for the correctness checks."""
    folder.mkdir(parents=True, exist_ok=True)
    identity = cfg_id(cfg)
    if (folder / 'config.json').exists():
        assert cfg_id(json.loads((folder / 'config.json').read_text())) == identity, \
            f'{folder}: a different configuration exists'
    else:
        assert not any(folder.iterdir()), f'{folder} has files but no config: refusing to overwrite'
        save_atomic(cfg, folder / 'config.json', keep_backup=False)
    if (folder / 'result.json').exists():
        log(f'{folder.name}: result already exists, skipping training')
        return 'done'
    seed_all(cfg['seed'])
    model = init_model if init_model is not None else load_init(
        cfg, json.loads((BASE_RUNS / f"fold{cfg['fold']}/config.json").read_text()), data)[0]
    dataset = VisDrone(DATA_DIR, 'train', data.coco, train_ids or data.train_ids, train=True, seed=cfg['seed'])
    batches_per_epoch = math.ceil(len(dataset) / cfg['bs'])
    total_steps = batches_per_epoch * cfg['epochs']
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=.9, weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule(cfg, total_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    history, resume_epoch, resume_batch, resumed_loss_sums, resumed_count = [], 0, 0, {}, 0
    saved_checkpoint = read_checkpoint(folder, identity)
    if saved_checkpoint is not None:
        model.load_state_dict(saved_checkpoint['model'])
        optimizer.load_state_dict(saved_checkpoint['optimizer'])
        scheduler.load_state_dict(saved_checkpoint['scheduler'])
        scaler.load_state_dict(saved_checkpoint['scaler'])
        history = saved_checkpoint['history']
        resume_epoch, resume_batch = saved_checkpoint['epoch'], saved_checkpoint['batch_start']
        resumed_loss_sums, resumed_count = saved_checkpoint['sums'], saved_checkpoint['seen']
        torch.set_rng_state(saved_checkpoint['torch_rng'])
        np.random.set_state(saved_checkpoint['numpy_rng'])
        random.setstate(saved_checkpoint['python_rng'])
        if DEVICE.type == 'cuda' and saved_checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved_checkpoint['cuda_rng'])
        log(f"{folder.name}: resume at epoch {resume_epoch+1}, batch {resume_batch+1}/{batches_per_epoch}")
    else:
        log(f"{folder.name}: start from {cfg['init_checkpoint']} | {len(dataset)} training frames, "
            f"{batches_per_epoch} batches per epoch, {total_steps} steps")
    if trace is not None:
        trace['resume_state'] = dict(
            scheduler_last_epoch=scheduler.last_epoch, scaler_scale=scaler.get_scale(),
            epoch=resume_epoch, batch_start=resume_batch,
            momentum_buffers=sum(1 for param_state in optimizer.state.values() if param_state.get('momentum_buffer') is not None),
            trainable_tensors=len(params))
        logged_image_sizes = []
        hook = model.transform.register_forward_hook(
            lambda module, inputs, output: logged_image_sizes.append([tuple(size) for size in output[0].image_sizes]))

    def save_training_state(epoch_index, batch_index, loss_sums, batches_counted):
        state = dict(cfg_id=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     history=history, epoch=epoch_index, batch_start=batch_index, sums=dict(loss_sums), seen=batches_counted,
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None)
        save_atomic(state, folder / 'last.pth')

    steps_done_now, start_time = 0, time.time()
    for epoch_index in range(resume_epoch, cfg['epochs']):
        start_batch = resume_batch if epoch_index == resume_epoch else 0
        loss_sums = defaultdict(float, resumed_loss_sums if epoch_index == resume_epoch else {})
        batches_counted = resumed_count if epoch_index == resume_epoch else 0
        model.train()
        loader = train_loader(dataset, cfg, epoch_index, start_batch)
        for batch_index, (images, targets) in enumerate(loader, start=start_batch):
            image_ids = [int(target['image_id']) for target in targets]
            images = [image.to(DEVICE, non_blocking=True) for image in images]
            targets = [{'boxes': target['boxes'].to(DEVICE), 'labels': target['labels'].to(DEVICE),
                        'ignored': target['ignored'].to(DEVICE)} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            lr_now = optimizer.param_groups[0]['lr']
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                save_training_state(epoch_index, batch_index, loss_sums, batches_counted)
                raise RuntimeError(f'Non-finite loss in epoch {epoch_index+1} batch {batch_index+1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 10.)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not USE_AMP or scaler.get_scale() >= scale_before_step:
                scheduler.step()
            for loss_name, loss_value in losses.items():
                loss_sums[loss_name] += float(loss_value.detach())
            batches_counted += 1
            steps_done_now += 1
            if trace is not None:
                trace.setdefault('steps', []).append(dict(epoch=epoch_index, batch=batch_index, ids=image_ids, lr=lr_now,
                                                          loss=float(loss.detach()), sizes=logged_image_sizes[-1]))
            if (batch_index + 1) % cfg['save_every'] == 0:
                save_training_state(epoch_index, batch_index + 1, loss_sums, batches_counted)
            if DEVICE.type == 'cuda' and (batch_index + 1) % 10 == 0:
                torch.cuda.empty_cache()
            if (batch_index + 1) % 50 == 0:
                iterations_per_second = steps_done_now / (time.time() - start_time)
                batches_left = (batches_per_epoch - batch_index - 1) + (cfg['epochs'] - epoch_index - 1) * batches_per_epoch
                log(f"epoch {epoch_index+1}/{cfg['epochs']} batch {batch_index+1}/{batches_per_epoch} | "
                    f"loss {sum(loss_sums.values())/batches_counted:.4f} | lr {lr_now:.6f} | "
                    f"{iterations_per_second:.2f} it/s | train left ~{batches_left / iterations_per_second / 60:.0f} min | "
                    f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
                    f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB")
            if stop_after is not None and steps_done_now >= stop_after:
                if not crash:
                    save_training_state(epoch_index, batch_index + 1, loss_sums, batches_counted)
                return 'interrupted'
        save_training_state(epoch_index, batches_per_epoch, loss_sums, batches_counted)
        metrics = {f'{metric}@{round(REF_MIN * eval_scale)}': value for eval_scale, eval_batch_size in EVAL_SCALES.items()
                   for metric, value in evaluate(model, data, eval_scale, eval_batch_size, eval_ids).items()}
        history.append(dict(epoch=epoch_index+1, **{name: total/max(batches_counted, 1) for name, total in loss_sums.items()},
                            loss=sum(loss_sums.values())/max(batches_counted, 1),
                            lr=optimizer.param_groups[0]['lr'], **metrics))
        save_atomic(dict(cfg_id=identity, model=model.state_dict(), epoch=epoch_index+1, metrics=metrics),
                    folder / f'epoch{epoch_index+1}.pth', keep_backup=False)
        log(f"epoch {epoch_index+1} done: loss {history[-1]['loss']:.4f} | held-out mAP@720 {metrics['mAP@720']:.4f} "
            f"APs {metrics['APs@720']:.4f} | "
            f"mAP@1080 {metrics['mAP@1080']:.4f} APs {metrics['APs@1080']:.4f} APl {metrics['APl@1080']:.4f}")
        save_training_state(epoch_index + 1, 0, {}, 0)
        resume_batch, resumed_loss_sums, resumed_count = 0, {}, 0
    shutil.copy2(folder / f"epoch{cfg['epochs']}.pth", folder / 'final.pth')
    save_atomic(dict(name=cfg['name'], fold=cfg['fold'], cfg=cfg, history=history, selected_checkpoint='final.pth',
                     selection=f"last epoch ({cfg['epochs']}), fixed before training; not selected on held-out data",
                     final_sha256=sha256(folder / 'final.pth'), train_frames=len(dataset),
                     versions=dict(torch=torch.__version__, torchvision=torchvision.__version__),
                     device=torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'cpu'),
                folder / 'result.json', keep_backup=False)
    if trace is not None:
        hook.remove()
    return 'done'

def run_checks(fold, data, cfg, original, folder):
    """Run all correctness checks, save them to <name>_checks.json and return True if every check passed."""
    results = {}
    def check(name, passed, detail=''):
        results[name] = dict(passed=bool(passed), detail=detail)
        log(f"{'PASS' if passed else 'FAIL'}: {name}" + (f' | {detail}' if detail else ''))

    recorded = json.loads((ROOT / 'faster_rcnn/inference_experiments/tta/tta_config.json').read_text())['checkpoints']
    shas = {fold_index: sha256(BASE_RUNS / f'fold{fold_index}/best.pth') for fold_index in range(5)}
    check('old checkpoints equal the hashes recorded on Nov 8 (before)',
          all(shas[fold_index] == recorded[str(fold_index)]['sha256'] for fold_index in range(5)))
    tree = lambda: sorted((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                          for path in (ROOT / 'faster_rcnn').rglob('*') if path.is_file())
    tree_before = tree()

    folds_table = data.folds
    train_videos, held_out_videos = (set(folds_table.loc[folds_table.Fold != fold, 'video']),
                                     set(folds_table.loc[folds_table.Fold == fold, 'video']))
    val_official = {frame['video'] for frame in json.loads((DATA_DIR / 'val/annotations_coco.json').read_text())['images']}
    dataset = VisDrone(DATA_DIR, 'train', data.coco, data.train_ids, train=True, seed=cfg['seed'])
    dataset_ids = {int(frame['id']) for frame in dataset.images}
    check('training uses only fold-0 training sequences',
          dataset_ids == set(data.train_ids) and not dataset_ids & set(data.val_ids)
          and not train_videos & held_out_videos and not train_videos & val_official and dataset.split == 'train',
          f'{len(dataset_ids)} frames from {len(train_videos)} videos; held-out {len(data.val_ids)} frames '
          f'from {len(held_out_videos)} videos; shared videos 0')

    model, original_checkpoint = load_init(cfg, original, data)
    stored = json.loads((BASE_RUNS / f'fold{fold}/result.json').read_text())['metrics']
    baseline_metrics = evaluate(model, data, 1.0, 4)
    max_abs_diff = max(abs(baseline_metrics[key] - stored[key]) for key in KEYS)
    check('best.pth reproduces its recorded held-out metrics',
          max_abs_diff < 1e-9 and all(abs(baseline_metrics[key] - original_checkpoint['metrics'][key]) < 1e-9 for key in KEYS),
          f"mAP {baseline_metrics['mAP']:.6f} (recorded {stored['mAP']:.6f}), max abs diff over 7 metrics {max_abs_diff:.1e}")
    check('transform is back to the training sizes after evaluation',
          model.transform.min_size == tuple(cfg['min_size']) and model.transform.max_size == cfg['max_size'])

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    groups = sorted({'.'.join(name.split('.')[:3]) if name.startswith('backbone.body')
                     else name.split('.')[0] + '.' + name.split('.')[1] for name in trainable})
    expected_frozen = all(name.startswith(('backbone.body.conv1', 'backbone.body.layer1')) for name in frozen)
    expected_trainable = all(name.startswith(('backbone.body.layer2', 'backbone.body.layer3', 'backbone.body.layer4',
                                              'backbone.fpn', 'rpn', 'roi_heads')) for name in trainable)
    batch_norm_layers = [module for module in model.modules() if isinstance(module, (torch.nn.BatchNorm2d, FrozenBatchNorm2d))]
    check('trainable = layer2-4, FPN, RPN, RoI heads; frozen = stem, layer1; all BatchNorm frozen (buffers only)',
          expected_frozen and expected_trainable and all(isinstance(module, FrozenBatchNorm2d) for module in batch_norm_layers)
          and all(len(list(module.parameters())) == 0 for module in batch_norm_layers),
          f'{len(trainable)} trainable tensors, {len(frozen)} frozen tensors, '
          f'{len(batch_norm_layers)} FrozenBatchNorm2d layers | {groups}')

    model.train()
    by_size = {}
    for i, frame in enumerate(dataset.images):
        if frame['id'] in dataset.ign and len(dataset.objs[frame['id']]):
            by_size.setdefault((frame['width'], frame['height']), []).append(i)
    picks = [i for indices in by_size.values() for i in indices[:2]]
    size_counts, dims_ok, boxes_ok, ignore_ok, frames_resized = defaultdict(int), True, True, True, 0
    torch.manual_seed(0)
    with torch.no_grad():
        for repetition in range(60):
            for first in range(0, len(picks) - 1, 2):
                samples = [dataset[picks[first]], dataset[picks[first + 1]]]
                images = [sample[0].to(DEVICE) for sample in samples]
                targets = [{key: value.clone().to(DEVICE) for key, value in sample[1].items()
                            if key in ('boxes', 'labels', 'ignored')} for sample in samples]
                transformed_batch, transformed_targets = model.transform(
                    images, [{key: value.clone() for key, value in target.items()} for target in targets])
                for frame, target, transformed_target, (new_height, new_width) in zip(
                        images, targets, transformed_targets, transformed_batch.image_sizes):
                    h, w = frame.shape[-2:]
                    size = min(cfg['min_size'], key=lambda candidate: abs(candidate - new_height))
                    size_counts[size] += 1
                    frames_resized += 1
                    scale = min(size / min(h, w), cfg['max_size'] / max(h, w))
                    dims_ok &= (abs(new_height - h * scale) <= 1 and abs(new_width - w * scale) <= 1
                                and max(new_height, new_width) <= cfg['max_size'])
                    ratio = torch.tensor([new_width / w, new_height / h, new_width / w, new_height / h], device=DEVICE)
                    boxes_ok &= torch.allclose(transformed_target['boxes'], target['boxes'] * ratio, atol=1e-2)
                    ignore_ok &= (torch.allclose(transformed_target['ignored'], target['ignored'] * ratio, atol=1e-2)
                                  and len(target['ignored']) > 0)
        flat = model.transform([torch.tensor(model.transform.image_mean, device=DEVICE).view(3, 1, 1).expand(3, 756, 1344).clone()
                                ])[0].tensors
    share = {size: round(size_counts[size] / frames_resized, 3) for size in cfg['min_size']}
    check('all five training scales are sampled',
          all(size_counts[size] > 0 for size in cfg['min_size']) and min(share.values()) > 0.1,
          f'{frames_resized} resized frames, share per short side {share}')
    check('effective image sizes follow min(size / short, max_size / long), long side <= 1920', dims_ok,
          f'frame sizes tested: {sorted(by_size)}')
    check('boxes and ignore regions are scaled with the image', boxes_ok and ignore_ok)
    check('normalisation is applied once (ImageNet mean/std on [0, 1] input)',
          model.transform.image_mean == [0.485, 0.456, 0.406]
          and model.transform.image_std == [0.229, 0.224, 0.225] and float(flat.abs().max()) < 1e-4
          and float(dataset[picks[0]][0].max()) <= 1.0)

    before = {name: tensor.detach().clone() for name, tensor in list(model.named_parameters()) + list(model.named_buffers())}
    stress_cfg = {**cfg, 'min_size': [max(cfg['min_size'])], 'save_every': 10**9}
    model.transform.min_size = (max(cfg['min_size']),)
    stress_dir = Path(tempfile.mkdtemp(prefix='hires_stress_'))
    torch.cuda.reset_peak_memory_stats()
    trace, start_time = {}, time.time()
    try:
        status, error = train(stress_cfg, data, stress_dir, stop_after=30, crash=True, trace=trace, init_model=model), ''
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exception:
        status, error = 'failed', str(exception)[:200]
    seconds = time.time() - start_time
    steps = trace.get('steps', [])
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    total_steps = math.ceil(len(data.train_ids) / cfg['bs']) * cfg['epochs']
    expected_lr = [cfg['lr'] * lr_schedule(cfg, total_steps)(i) for i in range(len(steps))]
    check('stress test: 30 iterations at 1080 x 1920, batch 2, no OOM, finite losses',
          status == 'interrupted' and len(steps) == 30
          and all(math.isfinite(step['loss']) for step in steps)
          and all(max(image_size) <= 1920 and min(image_size) >= 1070 for step in steps for image_size in step['sizes']),
          f"{error or ''}peak GPU memory {peak_gb:.2f} GB allocated / "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved, "
          f"{len(steps) / seconds:.2f} it/s incl. start-up, "
          f"loss {steps[0]['loss']:.3f} -> {np.mean([step['loss'] for step in steps[-10:]]):.3f}" if steps else error)
    check('optimizer: SGD(momentum 0.9, wd 1e-4) on the trainable tensors only; warmup LR stepped once per iteration',
          bool(steps) and np.allclose([step['lr'] for step in steps], expected_lr, rtol=1e-9)
          and abs(steps[0]['lr'] - cfg['lr'] / cfg['warmup']) < 1e-12
          and trace['resume_state']['trainable_tensors'] == len(trainable),
          f"lr step 1 {steps[0]['lr']:.3e}, step 30 {steps[-1]['lr']:.3e} (expected {expected_lr[-1]:.3e})" if steps else '')
    after = dict(list(model.named_parameters()) + list(model.named_buffers()))
    changed = {name for name in before if not torch.equal(before[name], after[name].detach())}
    check('only the intended parameters changed: frozen layers and all BatchNorm buffers are bit-identical',
          changed == set(trainable) and all(torch.isfinite(after[name]).all() for name in changed),
          f'{len(changed)} tensors changed = {len(trainable)} trainable; '
          f'{len(before) - len(changed)} frozen tensors/buffers unchanged')
    shutil.rmtree(stress_dir, ignore_errors=True)
    del model, before, after
    torch.cuda.empty_cache()

    small_cfg = {**cfg, 'epochs': 1, 'save_every': 4, 'warmup': 6}
    sub_train, sub_val = data.train_ids[:24], data.val_ids[:8]
    uninterrupted_dir, resumed_dir, repeat_dir = (Path(tempfile.mkdtemp(prefix=f'hires_resume_{letter}_')) for letter in 'abc')
    uninterrupted_trace, repeat_trace, crashed_trace, resumed_trace = {}, {}, {}, {}
    train(small_cfg, data, uninterrupted_dir, train_ids=sub_train, eval_ids=sub_val, trace=uninterrupted_trace)
    train(small_cfg, data, repeat_dir, train_ids=sub_train, eval_ids=sub_val, trace=repeat_trace)
    train(small_cfg, data, resumed_dir, train_ids=sub_train, eval_ids=sub_val, stop_after=6, crash=True, trace=crashed_trace)
    saved = torch.load(resumed_dir / 'last.pth', map_location='cpu', weights_only=False)
    train(small_cfg, data, resumed_dir, train_ids=sub_train, eval_ids=sub_val, trace=resumed_trace)
    first, second = uninterrupted_trace['steps'][4:], resumed_trace['steps']
    resume_state = resumed_trace['resume_state']
    state_ok = (resume_state['batch_start'] == 4 and resume_state['epoch'] == 0 and resume_state['scheduler_last_epoch'] == 4
                and resume_state['momentum_buffers'] == resume_state['trainable_tensors']
                and saved['scaler']['scale'] == resume_state['scaler_scale']
                and all(key in saved for key in ['optimizer', 'scheduler', 'scaler', 'torch_rng', 'cuda_rng', 'history', 'sums']))
    same_order = len(first) == len(second) == 8 and all(
        step_a['ids'] == step_b['ids'] and step_a['batch'] == step_b['batch']
        for step_a, step_b in zip(first, second))
    same_lr = np.allclose([step['lr'] for step in first], [step['lr'] for step in second], rtol=1e-12)
    same_sizes = all(step_a['sizes'] == step_b['sizes'] for step_a, step_b in zip(first, second))
    loss_diff = max(abs(step_a['loss'] - step_b['loss']) / step_a['loss']
                    for step_a, step_b in zip(first, second))
    loss_noise = max(abs(step_a['loss'] - step_b['loss']) / step_a['loss']
                     for step_a, step_b in zip(first, repeat_trace['steps'][4:]))
    weights = lambda run_dir: torch.load(run_dir / 'final.pth', map_location='cpu', weights_only=False)['model']
    uninterrupted_weights, resumed_weights, repeat_weights = weights(uninterrupted_dir), weights(resumed_dir), weights(repeat_dir)
    weight_diff = max(float((uninterrupted_weights[key].float() - resumed_weights[key].float()).abs().max())
                      for key in uninterrupted_weights)
    weight_noise = max(float((uninterrupted_weights[key].float() - repeat_weights[key].float()).abs().max())
                       for key in uninterrupted_weights)
    check('resume restores optimizer momentum, scheduler step, AMP scaler, RNG and batch position',
          state_ok and same_order and same_lr and same_sizes and weight_diff <= 2 * weight_noise,
          f"resumed at batch 5 with {resume_state['momentum_buffers']} momentum buffers, "
          f"scheduler step {resume_state['scheduler_last_epoch']}; steps 5-12: same frames, "
          f"same LR, same sampled scales; max weight difference vs uninterrupted run {weight_diff:.1e} "
          f"(two uninterrupted runs: {weight_noise:.1e}); "
          f"max relative loss difference {loss_diff:.1e} (two uninterrupted runs: {loss_noise:.1e})")
    for directory in (uninterrupted_dir, resumed_dir, repeat_dir):
        shutil.rmtree(directory, ignore_errors=True)

    check('old checkpoints and the original results folder are unchanged (after)',
          all(sha256(BASE_RUNS / f'fold{fold_index}/best.pth') == shas[fold_index] for fold_index in range(5))
          and tree() == tree_before, f'{len(tree_before)} files compared (size and mtime)')
    save_atomic(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), cfg=cfg,
                     all_passed=all(result['passed'] for result in results.values()), checks=results),
                folder.parent / f'{folder.name}_checks.json', keep_backup=False)
    return all(result['passed'] for result in results.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='run the correctness checks only')
    parser.add_argument('--skip-checks', action='store_true', help='continue an interrupted run whose checks already passed')
    args = parser.parse_args()
    data = Data(args.fold)
    cfg, original = make_config(args.fold, data)
    folder = OUTPUT_DIR / cfg['name'] / f'fold{args.fold}'
    assert OUTPUT_DIR.resolve() not in BASE_RUNS.resolve().parents and BASE_RUNS.resolve().parent not in folder.resolve().parents
    log(f"""config: {json.dumps({key: cfg[key] for key in [
        'name', 'fold', 'epochs', 'bs', 'lr', 'warmup', 'scheduler', 'min_size', 'max_size', 'trainable_layers',
        'init_checkpoint']})}""")
    log(f"device {torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'cpu'} | torch {torch.__version__} | "
        f"torchvision {torchvision.__version__}")
    if args.skip_checks:
        done = json.loads((folder.parent / f'{folder.name}_checks.json').read_text())
        assert done['all_passed'] and cfg_id(done['cfg']) == cfg_id(cfg), 'checks have not passed for this configuration'
    else:
        if not run_checks(args.fold, data, cfg, original, folder):
            log('A correctness check FAILED: training is not started.')
            sys.exit(1)
        log('All correctness checks passed.')
    if args.check:
        sys.exit(0)
    if DEVICE.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.8)
    status = train(cfg, data, folder)
    log(f'training {status}: {folder}')
