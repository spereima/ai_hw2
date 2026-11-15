"""
Final evaluation and testing of Faster R-CNN and FCOS (inference only).
"""
import datetime
import gc
import hashlib
import io
import json
import math
import os
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
DATA = ROOT / 'datasets' / 'multiple-object-video-detection-hw-2-ucu-ai-2026'
FINAL = ROOT / 'model_comparison'
FOLDS_CSV = ROOT / 'shared/sequence_folds.csv'
FCOS_SOURCE = ROOT / 'fcos/fcos_backup_new_account'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0

MODELS = {'faster_rcnn': dict(label='Faster R-CNN', results=ROOT / 'faster_rcnn', fold_dir='fold{}',
    experiments=['baseline', 'giou_loss', 'lr_scheduler'], eval_batch=4, notebook='faster_rcnn.ipynb'),
    'fcos': dict(label='FCOS', results=ROOT / 'fcos', fold_dir='fold_{}', experiments=['F0_baseline', 
    'F1_ciou', 'F2_radius_1_0'], eval_batch=2, notebook='fcos.ipynb'),}
COMPARED_MODELS = { **MODELS, 'rt_detr': dict(label='RT-DETR', results=ROOT / 'rt_detr/runs', fold_dir='fold{}',
    experiments=['recipe_warmup_ema', 'no_denoising', 'no_denoising_640x1152'], notebook='rtdetr.ipynb'),}

FOLDS = list(range(5))
CLASSES = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']
METRICS = ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']
COLUMNS = dict(mAP='map', AP50='map_50', AP75='map_75', APs='map_small', APm='map_medium', APl='map_large', AR='ar')
COLORS = {'baseline': '#9ecae1', 'giou_loss': '#4292c6', 'lr_scheduler': '#08519c',
          'F0_baseline': '#fdae6b', 'F1_ciou': '#f16913', 'F2_radius_1_0': '#a63603',
          'recipe_warmup_ema': '#a1d99b', 'no_denoising': '#41ab5d', 'no_denoising_640x1152': '#006d2c'}
MODEL_COLORS = {'faster_rcnn': '#2171b5', 'fcos': '#d94801', 'rt_detr': '#238b45'}

EVAL_CONFIG = dict(
    split='official validation (val/), all images', annotations='val/annotations_coco.json',
    ground_truth='all annotations; iscrowd=1 ignore regions (stored once per class) are kept, so COCOeval '
                 'ignores detections matched to them; area = bbox w*h (equals the stored area)',
    iouThrs='0.50:0.05:0.95', recThrs='0:0.01:1', maxDets=[1, 10, 100],
    areaRng={'small': [0, 32 ** 2], 'medium': [32 ** 2, 96 ** 2], 'large': [96 ** 2, 1e10]},
    mAP='stats[0]: AP@[.50:.95], all areas, maxDets=100', AR='stats[8]: AR@100, all areas',
    predictions='native postprocessing of each checkpoint config; boxes in original pixels (xywh for COCO); '
                'boxes with zero width/height and label 0 (unused FCOS background slot) are dropped',
    inference='torch.autocast fp16 on CUDA; float32 RGB in [0,1]; resize + ImageNet normalisation inside the '
              'torchvision model; batch 4 (Faster R-CNN) / 2 (FCOS) as in the training notebooks')
EVAL_ID = hashlib.sha256(json.dumps(EVAL_CONFIG, sort_keys=True).encode()).hexdigest()[:16]

SPEED_CONFIG = dict(gpu='NVIDIA GeForce RTX 5070 (local)', images='200 official-validation frames, 40 per resolution, '
                    'seed 0', preload='decoded once, kept on the GPU as uint8 (no disk I/O, no data loading in timing)',
                    timed='uint8->float conversion + model forward (resize, normalise, backbone, heads) + native '
                          'postprocessing (decode, NMS)', sync='torch.cuda.synchronize() before and after every batch',
                    warmup_batches=20, batch_sizes=[1, 4], precision='autocast fp16 (as in evaluation); fp32 for '
                    'the representative checkpoints', batches='same-resolution images only')
SPEED_ID = hashlib.sha256(json.dumps(SPEED_CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def _atomic(path, data):
    """Write bytes or text to `path` through a temporary file, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    tmp_path.write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))
    os.replace(tmp_path, path)


def save_new(path, data):
    """Write `data` to `path` unless that would overwrite different content.

    Same content: nothing is written. Different content: a timestamped sibling is written instead."""
    path = Path(path)
    data = data if isinstance(data, bytes) else data.encode('utf-8')
    if path.exists():
        if path.read_bytes() == data:
            return path
        path = path.with_name(f'{path.stem}.{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}')
        print('Kept the existing file; wrote the changed version to', path.name)
    _atomic(path, data)
    return path


def save_csv(table, path):
    """Save a DataFrame with save_new (never overwrites different content)."""
    return save_new(path, table.to_csv(index=False, lineterminator='\n'))


def save_json(obj, path):
    """Save a json-serialisable object with save_new."""
    return save_new(path, json.dumps(obj, indent=2, default=_json_default))


def save_html(fig, path):
    """Save a Plotly figure as an offline html file with save_new."""
    from plotly.offline import get_plotlyjs
    plotly_js_path = Path(path).parent / 'plotly.min.js'
    if not plotly_js_path.exists():
        _atomic(plotly_js_path, get_plotlyjs())
    return save_new(path, fig.to_html(include_plotlyjs='directory', full_html=True, div_id=Path(path).stem))


def _json_default(obj):
    """json fallback for numpy values and Path objects."""
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def check_memory(min_free_gb=4.0):
    """Stop before starting a checkpoint if the machine is low on RAM (finished results stay saved)."""
    if os.name != 'nt':
        return
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = ([('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)]
                    + [(field_name, ctypes.c_ulonglong)
                       for field_name in ('total', 'avail', 'total_page', 'avail_page', 'total_virt',
                                          'avail_virt', 'avail_ext')])
    memory_status = Status()
    memory_status.dwLength = ctypes.sizeof(memory_status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
    if memory_status.avail / 2 ** 30 < min_free_gb or memory_status.avail_page / 2 ** 30 < min_free_gb:
        raise MemoryError(
            f'only {memory_status.avail / 2 ** 30:.1f} GB RAM / {memory_status.avail_page / 2 ** 30:.1f} GB commit free: '
            'stopped before the next checkpoint; rerun to resume')


def sha256(path, chunk=1 << 22):
    """SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(chunk), b''):
            hasher.update(block)
    return hasher.hexdigest()


_cache = {}


def coco_json(split):
    """Annotations of a split ('train' or 'val') as a dict; loaded once and cached."""
    if split not in _cache:
        data = json.loads((DATA / split / 'annotations_coco.json').read_text(encoding='utf-8'))
        assert [category['id'] for category in data['categories']] == list(range(1, 9)), 'unexpected category ids'
        assert [category['name'] for category in data['categories']] == CLASSES, 'unexpected class names'
        _cache[split] = data
    return _cache[split]


def make_gt(coco, ids=None):
    """COCO ground truth for a subset; identical rules in both training notebooks for this data."""
    ids = {int(image_info['id']) for image_info in coco['images']} if ids is None else {int(image_id) for image_id in ids}
    annotations = []
    for annotation in coco['annotations']:
        if int(annotation['image_id']) in ids and annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0:
            annotation = dict(annotation)
            annotation['area'] = float(annotation['bbox'][2] * annotation['bbox'][3])
            annotation['iscrowd'] = int(annotation.get('iscrowd', 0))
            annotations.append(annotation)
    for index, annotation in enumerate(annotations):
        annotation['id'] = index + 1
    gt = COCO()
    gt.dataset = dict(
        info={}, licenses=[], images=[image_info for image_info in coco['images'] if int(image_info['id']) in ids],
        annotations=annotations, categories=coco['categories'])
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def official_gt():
    """COCO ground truth of the whole official validation set (cached)."""
    if 'official_gt' not in _cache:
        _cache['official_gt'] = make_gt(coco_json('val'))
    return _cache['official_gt']


def load_rgb(split, info):
    """Decoded RGB uint8 array (H, W, 3); safe to call from worker threads."""
    with Image.open(DATA / split / info['file_name']) as image:
        return np.array(image.convert('RGB'))


def to_tensor(rgb):
    """RGB float32 tensor in [0, 1], as both training datasets produce it."""
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)


def load_image(split, info):
    """Image of one COCO image entry as a float tensor in [0, 1]."""
    return to_tensor(load_rgb(split, info))


def image_targets(split, image_id):
    """Non-crowd boxes (xyxy, clipped), labels and unique ignore regions of one image."""
    key = ('targets', split)
    if key not in _cache:
        coco = coco_json(split)
        info = {int(image_info['id']): image_info for image_info in coco['images']}
        table = {image_id: dict(boxes=[], labels=[], occlusion=[], ignore=set()) for image_id in info}
        for annotation in coco['annotations']:
            entry = table[int(annotation['image_id'])]
            image_info = info[int(annotation['image_id'])]
            x, y, w, h = annotation['bbox']
            box = (max(0, x), max(0, y), min(image_info['width'], x + w), min(image_info['height'], y + h))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if annotation.get('iscrowd', 0):
                entry['ignore'].add(box)
            else:
                entry['boxes'].append(box)
                entry['labels'].append(int(annotation['category_id']))
                entry['occlusion'].append(int(annotation.get('occlusion', 0) or 0))
        _cache[key] = (info, table)
    info, table = _cache[key]
    entry = table[int(image_id)]
    return dict(info=info[int(image_id)], boxes=np.asarray(entry['boxes'], np.float32).reshape(-1, 4),
                labels=np.asarray(entry['labels'], np.int64), occlusion=np.asarray(entry['occlusion'], np.int64),
                ignore=np.asarray(sorted(entry['ignore']), np.float32).reshape(-1, 4))


def cv_fold_ids(fold):
    """Image ids of one cross-validation fold (the held-out frames of that fold)."""
    folds = pd.read_csv(FOLDS_CSV)
    return folds.loc[folds.Fold == fold, 'id'].astype(int).tolist()


def run_dir(model, experiment, fold):
    """Folder of one training run."""
    return COMPARED_MODELS[model]['results'] / experiment / COMPARED_MODELS[model]['fold_dir'].format(fold)


def load_result(model, experiment, fold):
    """result.json of one training run."""
    return json.loads((run_dir(model, experiment, fold) / 'result.json').read_text(encoding='utf-8'))


def build_model(model, cfg):
    """Same architecture and inference settings as `build_model` in the training notebook.

    The Faster R-CNN notebook additionally patches training-only parts (ignore-aware target assignment,
    GIoU loss); they do not run in eval mode and are therefore not needed to reproduce predictions."""
    if model == 'faster_rcnn':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        net = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=cfg['trainable_layers'],
            min_size=cfg['min_size'], max_size=cfg['max_size'],
            rpn_pre_nms_top_n_train=cfg['rpn_pre_train'], rpn_post_nms_top_n_train=cfg['rpn_post_train'],
            rpn_pre_nms_top_n_test=cfg['rpn_pre_test'], rpn_post_nms_top_n_test=cfg['rpn_post_test'],
            rpn_batch_size_per_image=cfg['rpn_batch'], box_batch_size_per_image=cfg['roi_batch'],
            box_positive_fraction=cfg['roi_pos_frac'], box_detections_per_img=cfg['det_per_img'],
            box_score_thresh=cfg['score_thr'], box_nms_thresh=cfg['nms_thr'])
        net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 9)
        return net
    from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
    net = fcos_resnet50_fpn(
        weights=FCOS_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=cfg['trainable_layers'],
        min_size=cfg['min_size'], max_size=cfg['max_size'], center_sampling_radius=cfg['center_radius'],
        score_thresh=cfg['score_thr'], nms_thresh=cfg['nms_thr'], topk_candidates=cfg['topk'],
        detections_per_img=cfg['det_per_img'])
    head = net.head.classification_head
    head.cls_logits = nn.Conv2d(256, 9, kernel_size=3, padding=1)
    head.num_classes = 9
    return net


def checkpoint_record(model, experiment, fold):
    """Identity of one checkpoint: model, experiment, fold, best epoch, path, sha256 and training config."""
    folder = run_dir(model, experiment, fold)
    result = load_result(model, experiment, fold)
    return dict(model=model, experiment=experiment, fold=fold, best_epoch=int(result['best_epoch']),
                checkpoint=str(folder / 'best.pth'), checkpoint_sha256=sha256(folder / 'best.pth'), cfg=result['cfg'])


def load_checkpoint(model, experiment, fold):
    """Rebuild the network from its saved configuration and load best.pth strictly (eval mode)."""
    folder = run_dir(model, experiment, fold)
    result = load_result(model, experiment, fold)
    checkpoint_data = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
    assert int(checkpoint_data['epoch']) == int(result['best_epoch']), 'best.pth epoch differs from result.json'
    if model == 'faster_rcnn':
        cfg = json.loads((folder / 'config.json').read_text())
        stable = {k: v for k, v in cfg.items() if k != 'save_every'}
        assert checkpoint_data['cfg_id'] == hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest(), \
            'best.pth does not belong to config.json'
    else:
        cfg = result['cfg']
        assert checkpoint_data['cfg'] == cfg, 'best.pth configuration differs from result.json'
    net = build_model(model, cfg)
    net.load_state_dict(checkpoint_data['model'], strict=True)
    return net.to(DEVICE).eval(), cfg


@torch.inference_mode()
def predict(net, split, image_ids, batch_size, amp=True, prefetch=8):
    """Predictions for the given images in their original pixel coordinates.

    JPEG decoding runs in 2 threads (uint8 only); tensors are created in the main thread, because float
    tensors allocated in worker threads made host memory grow by several GB per thousand 4K frames."""
    infos = {int(image_info['id']): image_info for image_info in coco_json(split)['images']}
    ids = [int(image_id) for image_id in image_ids]
    out = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        queue = deque(pool.submit(load_rgb, split, infos[image_id]) for image_id in ids[:prefetch])
        next_index = len(queue)
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            images = []
            for _ in batch:
                images.append(to_tensor(queue.popleft().result()))
                if next_index < len(ids):
                    queue.append(pool.submit(load_rgb, split, infos[ids[next_index]]))
                    next_index += 1
            with torch.autocast(DEVICE.type, enabled=amp and DEVICE.type == 'cuda'):
                predictions_batch = net([image.to(DEVICE, non_blocking=True) for image in images])
            for image_id, prediction in zip(batch, predictions_batch):
                out[image_id] = dict(
                    boxes=prediction['boxes'].float().cpu().numpy(),
                    scores=prediction['scores'].float().cpu().numpy(),
                    labels=prediction['labels'].cpu().numpy().astype(np.int64))
    return out


def to_coco_detections(preds):
    """Predictions as COCO detection dicts (xywh); returns (detections, number of dropped boxes)."""
    dets, dropped = [], 0
    for image_id, prediction in preds.items():
        for box, score, label in zip(
            prediction['boxes'].tolist(),
            prediction['scores'].tolist(),
            prediction['labels'].tolist()):
            if not 1 <= label <= 8 or box[2] <= box[0] or box[3] <= box[1]:
                dropped += 1
                continue
            dets.append(dict(image_id=int(image_id), category_id=int(label), score=float(score),
                             bbox=[box[0], box[1], box[2] - box[0], box[3] - box[1]]))
    return dets, dropped


def coco_eval(dets, gt):
    """Standard COCOeval (bbox); returns METRICS plus per-class AP."""
    img_ids = sorted(gt.getImgIds())
    assert {detection['image_id'] for detection in dets} <= set(img_ids), 'detections for images outside the ground truth'
    assert {detection['category_id'] for detection in dets} <= set(gt.getCatIds()), 'unknown category id'
    if not dets:
        return {**{k: 0.0 for k in METRICS}, 'per_class': {class_name: 0.0 for class_name in CLASSES}}
    with redirect_stdout(io.StringIO()):
        evaluator = COCOeval(gt, gt.loadRes([dict(detection) for detection in dets]), 'bbox')
        evaluator.params.imgIds = img_ids
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    assert stats[0] >= 0, 'COCOeval returned -1: invalid configuration'
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    per_class = {
        CLASSES[category_id - 1]: float(precision[:, :, k][precision[:, :, k] > -1].mean())
        if (precision[:, :, k] > -1).any() else float('nan')
        for k, category_id in enumerate(evaluator.params.catIds)}
    return dict(
        mAP=float(stats[0]), AP50=float(stats[1]), AP75=float(stats[2]), APs=float(stats[3]), APm=float(stats[4]),
        APl=float(stats[5]), AR=float(stats[8]), per_class=per_class, n_images=len(img_ids), n_detections=len(dets))


def evaluate_detections(dets, split='val'):
    """Evaluate COCO-format detections (e.g. from RT-DETR) with exactly the protocol used here."""
    gt = official_gt() if split == 'val' else make_gt(coco_json(split))
    return coco_eval(dets, gt)


def _pred_path(model, experiment, fold, tag):
    """File that stores the predictions of one checkpoint."""
    return ROOT / model / 'final_evaluation' / 'predictions' / f'{experiment}_fold{fold}_{tag}.npz'


def save_predictions(preds, path):
    """Store predictions as a compressed npz (an existing file is kept)."""
    box_image_ids = np.concatenate(
        [np.full(len(prediction['scores']), image_id, np.int64) for image_id, prediction in preds.items()])
    arrays = {
        key: np.concatenate([prediction[key] for prediction in preds.values()])
        for key in ('boxes', 'scores', 'labels')}
    if Path(path).exists():
        return path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + '.tmp.npz')
    np.savez_compressed(tmp_path, image_ids=box_image_ids, all_image_ids=np.asarray(sorted(preds), np.int64), **arrays)
    os.replace(tmp_path, path)
    return path


def load_predictions(path):
    """Inverse of save_predictions: {image id: dict(boxes, scores, labels)}."""
    with np.load(path) as npz:
        arrays = {k: npz[k] for k in npz.files}
    out = {int(image_id): dict(boxes=np.zeros((0, 4), np.float32), scores=np.zeros(0, np.float32),
                        labels=np.zeros(0, np.int64)) for image_id in arrays['all_image_ids']}
    order = np.argsort(arrays['image_ids'], kind='stable')
    uniq, starts = np.unique(arrays['image_ids'][order], return_index=True)
    for image_id, start, end in zip(uniq, starts, np.r_[starts[1:], len(order)]):
        selected = order[start:end]
        out[int(image_id)] = dict(
            boxes=arrays['boxes'][selected],
            scores=arrays['scores'][selected],
            labels=arrays['labels'][selected])
    return out


def evaluate_official(model, experiment, fold, verbose=True):
    """Official-validation COCO metrics of one best.pth; cached, resumable, never overwritten."""
    path = ROOT / model / 'final_evaluation' / 'official_val' / f'{experiment}_fold{fold}.json'
    record = checkpoint_record(model, experiment, fold)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['eval_id'] == EVAL_ID and saved['checkpoint_sha256'] == record['checkpoint_sha256']:
            return saved
        raise RuntimeError(f'{path} exists for a different checkpoint/config; move it away (not overwritten)')
    check_memory()
    start_time = time.perf_counter()
    net, _ = load_checkpoint(model, experiment, fold)
    image_ids = [int(image_info['id']) for image_info in coco_json('val')['images']]
    predictions = predict(net, 'val', image_ids, MODELS[model]['eval_batch'])
    del net
    torch.cuda.empty_cache()
    save_predictions(predictions, _pred_path(model, experiment, fold, 'official_val'))
    dets, dropped = to_coco_detections(predictions)
    metrics = coco_eval(dets, official_gt())
    out = {**{k: v for k, v in record.items() if k != 'cfg'}, 'evaluation_split': 'official_val', 'eval_id': EVAL_ID,
           'metrics': metrics, 'dropped_detections': dropped, 'seconds': time.perf_counter() - start_time,
           'created': datetime.datetime.now().isoformat(timespec='seconds'), 'source': 'final_evaluation.py'}
    _atomic(path, json.dumps(out, indent=2, default=_json_default))
    if verbose:
        print(f"{MODELS[model]['label']} {experiment} fold {fold}: official-val mAP {metrics['mAP']:.4f} "
              f"({out['seconds'] / 60:.1f} min)")
    return out


def stored_official(model):
    """Official-validation metrics written earlier by the training notebooks (where they exist)."""
    rows = []
    if model == 'faster_rcnn':
        table = pd.read_csv(MODELS[model]['results'] / 'official_val_folds.csv')
        for row in table.itertuples():
            rows.append(
                dict(
                    experiment=row.run, fold=int(row.fold), **{k: getattr(row, k) for k in METRICS},
                    original_ms_per_img=row.ms_per_img, stored_in='faster_rcnn/official_val_folds.csv'))
    else:
        stored = json.loads((MODELS[model]['results'] / 'official_validation.json').read_text())
        for experiment, metrics in stored.items():
            folds = [(load_result(model, experiment, fold)['metrics']['mAP'], fold) for fold in FOLDS]
            rows.append(dict(experiment=experiment, fold=max(folds)[1], **{k: metrics[k] for k in METRICS},
                             original_ms_per_img=np.nan, stored_in='fcos/official_validation.json'))
    return pd.DataFrame(rows)


def run_official_evaluation(model, verify=None):
    """Return official-validation metrics for all 15 checkpoints of a model.

    Faster R-CNN: the training notebook already evaluated all 15 checkpoints; those numbers are reused
    after recomputing the checkpoints in `verify` with this module and comparing.
    FCOS: only 2 of 15 checkpoints had official metrics, so all 15 are evaluated here (the 2 stored ones
    serve as a check)."""
    stored = stored_official(model)
    rows, checks = [], []
    todo = ([(experiment, fold) for experiment in MODELS[model]['experiments'] for fold in FOLDS]
            if model == 'fcos' else list(verify or []))
    for experiment, fold in todo:
        entry = evaluate_official(model, experiment, fold)
        stored_row = stored[(stored.experiment == experiment) & (stored.fold == fold)]
        if len(stored_row):
            diff = max(abs(entry['metrics'][k] - float(stored_row[k].iloc[0])) for k in METRICS)
            checks.append(dict(model=model, experiment=experiment, fold=fold, stored_mAP=float(stored_row.mAP.iloc[0]),
                               recomputed_mAP=entry['metrics']['mAP'], max_abs_diff_all_metrics=diff,
                               stored_in=stored_row.stored_in.iloc[0], agrees=diff <= 2e-3))
        if model == 'fcos':
            rows.append(
                dict(
                    experiment=experiment, fold=fold, **{k: entry['metrics'][k] for k in METRICS},
                    source='new: final_evaluation.py' + (' (matches stored notebook value)' if len(stored_row) else '')))
    if model == 'faster_rcnn':
        ok = all(check['agrees'] for check in checks)
        for entry in stored.itertuples():
            rows.append(
                dict(
                    experiment=entry.experiment, fold=entry.fold, **{k: getattr(entry, k) for k in METRICS},
                    source='existing: faster_rcnn.ipynb section 9' + (
                        ' (recomputed here, agrees)' if (entry.experiment, entry.fold) in todo and ok else '')))
        if not ok:
            print('WARNING: recomputed Faster R-CNN metrics differ from the stored ones:', checks)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def speed_image_ids():
    """The 200 official-validation frames of the speed protocol (40 per resolution, seed 0)."""
    images = pd.DataFrame(coco_json('val')['images'])
    assert images.groupby(['width', 'height']).size().min() >= 40
    picked = images.groupby(['width', 'height']).sample(n=40, random_state=SEED)
    return picked.sort_values('id')[['id', 'file_name', 'width', 'height']].reset_index(drop=True)


def _speed_images():
    """The speed frames decoded once and kept on the GPU as uint8 (cached)."""
    if 'speed_images' not in _cache:
        infos = {int(image_info['id']): image_info for image_info in coco_json('val')['images']}
        table = speed_image_ids()
        _cache['speed_images'] = [
            (int(image_id), (load_image('val', infos[int(image_id)]) * 255).round().to(torch.uint8).to(DEVICE))
            for image_id in table.id]
    return _cache['speed_images']


@torch.inference_mode()
def benchmark(net, batch_size, amp=True, warmup=20):
    """Time the model on same-resolution batches of the speed frames; latency in ms per image."""
    images = _speed_images()
    groups = {}
    for _, image in images:
        groups.setdefault(tuple(image.shape), []).append(image)
    batches = [
        group[start:start + batch_size]
        for group in groups.values()
        for start in range(0, len(group), batch_size)]

    def run(batch):
        with torch.autocast(DEVICE.type, enabled=amp and DEVICE.type == 'cuda'):
            return net([image.float().div(255) for image in batch])
    for step in range(warmup):
        run(batches[step % len(batches)])
    per_image = []
    for batch in batches:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        run(batch)
        torch.cuda.synchronize()
        per_image += [(time.perf_counter() - start_time) * 1000 / len(batch)] * len(batch)
    timings = np.asarray(per_image)
    return dict(
        ms_per_image=float(timings.mean()), ms_median=float(np.median(timings)),
        ms_p90=float(np.percentile(timings, 90)), fps=float(1000 / timings.mean()), n_images=len(timings),
        n_batches=len(batches))


def measure_speed(model, experiment, fold, batch_size=1, amp=True):
    """Speed of one checkpoint; cached, never overwritten."""
    path = ROOT / model / 'final_evaluation' / 'speed' / f'{experiment}_fold{fold}_bs{batch_size}_{"amp" if amp else "fp32"}.json'
    record = checkpoint_record(model, experiment, fold)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['speed_id'] == SPEED_ID and saved['checkpoint_sha256'] == record['checkpoint_sha256']:
            return saved
        raise RuntimeError(f'{path} exists for a different checkpoint/protocol; move it away (not overwritten)')
    check_memory()
    net, cfg = load_checkpoint(model, experiment, fold)
    speed = benchmark(net, batch_size, amp)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    out = {**{k: v for k, v in record.items() if k != 'cfg'}, 'speed_id': SPEED_ID, 'batch_size': batch_size,
           'precision': 'amp_fp16' if amp else 'fp32', 'gpu': torch.cuda.get_device_name(0),
           'input_resize': f"min_size={cfg['min_size']}, max_size={cfg['max_size']} (inside the model)", **speed,
           'created': datetime.datetime.now().isoformat(timespec='seconds')}
    _atomic(path, json.dumps(out, indent=2, default=_json_default))
    return out


def run_speed(model, representative=None):
    """Speed table of all checkpoints of a model (batch sizes 1 and 4), plus the fp32 representative checkpoint."""
    rows = []
    for experiment in MODELS[model]['experiments']:
        for fold in FOLDS:
            for batch_size in (1, 4):
                rows.append(measure_speed(model, experiment, fold, batch_size))
    if representative is not None:
        rows.append(measure_speed(model, *representative, batch_size=1, amp=False))
    keep = ['model', 'experiment', 'fold', 'batch_size', 'precision', 'ms_per_image', 'ms_median', 'ms_p90', 'fps',
            'n_images', 'n_batches', 'gpu', 'input_resize']
    table = pd.DataFrame(rows)[keep]
    training_ms_per_img = {
        (experiment, fold): load_result(model, experiment, fold)['ms_per_img']
        for experiment in MODELS[model]['experiments']
        for fold in FOLDS}
    table['original_training_ms_per_img'] = [
        training_ms_per_img[(experiment, fold)]
        for experiment, fold in zip(table.experiment, table.fold)]
    return table


def gpu_status():
    """Name, utilisation and memory of the GPU from nvidia-smi."""
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
        name, util, used, total = [value.strip() for value in output.strip().split(',')]
        return dict(gpu=name, utilization_pct=int(util), memory_used_mib=int(used), memory_total_mib=int(total))
    except Exception as exc:
        return dict(error=str(exc))


def _fcos_origins():
    """Where each FCOS fold was trained, from the read-only backup and the local run log."""
    run_log = (MODELS['fcos']['results'] / 'fcos_run_log.txt').read_text(encoding='utf-8', errors='replace')
    resumed = {}
    for line in run_log.splitlines():
        if line.startswith('Resumed '):
            name, epoch = line.split()[1], line.split()[-1].split('/')[0]
            resumed.setdefault(name, []).append(int(epoch))
    origins = {}
    for experiment in MODELS['fcos']['experiments']:
        for fold in FOLDS:
            source_folder = FCOS_SOURCE / experiment / f'fold_{fold}'
            history_file = source_folder / 'history.csv'
            if history_file.exists() and (pd.read_csv(history_file).mAP < 0).any():
                origins[experiment, fold] = 'Colab run had mAP=-1 (old maxDets bug); retrained locally from scratch'
            elif (source_folder / 'result.json').exists():
                identical = sha256(source_folder / 'best.pth') == sha256(run_dir('fcos', experiment, fold) / 'best.pth')
                origins[experiment, fold] = 'Colab (restored from read-only backup' + (
                    ', identical best.pth)' if identical else '; best.pth DIFFERS)')
            elif (source_folder / 'last.pth').exists():
                num_epochs = len(pd.read_csv(history_file)) if history_file.exists() else '?'
                origins[experiment, fold] = f'Colab epochs 1-{num_epochs}; resumed locally from last.pth'
            else:
                origins[experiment, fold] = 'local RTX 5070'
    if 3 in resumed.get('F1_ciou', []):
        origins['F1_ciou', 4] = 'local; interrupted after epoch 3 and resumed locally from last.pth'
    return origins


def training_inventory():
    """One row per training run with consistency checks of its result, history and checkpoint files."""
    rows = []
    origins = _fcos_origins()
    for model, spec in MODELS.items():
        for experiment in spec['experiments']:
            for fold in FOLDS:
                folder = run_dir(model, experiment, fold)
                issues = []
                result = load_result(model, experiment, fold) if (folder / 'result.json').exists() else None
                if result is None:
                    issues.append('result.json missing')
                if not (folder / 'best.pth').exists():
                    issues.append('best.pth missing')
                history = pd.DataFrame(result['history']) if result else pd.DataFrame(columns=['epoch', 'mAP'])
                if result:
                    if result['name'] != experiment or int(result['fold']) != fold:
                        issues.append('name/fold mismatch in result.json')
                    if list(history.epoch) != list(range(1, 7)):
                        issues.append(f'history epochs {list(history.epoch)}')
                    if (history.mAP < 0).any() or result['metrics']['mAP'] < 0:
                        issues.append('invalid mAP (-1)')
                    if int(history.loc[history.mAP.idxmax(), 'epoch']) != int(result['best_epoch']):
                        issues.append('best_epoch is not the argmax of history mAP')
                    if abs(float(history.loc[history.epoch == result['best_epoch'], 'mAP'].iloc[0])
                           - result['metrics']['mAP']) > 1e-9:
                        issues.append('result metrics differ from history at best epoch')
                if result and (folder / 'best.pth').exists():
                    checkpoint_data = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
                    if int(checkpoint_data['epoch']) != int(result['best_epoch']):
                        issues.append('best.pth epoch differs')
                    if abs(checkpoint_data['metrics']['mAP'] - result['metrics']['mAP']) > 1e-9:
                        issues.append('best.pth metrics differ')
                    if model == 'fcos':
                        if checkpoint_data['cfg'] != result['cfg']:
                            issues.append('best.pth cfg differs')
                        history_csv = pd.read_csv(folder / 'history.csv')
                        if len(history_csv) != len(history) or np.abs(history_csv.mAP.values - history.mAP.values).max() > 1e-9:
                            issues.append('history.csv differs from result.json')
                        last = torch.load(folder / 'last.pth', map_location='cpu', weights_only=False)
                        if int(last['epoch']) != 6 or int(last['best_epoch']) != int(result['best_epoch']):
                            issues.append('last.pth state differs')
                        del last
                    del checkpoint_data
                notes = []
                if model == 'fcos' and origins[experiment, fold].startswith('Colab (restored'):
                    notes.append('training-time ms_per_img was measured on the Colab GPU')
                rows.append(
                    dict(
                        model=spec['label'], experiment=experiment, fold=fold,
                        status='complete' if not issues else 'CHECK', expected_epochs=6, completed_epochs=len(history),
                        best_epoch=result['best_epoch'] if result else None, best_checkpoint=str(folder / 'best.pth'),
                        cv_map=result['metrics']['mAP'] if result else None,
                        cv_map_50=result['metrics']['AP50'] if result else None,
                        cv_map_small=result['metrics']['APs'] if result else None,
                        trained_on=origins.get(
                            (experiment, fold),
                            'local RTX 5070 (mid-epoch resume supported; ' 'resume events are not stored in the result files)'),
                        issues='; '.join(issues) or 'none', notes='; '.join(notes)))
    return pd.DataFrame(rows)


def overlap_check():
    """Exact-duplicate and video-name checks between train, official validation and test images."""
    path = FINAL / 'audit' / 'overlap_check.json'
    if path.exists():
        return json.loads(path.read_text())
    train, val = coco_json('train'), coco_json('val')
    test = pd.read_csv(DATA / 'test_images.csv')
    train_videos, val_videos = ({image_info['video'] for image_info in train['images']},
                                {image_info['video'] for image_info in val['images']})
    hashes = {}
    for split, files in [('train', [image_info['file_name'] for image_info in train['images']]),
                         ('val', [image_info['file_name'] for image_info in val['images']]),
                         ('test', [file_name.split('/', 1)[1] for file_name in test.file_name])]:
        with ThreadPoolExecutor(8) as pool:
            hashes[split] = set(
                pool.map(
                    lambda file_name,
                    split_name=split: hashlib.md5((DATA / split_name / file_name).read_bytes()).hexdigest(),
                    files))
    folds = pd.read_csv(FOLDS_CSV)
    train_files = {image_info['file_name'] for image_info in train['images']}
    out = dict(
        train_videos=len(train_videos), val_videos=len(val_videos),
        shared_video_names=sorted(train_videos & val_videos),
        shared_source_flights=sorted(
            {video.split('_')[0] for video in train_videos} & {video.split('_')[0] for video in val_videos}),
        train_images=len(train['images']), val_images=len(val['images']), test_images=len(test),
        identical_files_train_val=len(hashes['train'] & hashes['val']),
        identical_files_train_test=len(hashes['train'] & hashes['test']),
        identical_files_val_test=len(hashes['val'] & hashes['test']), cv_frames=len(folds),
        cv_frames_all_from_train_split=bool(set(folds.file_name) <= train_files),
        cv_videos_in_val=sorted(set(folds.video) & val_videos),
        note='image ids are numbered separately per split, so ids are compared within a split only; '
             'exact duplicates are detected by MD5 of the JPEG files (near-duplicates are not checked)',
        test_labels_available=(DATA / 'test' / 'annotations_coco.json').exists())
    save_json(out, path)
    return out


def fold_table(model, official):
    """Per-fold metrics of both splits (CV out-of-fold and official validation) with speed."""
    rows = []
    for experiment in MODELS[model]['experiments']:
        for fold in FOLDS:
            result = load_result(model, experiment, fold)
            speed = measure_speed(model, experiment, fold, 1)
            base = dict(
                model=MODELS[model]['label'], experiment=experiment, fold=fold, best_epoch=int(result['best_epoch']),
                ms_per_image=speed['ms_per_image'], fps=speed['fps'])
            rows.append(
                dict(
                    base, evaluation_split='cv_out_of_fold', **{COLUMNS[k]: result['metrics'][k] for k in METRICS},
                    n_images=int(result['val_frames']), metrics_source=f"{MODELS[model]['notebook']} result.json"))
            official_row = official[(official.experiment == experiment) & (official.fold == fold)].iloc[0]
            rows.append(
                dict(
                    base, evaluation_split='official_val', **{COLUMNS[k]: float(official_row[k]) for k in METRICS},
                    n_images=len(coco_json('val')['images']), metrics_source=official_row.source))
    table = pd.DataFrame(rows)
    order = ['model', 'experiment', 'fold', 'best_epoch', 'evaluation_split'] + list(COLUMNS.values()) + \
            ['ms_per_image', 'fps', 'n_images', 'metrics_source']
    table = table[order].sort_values(['evaluation_split', 'experiment', 'fold']).reset_index(drop=True)
    table['speed_protocol'] = 'standardised: RTX 5070, batch 1, AMP, 200 official-val frames (see inference_speed.csv)'
    return table


def summarize(folds):
    """Mean and sample standard deviation (ddof=1, n=5 folds) per model, experiment and split."""
    metric_columns = list(COLUMNS.values()) + ['ms_per_image', 'fps']
    grouped = folds.groupby(['model', 'experiment', 'evaluation_split'])
    out = grouped[metric_columns].mean().add_suffix('_mean').join(
        grouped[metric_columns].std(ddof=1).add_suffix('_std'))
    out['n_folds'] = grouped.size()
    return out.reset_index()


def epoch_metrics(models=MODELS):
    """Validation metrics of every epoch of every training run of `models`."""
    rows = []
    names = dict(mAP='map', AP50='map_50', AP75='map_75', APs='map_small', APm='map_medium', APl='map_large')
    for model, spec in models.items():
        for experiment in spec['experiments']:
            for fold in FOLDS:
                for epoch_record in load_result(model, experiment, fold)['history']:
                    rows.append(
                        dict(
                            model=spec['label'], experiment=experiment, fold=fold, epoch=int(epoch_record['epoch']),
                            evaluation_split='cv_out_of_fold',
                            **{v: epoch_record.get(k, np.nan) for k, v in names.items()}))
    return pd.DataFrame(rows)


def representative_fold(official, experiment):
    """Fold whose official-val mAP is the median of the five (a typical, not the luckiest, model)."""
    experiment_rows = official[official.experiment == experiment].sort_values('mAP').reset_index(drop=True)
    return int(experiment_rows.loc[len(experiment_rows) // 2, 'fold'])


def _ioa(boxes, regions):
    """For each box, the largest intersection-over-own-area with any ignore region."""
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes))
    top_left = np.maximum(boxes[:, None, :2], regions[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], regions[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(-1)
    area = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)
    return (intersection / area[:, None]).max(1)


def classify_detections(prediction, target, threshold, iou_thr=0.5):
    """Greedy (by score) per-image diagnosis at a score threshold, COCO-style matching at IoU 0.5.

    Detection types: tp, duplicate (same class, GT already taken), classification (IoU>=0.5 with another
    class only), localization (same class, 0.1<=IoU<0.5), background (IoU<0.1), ignored (IoA>=0.5 with
    an ignore region and no true positive). Unmatched ground truth = false negatives."""
    keep = prediction['scores'] >= threshold
    order = np.argsort(-prediction['scores'][keep], kind='stable')
    pred_boxes, pred_labels, pred_scores = (prediction['boxes'][keep][order], prediction['labels'][keep][order],
                                            prediction['scores'][keep][order])
    gt_boxes, gt_labels = target['boxes'], target['labels']
    ious = (box_iou(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).numpy()
            if len(pred_boxes) and len(gt_boxes) else np.zeros((len(pred_boxes), len(gt_boxes))))
    ignored = _ioa(pred_boxes, target['ignore']) >= 0.5
    taken = np.zeros(len(gt_boxes), bool)
    kinds = []
    for i in range(len(pred_boxes)):
        same_class_iou = np.where(gt_labels == pred_labels[i], ious[i], -1) if len(gt_boxes) else np.zeros(0)
        free_iou = np.where(taken, -1, same_class_iou) if len(gt_boxes) else np.zeros(0)
        if len(gt_boxes) and free_iou.max() >= iou_thr:
            taken[int(free_iou.argmax())] = True
            kinds.append('tp')
        elif ignored[i]:
            kinds.append('ignored')
        elif len(gt_boxes) and same_class_iou.max() >= iou_thr:
            kinds.append('duplicate')
        elif len(gt_boxes) and ious[i].max() >= iou_thr:
            kinds.append('classification')
        elif len(gt_boxes) and same_class_iou.max() >= 0.1:
            kinds.append('localization')
        else:
            kinds.append('background')
    return dict(
        boxes=pred_boxes, labels=pred_labels, scores=pred_scores, kinds=np.asarray(kinds, dtype=object),
        gt_matched=taken)


def size_name(boxes):
    """COCO size class ('small', 'medium', 'large') of each xyxy box."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.where(area < 32 ** 2, 'small', np.where(area < 96 ** 2, 'medium', 'large'))


def select_threshold(model, experiment, fold):
    """Score threshold for the diagnosis, chosen on the checkpoint's own held-out CV fold (max F1 @ IoU .5)."""
    path = ROOT / model / 'final_evaluation' / 'failure_analysis' / f'threshold_{experiment}_fold{fold}.json'
    if path.exists():
        return json.loads(path.read_text())
    net, _ = load_checkpoint(model, experiment, fold)
    ids = cv_fold_ids(fold)
    preds = predict(net, 'train', ids, MODELS[model]['eval_batch'])
    del net
    torch.cuda.empty_cache()
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    stats = []
    for score_threshold in grid:
        tp = fp = fn = 0
        for image_id in ids:
            diagnosis = classify_detections(preds[image_id], image_targets('train', image_id), score_threshold)
            tp += int((diagnosis['kinds'] == 'tp').sum())
            fp += int(np.isin(diagnosis['kinds'], ['duplicate', 'classification', 'localization', 'background']).sum())
            fn += int((~diagnosis['gt_matched']).sum())
        stats.append(dict(threshold=float(score_threshold), tp=tp, fp=fp, fn=fn, f1=2 * tp / max(2 * tp + fp + fn, 1)))
    best = max(stats, key=lambda row: row['f1'])
    out = dict(model=model, experiment=experiment, fold=fold, selected_on='CV held-out fold (train split frames)',
               threshold=best['threshold'], f1=best['f1'], grid=stats)
    save_json(out, path)
    return out


def failure_analysis(model, experiment, fold, threshold):
    """Error diagnosis of one checkpoint on the official validation set: per-image table, totals, recall by size."""
    preds = load_predictions(_pred_path(model, experiment, fold, 'official_val'))
    per_image, totals = [], {}
    fn_by_size = {size: [0, 0] for size in ('small', 'medium', 'large')}
    for image_id, prediction in preds.items():
        target = image_targets('val', image_id)
        diagnosis = classify_detections(prediction, target, threshold)
        kinds = pd.Series(diagnosis['kinds'], dtype=object).value_counts().to_dict()
        sizes = size_name(target['boxes']) if len(target['boxes']) else np.zeros(0, dtype=object)
        for size in fn_by_size:
            fn_by_size[size][0] += int((sizes == size).sum())
            fn_by_size[size][1] += int(((sizes == size) & ~diagnosis['gt_matched']).sum())
        row = dict(
            image_id=image_id, video=target['info']['video'], frame_id=target['info']['frame_id'],
            n_gt=len(target['boxes']),
            **{
                k: int(kinds.get(k, 0))
                for k in ('tp', 'duplicate', 'classification', 'localization', 'background', 'ignored')},
            missed=int((~diagnosis['gt_matched']).sum()),
            missed_small=int(((sizes == 'small') & ~diagnosis['gt_matched']).sum()))
        per_image.append(row)
    table = pd.DataFrame(per_image)
    for k in ('tp', 'duplicate', 'classification', 'localization', 'background', 'ignored', 'missed', 'n_gt'):
        totals[k] = int(table[k].sum())
    sizes = pd.DataFrame(
        [
            dict(size=size, n_gt=num_gt, missed=num_missed, recall=1 - num_missed / max(num_gt, 1))
            for size, (num_gt, num_missed) in fn_by_size.items()])
    return table, totals, sizes


def pick_failure_cases(table, num_cases=4):
    """Pick one image per failure type, never the same image twice."""
    picks, used = [], set()
    for kind, title in [('missed_small', 'most missed small objects (false negatives)'),
                        ('classification', 'most classification errors'),
                        ('localization', 'most localization errors'),
                        ('background', 'most background false positives')][:num_cases]:
        candidates = table[~table.image_id.isin(used)].sort_values([kind, 'n_gt'], ascending=[False, True])
        image_id = int(candidates.image_id.iloc[0])
        used.add(image_id)
        picks.append(dict(case=kind, title=title, image_id=image_id, count=int(candidates[kind].iloc[0])))
    return picks


def representative_images():
    """Official-val frames chosen from ground-truth statistics only (not from predictions)."""
    path = FINAL / 'visualizations' / 'representative_images.json'
    if path.exists():
        return json.loads(path.read_text())
    rows = []
    for image_info in coco_json('val')['images']:
        target = image_targets('val', image_info['id'])
        rows.append(
            dict(
                image_id=int(image_info['id']), video=image_info['video'], n_gt=len(target['labels']),
                n_small=int((size_name(target['boxes']) == 'small').sum()) if len(target['boxes']) else 0,
                n_classes=len(set(target['labels'].tolist())), n_heavy_occlusion=int((target['occlusion'] == 2).sum())))
    table = pd.DataFrame(rows)
    sample = table.iloc[::5].copy()
    infos = {int(image_info['id']): image_info for image_info in coco_json('val')['images']}

    def brightness(image_id):
        with Image.open(DATA / 'val' / infos[image_id]['file_name']) as image:
            return float(np.asarray(image.convert('L').reduce(8)).mean())
    with ThreadPoolExecutor(4) as pool:
        sample['brightness'] = list(pool.map(brightness, sample.image_id))
    table = table.merge(sample[['image_id', 'brightness']], on='image_id', how='left')
    picks, used_videos, used = [], set(), set()
    specs = [('small_objects', 'n_small', False), ('dense_scene', 'n_gt', False), ('many_classes', 'n_classes', False),
             ('heavy_occlusion', 'n_heavy_occlusion', False), ('low_light_background', 'brightness', True)]
    for tag, col, ascending in specs:
        cand = table.dropna(subset=[col]).sort_values([col, 'n_gt'], ascending=[ascending, False])
        fresh = cand[~cand.video.isin(used_videos) & ~cand.image_id.isin(used)]
        row = (fresh if len(fresh) else cand[~cand.image_id.isin(used)]).iloc[0]
        picks.append(dict(tag=tag, image_id=int(row.image_id), video=row.video, criterion=col, value=float(row[col])))
        used_videos.add(row.video)
        used.add(int(row.image_id))
    random_row = table[~table.image_id.isin(used)].sample(1, random_state=SEED).iloc[0]
    picks.append(
        dict(
            tag='random', image_id=int(random_row.image_id), video=random_row.video, criterion='random (seed 0)',
            value=0.0))
    save_json(picks, path)
    return picks


KIND_STYLE = {'tp': ('#1f77b4', 'true positive'), 'background': ('#d62728', 'FP: background'),
              'classification': ('#ff7f0e', 'FP: wrong class'), 'localization': ('#e377c2', 'FP: poor localization'),
              'duplicate': ('#9467bd', 'FP: duplicate'), 'ignored': ('#7f7f7f', 'in ignore region'),
              'fn': ('#2ca02c', 'missed ground truth (FN)')}


def _box_trace(boxes, color, name, texts, dash='solid', width=1.5, showlegend=True):
    """Plotly traces (outline + hover markers) of a group of boxes."""
    import plotly.graph_objects as go
    xs, ys = [], []
    for x1, y1, x2, y2 in boxes:
        xs += [x1, x2, x2, x1, x1, None]
        ys += [y1, y1, y2, y2, y1, None]
    lines = go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, width=width, dash=dash), name=name,
                       legendgroup=name, hoverinfo='skip', showlegend=showlegend)
    marks = go.Scatter(
        x=[box[0] for box in boxes], y=[box[1] for box in boxes], mode='markers', marker=dict(size=4, color=color),
        text=texts, hoverinfo='text', name=name, legendgroup=name, showlegend=False)
    return [lines, marks]


def _image_source(image_id, max_width=1600):
    """Validation image as a base64 jpeg data uri for a Plotly layout image."""
    import base64
    info = image_targets('val', image_id)['info']
    with Image.open(DATA / 'val' / info['file_name']) as image:
        image = image.convert('RGB')
        scale = min(1.0, max_width / image.width)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=82)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def prediction_figure(image_id, panels, threshold_note=''):
    """panels: list of (title, prediction dict, threshold). Ground truth and ignore regions shown in every panel."""
    from plotly.subplots import make_subplots
    target = image_targets('val', image_id)
    width, height = target['info']['width'], target['info']['height']
    fig = make_subplots(rows=len(panels), cols=1, subplot_titles=[panel[0] for panel in panels], vertical_spacing=0.06)
    image_source = _image_source(image_id)
    for panel_index, (title, pred, threshold) in enumerate(panels, 1):
        diagnosis = classify_detections(pred, target, threshold)
        first = panel_index == 1
        for kind, (color, label) in KIND_STYLE.items():
            if kind == 'fn':
                selected = ~diagnosis['gt_matched']
                boxes, texts = target['boxes'][selected], [
                    f'missed {CLASSES[label_id - 1]}'
                    for label_id in target['labels'][selected]]
                dash = 'dot'
            else:
                selected = diagnosis['kinds'] == kind
                boxes = diagnosis['boxes'][selected]
                texts = [
                    f'{CLASSES[label_id - 1]} {score:.2f} ({label})'
                    for label_id, score in zip(diagnosis['labels'][selected], diagnosis['scores'][selected])]
                dash = 'solid'
            if len(boxes):
                for trace in _box_trace(boxes.tolist(), color, label, texts, dash, showlegend=first):
                    fig.add_trace(trace, row=panel_index, col=1)
        if len(target['ignore']):
            for trace in _box_trace(
                target['ignore'].tolist(), 'rgba(255,255,255,0.8)', 'ignore region',
                ['ignore region'] * len(target['ignore']), 'dash', 1, showlegend=first):
                fig.add_trace(trace, row=panel_index, col=1)
        xref = 'x' if panel_index == 1 else f'x{panel_index}'
        yref = 'y' if panel_index == 1 else f'y{panel_index}'
        fig.add_layout_image(
            source=image_source, xref=xref, yref=yref, x=0, y=0, sizex=width, sizey=height, xanchor='left',
            yanchor='top', sizing='stretch', layer='below')
        fig.update_xaxes(range=[0, width], showgrid=False, visible=False, row=panel_index, col=1)
        fig.update_yaxes(range=[height, 0], showgrid=False, visible=False, scaleanchor=xref, row=panel_index, col=1)
    info = target['info']
    fig.update_layout(title=f"Official validation · {info['video']} frame {info['frame_id']} · {width}x{height} · "
                            f"{len(target['boxes'])} GT objects {threshold_note}",
                      height=int(560 * len(panels) * min(1.0, height / width * 1.8)) + 120, width=1200,
                      legend=dict(orientation='h', y=-0.02), margin=dict(l=10, r=10, t=70, b=40),
                      plot_bgcolor='black')
    return fig


def official_available(model):
    """True if the official-validation metrics of all checkpoints of the model exist."""
    if model == 'faster_rcnn':
        return (MODELS[model]['results'] / 'official_val_folds.csv').exists()
    return all((ROOT / model / 'final_evaluation' / 'official_val' / f'{experiment}_fold{fold}.json').exists()
               for experiment in MODELS[model]['experiments'] for fold in FOLDS)


VERIFY = {'faster_rcnn': [('lr_scheduler', 4), ('baseline', 0)], 'fcos': None}


def model_section(model):
    """Official validation, speed, fold tables and selection for one model (cached and resumable)."""
    official, checks = run_official_evaluation(model, VERIFY[model])
    cv_means = {
        experiment: np.mean([load_result(model, experiment, fold)['metrics']['mAP'] for fold in FOLDS])
        for experiment in MODELS[model]['experiments']}
    best_cv = max(cv_means, key=cv_means.get)
    representative = representative_fold(official, best_cv)
    speed = run_speed(model, representative=(best_cv, representative))
    folds = fold_table(model, official)
    summary = summarize(folds)
    off_means = official.groupby('experiment').mAP.mean().to_dict()
    best_off = max(off_means, key=off_means.get)
    selection = dict(
        model=MODELS[model]['label'], best_by_cv=best_cv, cv_mean_map=cv_means, best_by_official_val=best_off,
        official_val_mean_map=off_means, agree=best_cv == best_off, representative_fold=representative,
        representative_checkpoint=str(run_dir(model, best_cv, representative) / 'best.pth'),
        representative_rule='fold with the median official-validation mAP of the CV-selected experiment',
        fold_checkpoints=[str(run_dir(model, best_cv, fold) / 'best.pth') for fold in FOLDS],
        rule='experiment selected by mean CV (out-of-fold) mAP; official validation only as an independent check')
    out = ROOT / model / 'final_evaluation'
    save_csv(folds, out / f'{model}_folds.csv')
    save_csv(summary, out / f'{model}_summary_mean_std.csv')
    save_csv(speed, out / f'{model}_inference_speed.csv')
    save_json(selection, out / 'selection.json')
    if len(checks):
        save_csv(checks, out / 'official_val_verification.csv')
    return dict(official=official, checks=checks, speed=speed, folds=folds, summary=summary, selection=selection)


def failure_section(model, experiment, fold):
    """Threshold from the CV fold, error diagnosis on the official validation set, failure-case figures."""
    threshold = select_threshold(model, experiment, fold)['threshold']
    per_image, totals, sizes = failure_analysis(model, experiment, fold, threshold)
    out = ROOT / model / 'final_evaluation' / 'failure_analysis'
    save_csv(per_image, out / f'per_image_errors_{experiment}_fold{fold}.csv')
    save_csv(sizes, out / f'recall_by_size_{experiment}_fold{fold}.csv')
    cases = pick_failure_cases(per_image)
    preds = load_predictions(_pred_path(model, experiment, fold, 'official_val'))
    label = MODELS[model]['label']
    for case_number, case in enumerate(cases, 1):
        case['figure'] = str(
            save_html(
                prediction_figure(
                    case['image_id'],
                    [
                        (
                            f'{label} · {experiment} fold {fold} · score ≥ {threshold:.2f} · '
                            f'failure case {case_number}: {case["title"]}',
                            preds[case['image_id']],
                            threshold)]),
                FINAL / 'visualizations' / f'failure_{model}_{case_number}_{case["case"]}.html'))
        row = per_image[per_image.image_id == case['image_id']].iloc[0].to_dict()
        case.update({key: row[key] for key in ('video', 'frame_id', 'n_gt', 'tp', 'missed', 'missed_small',
                                               'classification', 'localization', 'background', 'duplicate')})
    summary = dict(model=label, experiment=experiment, fold=fold, threshold=threshold, totals=totals,
                   recall_by_size=sizes.to_dict('records'), cases=cases,
                   note='threshold-based diagnosis at IoU 0.5 for analysis only; COCO metrics are the reference')
    save_json(summary, out / f'failure_summary_{experiment}_fold{fold}.json')
    return summary


def _selection(model):
    """selection.json of a model, or None if it was not written yet."""
    path = ROOT / model / 'final_evaluation' / 'selection.json'
    return json.loads(path.read_text()) if path.exists() else None


def _latest_csv(path):
    """Newest timestamped version of a table written by save_new."""
    path = Path(path)
    versions = sorted(path.parent.glob(f'{path.stem}*{path.suffix}'), key=lambda file: file.stat().st_mtime)
    return pd.read_csv(versions[-1]) if versions else None


def build_comparison(rtdetr=None):
    """Cross-model tables and Plotly figures from the cached per-model results (nothing is recomputed)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    save_csv(
        pd.DataFrame(
            [
                dict(
                    image_id=image_info['id'], file_name=image_info['file_name'], video=image_info['video'],
                    width=image_info['width'], height=image_info['height'])
                for image_info in coco_json('val')['images']]),
        FINAL / 'audit' / 'official_val_image_ids.csv')
    save_csv(speed_image_ids(), FINAL / 'audit' / 'speed_image_ids.csv')
    save_json(
        dict(coco=EVAL_CONFIG, coco_id=EVAL_ID, speed=SPEED_CONFIG, speed_id=SPEED_ID),
        FINAL / 'audit' / 'evaluation_config.json')
    ready = [model for model in COMPARED_MODELS if _selection(model) is not None]
    folds = pd.concat([_latest_csv(ROOT / model / 'final_evaluation' / f'{model}_folds.csv') for model in ready], ignore_index=True)
    speed = pd.concat(
        [_latest_csv(ROOT / model / 'final_evaluation' / f'{model}_inference_speed.csv') for model in ready],
        ignore_index=True)
    summary = summarize(folds)
    selection = {model: _selection(model) for model in ready}
    label_to_key = {COMPARED_MODELS[model]['label']: model for model in COMPARED_MODELS}
    summary['selected_by_cv'] = [
        selection[label_to_key[row.model]]['best_by_cv'] == row.experiment
        for row in summary.itertuples()]
    summary['best_by_official_val'] = [selection[label_to_key[row.model]]['best_by_official_val'] == row.experiment
                                       for row in summary.itertuples()]
    comparison = summary
    if 'rt_detr' not in ready:
        rtdetr_rows = pd.DataFrame(
            [
                dict(model='RT-DETR', experiment='(teammate: fill in after verified results)', evaluation_split=split)
                for split in ('cv_out_of_fold', 'official_val')])
        comparison = pd.concat([summary, rtdetr_rows], ignore_index=True)
    out = FINAL
    save_csv(comparison, out / 'model_comparison_summary.csv')
    save_json(dict(std_convention='sample standard deviation over the 5 folds (ddof=1)',
                   splits=dict(cv_out_of_fold='each fold model on its own held-out training videos',
                               official_val='every fold model on the official validation set (2846 frames, 7 videos)'),
                   selection=selection, summary=comparison.replace({np.nan: None}).to_dict('records')),
              out / 'model_comparison_summary.json')
    save_csv(epoch_metrics(COMPARED_MODELS), out / 'epoch_metrics.csv')
    save_csv(speed, out / 'inference_speed.csv')
    rows = []
    for model in ready:
        entry = summary[
            (summary.model == COMPARED_MODELS[model]['label']) & (summary.experiment == selection[model]['best_by_cv'])]
        for row in entry.itertuples():
            rows.append(
                dict(
                    model=COMPARED_MODELS[model]['label'], experiment=row.experiment, evaluation_split=row.evaluation_split,
                    **{
                        column: f"{getattr(row, column + '_mean'):.4f} ± {getattr(row, column + '_std'):.4f}"
                        for column in COLUMNS.values()},
                    ms_per_image=f"{row.ms_per_image_mean:.1f} ± {row.ms_per_image_std:.1f}", fps=f"{row.fps_mean:.1f}"))
    if 'rt_detr' not in ready:
        rows += [
            dict(model='RT-DETR', experiment='', evaluation_split=split)
            for split in ('cv_out_of_fold', 'official_val')]
    save_csv(pd.DataFrame(rows), out / 'comparison_ready_table.csv')
    figures = _comparison_figures(folds, summary, speed, selection, ready, go, make_subplots)
    for name, fig in figures.items():
        save_html(fig, FINAL / 'plots' / f'{name}.html')
    _representative_figures(ready, selection)
    return comparison, pd.DataFrame(rows)


def _exp_label(model, experiment):
    """Legend label of an experiment."""
    return f"{COMPARED_MODELS[model]['label']} · {experiment}"


def _comparison_figures(folds, summary, speed, selection, ready, go, make_subplots):
    """The cross-model Plotly figures, as {file name: figure}."""
    figs = {}
    epochs = epoch_metrics(COMPARED_MODELS)
    exps = [(model, experiment) for model in ready for experiment in COMPARED_MODELS[model]['experiments']]
    fig = make_subplots(rows=math.ceil(len(exps) / 3), cols=3, shared_yaxes=True,
                        subplot_titles=[_exp_label(model, experiment) for model, experiment in exps])
    for i, (model, experiment) in enumerate(exps):
        row, col = i // 3 + 1, i % 3 + 1
        sub = epochs[(epochs.model == COMPARED_MODELS[model]['label']) & (epochs.experiment == experiment)]
        for fold in FOLDS:
            fold_rows = sub[sub.fold == fold]
            fig.add_trace(
                go.Scatter(
                    x=fold_rows.epoch, y=fold_rows['map'], mode='lines+markers', name=f'fold {fold}',
                    legendgroup=f'fold {fold}', showlegend=i == 0,
                    line=dict(color=['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'][fold])),
                row=row,
                col=col)
    fig.update_yaxes(title_text='mAP@[.5:.95]', col=1)
    fig.update_xaxes(title_text='epoch', dtick=1)
    fig.update_layout(title='Validation mAP by epoch, each fold · split: CV out-of-fold (held-out training videos)',
                      height=330 * math.ceil(len(exps) / 3) + 80, width=1200)
    figs['cv_map_by_epoch_all_folds'] = fig
    fig = make_subplots(rows=1, cols=2, subplot_titles=['mAP@[.5:.95]', 'AP_small'])
    for model, experiment in exps:
        sub = epochs[(epochs.model == COMPARED_MODELS[model]['label']) & (epochs.experiment == experiment)]
        for col, key in enumerate(['map', 'map_small'], 1):
            grouped = sub.groupby('epoch')[key].agg(['mean', 'std'])
            fig.add_trace(
                go.Scatter(
                    x=grouped.index, y=grouped['mean'], error_y=dict(type='data', array=grouped['std'], thickness=1),
                    mode='lines+markers', name=_exp_label(model, experiment), legendgroup=experiment,
                    showlegend=col == 1,
                    line=dict(color=COLORS[experiment], dash='solid' if model == 'faster_rcnn' else 'dash')),
                row=1,
                col=col)
    fig.update_xaxes(title_text='epoch', dtick=1)
    fig.update_layout(title='Mean ± std over 5 folds by epoch · split: CV out-of-fold', height=480, width=1200)
    figs['cv_map_by_epoch_mean'] = fig
    split_title = {'cv_out_of_fold': 'CV out-of-fold (mean ± std, 5 folds)',
                   'official_val': 'Official validation (mean ± std, 5 fold models)'}
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(split_title.values()), shared_yaxes=True)
    for col, split in enumerate(split_title, 1):
        for model, experiment in exps:
            fold_rows = folds[
                (folds.model == COMPARED_MODELS[model]['label']) & (folds.experiment == experiment) & (folds.evaluation_split == split)]
            name = _exp_label(model, experiment)
            fig.add_trace(
                go.Bar(
                    x=[name], y=[fold_rows['map'].mean()],
                    error_y=dict(type='data', array=[fold_rows['map'].std(ddof=1)]), marker_color=COLORS[experiment],
                    name=name, legendgroup=name, showlegend=col == 1),
                row=1,
                col=col)
            fig.add_trace(
                go.Scatter(
                    x=[name] * len(fold_rows), y=fold_rows['map'], mode='markers', marker=dict(color='black', size=6),
                    text=[f'fold {fold}' for fold in fold_rows.fold], hoverinfo='text+y', showlegend=False),
                row=1,
                col=col)
    fig.update_yaxes(title_text='mAP@[.5:.95]', col=1)
    fig.update_layout(title='mAP by experiment (bars: mean ± std; dots: individual folds)', height=520, width=1250,
                      barmode='group')
    figs['map_by_experiment'] = fig
    metrics = [('map_50', 'AP50'), ('map_75', 'AP75'), ('map_small', 'AP_small'), ('map_medium', 'AP_medium'),
               ('map_large', 'AP_large')]
    fig = make_subplots(rows=2, cols=1, subplot_titles=list(split_title.values()), shared_xaxes=True)
    for row, split in enumerate(split_title, 1):
        for model, experiment in exps:
            entry = summary[
                (summary.model == COMPARED_MODELS[model]['label']) & (summary.experiment == experiment)
                & (summary.evaluation_split == split)]
            fig.add_trace(
                go.Bar(
                    x=[metric_label for _, metric_label in metrics],
                    y=[float(entry[metric_key + '_mean'].iloc[0]) for metric_key, _ in metrics],
                    error_y=dict(
                        type='data',
                        array=[float(entry[metric_key + '_std'].iloc[0]) for metric_key, _ in metrics]),
                    marker_color=COLORS[experiment], name=_exp_label(model, experiment), legendgroup=experiment,
                    showlegend=row == 1),
                row=row,
                col=1)
    fig.update_yaxes(title_text='AP')
    fig.update_layout(title='AP50, AP75 and AP by object size (COCO: small < 32², medium < 96² px)', barmode='group',
                      height=800, width=1250)
    figs['ap_breakdown'] = fig
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['Latency, ms per image (lower is better)', 'Throughput, images/s'])
    labels = [_exp_label(model, experiment) for model, experiment in exps]
    for batch_size, pattern in [(1, ''), (4, '/')]:
        stats = [speed[(speed.model == model) & (speed.experiment == experiment) & (speed.batch_size == batch_size)
                       & (speed.precision == 'amp_fp16')] for model, experiment in exps]
        for col, key in enumerate(['ms_per_image', 'fps'], 1):
            fig.add_trace(
                go.Bar(
                    x=labels, y=[batch_rows[key].mean() for batch_rows in stats],
                    error_y=dict(type='data', array=[batch_rows[key].std(ddof=1) for batch_rows in stats]),
                    name=f'batch {batch_size}', legendgroup=f'batch {batch_size}', showlegend=col == 1,
                    marker=dict(
                        color=[COLORS[experiment] for _, experiment in exps],
                        pattern_shape=pattern,
                        line=dict(color='black', width=0.5))),
                row=1,
                col=col)
    fig.update_layout(title='Inference speed · RTX 5070 · AMP fp16 · 200 official-val frames (mean ± std over 5 fold '
                            'checkpoints; solid = batch 1, hatched = batch 4) · excludes decoding and data loading',
                      barmode='group', height=520, width=1250)
    figs['inference_speed'] = fig
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(split_title.values()), shared_yaxes=True)
    keys = [('map', 'mAP'), ('map_50', 'AP50'), ('map_75', 'AP75'), ('map_small', 'AP_small'),
            ('map_medium', 'AP_medium'), ('map_large', 'AP_large'), ('ar', 'AR@100')]
    for col, split in enumerate(split_title, 1):
        for model in ready:
            experiment = selection[model]['best_by_cv']
            entry = summary[
                (summary.model == COMPARED_MODELS[model]['label']) & (summary.experiment == experiment)
                & (summary.evaluation_split == split)]
            fig.add_trace(
                go.Bar(
                    x=[metric_label for _, metric_label in keys],
                    y=[float(entry[metric_key + '_mean'].iloc[0]) for metric_key, _ in keys],
                    error_y=dict(
                        type='data',
                        array=[float(entry[metric_key + '_std'].iloc[0]) for metric_key, _ in keys]),
                    marker_color=MODEL_COLORS[model], name=_exp_label(model, experiment), legendgroup=model,
                    showlegend=col == 1),
                row=1,
                col=col)
    latency = {
        model: speed[(speed.model == model) & (speed.experiment == selection[model]['best_by_cv'])
                     & (speed.batch_size == 1) & (speed.precision == 'amp_fp16')].ms_per_image.mean()
        for model in ready}
    pending = '' if 'rt_detr' in ready else ' · RT-DETR: pending verified results'
    fig.update_layout(title=' vs '.join(COMPARED_MODELS[model]['label'] for model in ready) +
                            ' (experiments selected by CV) · latency batch 1: ' +
                            ', '.join(f"{COMPARED_MODELS[model]['label']} {latency[model]:.1f} ms" for model in ready) +
                            pending, barmode='group', height=520, width=1250)
    figs['faster_rcnn_vs_fcos'] = fig
    fig = go.Figure()
    for model in ready:
        experiment = selection[model]['best_by_cv']
        per_class_ap = pd.DataFrame([load_result(model, experiment, fold)['metrics']['per_class'] for fold in FOLDS])[CLASSES]
        fig.add_trace(
            go.Bar(
                x=CLASSES, y=per_class_ap.mean(), error_y=dict(type='data', array=per_class_ap.std(ddof=1)),
                marker_color=MODEL_COLORS[model], name=_exp_label(model, experiment)))
    fig.update_layout(title='Per-class AP@[.5:.95] · split: CV out-of-fold (mean ± std, 5 folds)', barmode='group',
                      height=480, width=1100, yaxis_title='AP')
    figs['per_class_ap_cv'] = fig
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Detections above the threshold, by outcome',
                                                        'Recall at the threshold, by GT object size'])
    for model in ready:
        experiment, fold = selection[model]['best_by_cv'], selection[model]['representative_fold']
        summary_path = ROOT / model / 'final_evaluation' / 'failure_analysis' / f'failure_summary_{experiment}_fold{fold}.json'
        if not summary_path.exists():
            continue
        entry = json.loads(summary_path.read_text())
        kinds = ['tp', 'background', 'classification', 'localization', 'duplicate', 'missed']
        fig.add_trace(
            go.Bar(
                x=kinds, y=[entry['totals'][kind] for kind in kinds], marker_color=MODEL_COLORS[model],
                name=f"{_exp_label(model, experiment)} fold {fold} (score ≥ {entry['threshold']:.2f})"),
            row=1,
            col=1)
        fig.add_trace(
            go.Bar(
                x=[size_row['size'] for size_row in entry['recall_by_size']],
                y=[size_row['recall'] for size_row in entry['recall_by_size']], marker_color=MODEL_COLORS[model],
                showlegend=False),
            row=1,
            col=2)
    fig.update_layout(title='Error diagnosis · official validation · representative checkpoints (IoU 0.5; '
                            'threshold chosen on each checkpoint\'s CV fold)', barmode='group', height=480, width=1250)
    figs['error_breakdown_official_val'] = fig
    return figs


def _representative_figures(ready, selection):
    """Side-by-side prediction figures of the representative checkpoints on the representative images."""
    panels_for = {}
    for m in ready:
        experiment, fold = selection[m]['best_by_cv'], selection[m]['representative_fold']
        summary_path = ROOT / m / 'final_evaluation' / 'failure_analysis' / f'failure_summary_{experiment}_fold{fold}.json'
        pred_path = _pred_path(m, experiment, fold, 'official_val')
        if summary_path.exists() and pred_path.exists():
            panels_for[m] = (
                experiment, fold, json.loads(summary_path.read_text())['threshold'], load_predictions(pred_path))
    if not panels_for:
        return
    for number, pick in enumerate(representative_images(), 1):
        image_id = pick['image_id']
        panels = [
            (
                f"{COMPARED_MODELS[model]['label']} · {experiment} fold {fold} · score ≥ {threshold:.2f}",
                predictions[image_id],
                threshold)
            for model, (experiment, fold, threshold, predictions) in panels_for.items()]
        fig = prediction_figure(image_id, panels, f"· selected for: {pick['tag'].replace('_', ' ')}")
        save_html(fig, FINAL / 'visualizations' / f"example_{number}_{pick['tag']}.html")
