import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ensemble_boxes import weighted_boxes_fusion
from importlib.metadata import version

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
DATA = ROOT / 'datasets/multiple-object-video-detection-hw-2-ucu-ai-2026'
RUNS = ROOT / 'faster_rcnn/lr_scheduler'
FOLDS_CSV = ROOT / 'shared/sequence_folds.csv'
VAL_CACHE = ROOT / 'faster_rcnn/official_validation/cache'
OUT = ROOT / 'kaggle_submissions'
CACHE = OUT / 'cache'
TTA_VIEWS = ['none@1.0', 'h@1.0', 'none@1.25', 'h@1.25', 'none@1.5', 'h@1.5']
TTA_BATCH = {1.0: 4, 1.25: 4, 1.5: 2}
WBF_IOU = 0.55
DET_PER_IMG = 500
CHUNK = 400
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEV.type == 'cuda'
KAGGLE_CLASSES = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']

categories = json.loads((DATA / 'val/annotations_coco.json').read_text())['categories']
CAT_IDS = sorted(int(category['id']) for category in categories)
LABEL_TO_CAT = {i + 1: category_id for i, category_id in enumerate(CAT_IDS)}
N_CLASSES = len(CAT_IDS) + 1
assert ([category['name'] for category in sorted(categories, key=lambda category: category['id'])] == KAGGLE_CLASSES
        and CAT_IDS == list(range(1, 9)))
FOLDS_SHA = hashlib.sha256(FOLDS_CSV.read_bytes()).hexdigest()[:12]
INFERENCE_KEYS = ['min_size', 'max_size', 'rpn_pre_test', 'rpn_post_test', 'det_per_img', 'score_thr', 'nms_thr']
INFERENCE = {key: json.loads((RUNS / 'fold0/result.json').read_text())['cfg'][key] for key in INFERENCE_KEYS}
assert all(
    {key: json.loads((RUNS / f'fold{fold}/result.json').read_text())['cfg'][key] for key in INFERENCE_KEYS} == INFERENCE
    for fold in range(5))


def cfg_id(cfg):
    """Hash of a training config (without `save_every`), as stored in best.pth."""
    stable = {k: v for k, v in cfg.items() if k != 'save_every'}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def build_model(cfg):
    """Faster R-CNN (ResNet-50 FPN) with the settings of a training config and a new box predictor."""
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=cfg['trainable_layers'],
        min_size=cfg['min_size'], max_size=cfg['max_size'],
        rpn_pre_nms_top_n_train=cfg['rpn_pre_train'], rpn_post_nms_top_n_train=cfg['rpn_post_train'],
        rpn_pre_nms_top_n_test=cfg['rpn_pre_test'], rpn_post_nms_top_n_test=cfg['rpn_post_test'],
        rpn_batch_size_per_image=cfg['rpn_batch'],
        box_batch_size_per_image=cfg['roi_batch'], box_positive_fraction=cfg['roi_pos_frac'],
        box_detections_per_img=cfg['det_per_img'], box_score_thresh=cfg['score_thr'],
        box_nms_thresh=cfg['nms_thr'])
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, N_CLASSES)
    return model


def load_model(fold):
    """Load best.pth of one fold (checked against its config); returns (model in eval mode, config, checkpoint data)."""
    folder = RUNS / f'fold{fold}'
    cfg = json.loads((folder / 'config.json').read_text())
    checkpoint_data = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
    assert checkpoint_data['cfg_id'] == cfg_id(cfg) and cfg['fold'] == fold and cfg['folds'] == FOLDS_SHA
    model = build_model(cfg)
    model.load_state_dict(checkpoint_data['model'])
    return model.to(DEV).eval(), cfg, checkpoint_data


def sha256(path):
    """SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_images(paths):
    """Decode image files to RGB uint8 arrays."""
    arrays = []
    for path in paths:
        with Image.open(path) as picture:
            arrays.append(np.array(picture.convert('RGB')))
    return arrays


def to_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255)


def batches(paths, batch_size):
    """Yield (paths, image tensors) per batch; the next batch is decoded in a background thread."""
    chunks = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(read_images, chunks[0])
        for chunk_index, chunk in enumerate(chunks):
            arrays = pending.result()
            if chunk_index + 1 < len(chunks):
                pending = pool.submit(read_images, chunks[chunk_index + 1])
            yield chunk, [to_tensor(array) for array in arrays]


def set_scale(model, scale):
    """Resize the model input by `scale` (1.0 = the size used in training)."""
    model.transform.min_size = (round(INFERENCE['min_size'] * scale),)
    model.transform.max_size = round(INFERENCE['max_size'] * scale)


def parse_view(view):
    """Split a TTA view name such as 'h@1.25' into (flip, scale)."""
    flip, scale = view.split('@')
    return flip, float(scale)


def flip_image(image, flip):
    """Flip a CHW image horizontally ('h'), vertically ('v') or not at all."""
    return image.flip(-1) if flip == 'h' else image.flip(-2) if flip == 'v' else image


def unflip(boxes, flip, h, w):
    """Map xyxy boxes predicted on a flipped image back to the original image."""
    x1, y1, x2, y2 = boxes.unbind(1)
    if flip == 'h':
        return torch.stack([w - x2, y1, w - x1, y2], 1)
    if flip == 'v':
        return torch.stack([x1, h - y2, x2, h - y1], 1)
    return boxes


def tta_merge(parts, method, h, w):
    if len(parts) == 1:
        return parts[0]
    if method == 'nms':
        boxes, scores, labels = (np.concatenate(field_arrays) for field_arrays in zip(*parts))
        keep = box_ops.batched_nms(
            torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(labels), 0.5).numpy()
    else:
        size = np.array([w, h, w, h], dtype=np.float32)
        boxes, scores, labels = weighted_boxes_fusion(
            [part[0] / size for part in parts], [part[1] for part in parts], [part[2] for part in parts],
            iou_thr=WBF_IOU, skip_box_thr=0.0, conf_type='avg')
        boxes, scores, labels = (boxes * size).astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)
        keep = np.argsort(-scores, kind='stable')
    keep = keep[:DET_PER_IMG]
    return boxes[keep], scores[keep], labels[keep]


def to_numpy(boxes, scores, labels):
    """Move the three prediction tensors to numpy arrays."""
    return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


@torch.inference_mode()
def view_predictions(model, paths, view):
    flip, scale = parse_view(view)
    set_scale(model, scale)
    rows = []
    try:
        for chunk_paths, images in batches(paths, TTA_BATCH[scale]):
            images = [image.to(DEV, non_blocking=True) for image in images]
            with torch.autocast(device_type=DEV.type, enabled=AMP):
                predictions = model([flip_image(image, flip) for image in images])
            rows += [
                to_numpy(
                    unflip(prediction['boxes'].float(), flip, *image.shape[-2:]),
                    prediction['scores'].float(),
                    prediction['labels'])
                for image, prediction in zip(images, predictions)]
    finally:
        set_scale(model, 1.0)
    return rows


def same_arrays(arrays_a, arrays_b):
    """True if two (boxes, scores, labels) tuples are exactly equal."""
    return all(np.array_equal(array_a, array_b) for array_a, array_b in zip(arrays_a, arrays_b))


def unpack(cached, prefix):
    """Split the flat cached arrays '<prefix>_boxes' etc. back into per-frame rows."""
    ends = np.cumsum(cached[prefix + '_counts'])
    boxes, scores, labels = cached[prefix + '_boxes'], cached[prefix + '_scores'], cached[prefix + '_labels']
    return [(boxes[i:j], scores[i:j], labels[i:j]) for i, j in zip(ends - cached[prefix + '_counts'], ends)]


def pack(rows):
    """Stack per-frame (boxes, scores, labels) rows into flat arrays plus the number of boxes per frame."""
    return dict(
        counts=np.array([len(row[1]) for row in rows]),
        boxes=np.concatenate([row[0] for row in rows]).reshape(-1, 4).astype(np.float32),
        scores=np.concatenate([row[1] for row in rows]).astype(np.float32),
        labels=np.concatenate([row[2] for row in rows]).astype(np.int64))


def prediction_string(boxes, scores, labels, h, w):
    parts = []
    for (x1, y1, x2, y2), score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1, x2 = round(min(max(x1, 0.), w), 2), round(min(max(x2, 0.), w), 2)
        y1, y2 = round(min(max(y1, 0.), h), 2), round(min(max(y2, 0.), h), 2)
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            continue
        parts.append(f'{LABEL_TO_CAT[int(label)] - 1} {score:.6f} {x1:.2f} {y1:.2f} {x2 - x1:.2f} {y2 - y1:.2f}')
    return ' '.join(parts)


def validate_submission(path, images):
    """Check the submission csv against the competition format; returns (checks, statistics)."""
    submission = pd.read_csv(path, dtype=str, keep_default_na=False)
    checks = {}
    checks['columns are exactly image_id, PredictionString'] = (
        list(submission.columns) == ['image_id', 'PredictionString'])
    checks['one row per test image, same order as test_images.csv'] = (
        submission.image_id.tolist() == images.image_id.tolist())
    checks['no duplicate image ids'] = submission.image_id.is_unique
    checks['no missing image ids'] = set(submission.image_id) == set(images.image_id)
    sizes = dict(zip(images.image_id, zip(images.width, images.height)))
    bad, counts, classes, scores = [], [], set(), []
    for image_id, text in zip(submission.image_id, submission.PredictionString):
        tokens = text.split()
        if len(tokens) % 6:
            bad.append((image_id, 'not a multiple of 6 values'))
            continue
        w, h = sizes[image_id]
        groups = [tokens[i:i + 6] for i in range(0, len(tokens), 6)]
        counts.append(len(groups))
        for class_id, score, x, y, box_width, box_height in groups:
            if not class_id.isdigit() or not 0 <= int(class_id) <= 7:
                bad.append((image_id, f'class {class_id}'))
                continue
            classes.add(int(class_id))
            values = np.array([score, x, y, box_width, box_height], dtype=np.float64)
            if not np.isfinite(values).all():
                bad.append((image_id, 'non-finite value'))
            elif not 0 < values[0] <= 1:
                bad.append((image_id, f'score {score}'))
            elif (values[1] < 0 or values[2] < 0 or values[3] <= 0 or values[4] <= 0
                  or values[1] + values[3] > w + 1e-6 or values[2] + values[4] > h + 1e-6):
                bad.append((image_id, f'box {x} {y} {box_width} {box_height} outside {w}x{h}'))
            scores.append(values[0])
    checks['every prediction is "class score x y w h" with class 0-7, score in (0, 1], '
           'finite box inside the image'] = not bad
    checks['at most 500 detections per image'] = max(counts, default=0) <= DET_PER_IMG
    checks['empty PredictionString is an empty field (no NaN/None text)'] = not any(
        text.strip() in ('nan', 'None', 'NaN') for text in submission.PredictionString)
    stats = dict(
        rows=len(submission), images_with_detections=int(sum(count > 0 for count in counts)),
        images_without_detections=int((submission.PredictionString == '').sum()), detections=int(sum(counts)),
        max_per_image=int(max(counts, default=0)), mean_per_image=float(np.mean(counts)) if counts else 0.,
        classes_present=sorted(classes), min_score=float(min(scores)) if scores else None, first_errors=bad[:5])
    return checks, stats


def log(msg):
    """Print a message with the current time."""
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def load_chunks(fold, view, num_frames):
    """Read the cached chunk files of one fold and view; returns (image ids, per-frame rows)."""
    rows, ids = [], []
    for start in range(0, num_frames, CHUNK):
        cached = np.load(CACHE / f'fold{fold}' / f'{view}_{start:05d}.npz', allow_pickle=True)
        ids += cached['ids'].tolist()
        counts, boxes, scores, labels = cached['counts'], cached['boxes'], cached['scores'], cached['labels']
        ends = np.cumsum(counts)
        rows += [(boxes[i:j], scores[i:j], labels[i:j]) for i, j in zip(ends - counts, ends)]
    return ids, rows


def fuse_frame(args):
    parts, h, w = args
    boxes, scores, labels = tta_merge(parts, 'wbf', h, w)
    assert (len(scores) <= DET_PER_IMG and np.isfinite(boxes).all() and np.isfinite(scores).all()
            and set(labels.tolist()) <= set(range(1, 9)))
    return prediction_string(boxes, scores, labels, h, w)


def run_checkpoint(fold, images, paths, started):
    checkpoint_sha256 = sha256(RUNS / f'fold{fold}/best.pth')
    folder = CACHE / f'fold{fold}'
    num_frames = len(paths)
    if all(
        (folder / f'{view}_{start:05d}.npz').exists()
        for view in TTA_VIEWS
        for start in range(0, num_frames, CHUNK)):
        log(f'fold {fold}: all test views cached, no inference needed')
        return checkpoint_sha256
    model, cfg, checkpoint_data = load_model(fold)
    log(f'checkpoint lr_scheduler/fold{fold}/best.pth (epoch {checkpoint_data["epoch"]}, '
        f'sha256 {checkpoint_sha256[:12]})')
    coco_val = json.loads((DATA / 'val/annotations_coco.json').read_text())
    val_paths = [DATA / 'val' / image_info['file_name'] for image_info in coco_val['images'][:8]]
    assert all(
        json.loads((VAL_CACHE / f'fold{fold}_{view}.json').read_text())['checkpoint_sha256'] == checkpoint_sha256
        for view in TTA_VIEWS)
    new_views = {view: view_predictions(model, val_paths, view) for view in TTA_VIEWS}
    saved_views = {view: unpack(np.load(VAL_CACHE / f'fold{fold}_{view}.npz'), 'full')[:8] for view in TTA_VIEWS}
    assert all(same_arrays(new_rows, saved_rows)
               for view in TTA_VIEWS
               for new_rows, saved_rows in zip(new_views[view], saved_views[view])
               ), 'views differ from the saved official-validation predictions'
    log(f'fold {fold}: reproduction check passed '
        '(six views equal the saved official-validation predictions, 8 frames)')
    folder.mkdir(parents=True, exist_ok=True)
    for view in TTA_VIEWS:
        for start in range(0, num_frames, CHUNK):
            path = folder / f'{view}_{start:05d}.npz'
            if path.exists():
                continue
            start_time = time.time()
            rows = view_predictions(model, paths[start:start + CHUNK], view)
            tmp_path = path.with_name(path.stem + '.tmp.npz')
            np.savez(tmp_path, ids=np.array(images.image_id.tolist()[start:start + CHUNK], dtype=str), **pack(rows))
            os.replace(tmp_path, path)
            log(f'fold {fold} {view}: frames {start + len(rows)}/{num_frames} | chunk {time.time() - start_time:.0f} s | '
                f'elapsed {(time.time() - started) / 60:.1f} min | saved {path.name}')
        log(f'fold {fold} view {view} complete')
    del model
    torch.cuda.empty_cache() if DEV.type == 'cuda' else None
    return checkpoint_sha256


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=[2], help='checkpoints lr_scheduler/fold<k>/best.pth')
    parser.add_argument('--output', default='faster_rcnn_submission.csv')
    parser.add_argument('--workers', type=int, default=8, help='CPU processes for WBF')
    args = parser.parse_args()
    folds = sorted(set(args.folds))
    SUBMISSION = OUT / args.output
    started = time.time()
    images = pd.read_csv(DATA / 'test_images.csv')
    assert images.image_id.is_unique
    paths = [DATA / file_name for file_name in images.file_name]
    assert all(path.exists() for path in paths), 'missing test images'
    num_frames = len(images)
    log(f'test images: {num_frames} in {images["clip"].nunique()} clips (from test_images.csv) | checkpoints {folds} | device: '
        f'{torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu"} | torch {torch.__version__}, '
        f'torchvision {torchvision.__version__}, '
        f'ensemble-boxes {version("ensemble-boxes")} | inference {INFERENCE}')
    cv_map = {fold: json.loads((RUNS / f'fold{fold}/result.json').read_text())['metrics']['mAP'] for fold in range(5)}
    shas = {fold: run_checkpoint(fold, images, paths, started) for fold in folds}
    inference_min = (time.time() - started) / 60

    sets = []
    for fold in folds:
        for view in TTA_VIEWS:
            ids, rows = load_chunks(fold, view, num_frames)
            assert ids == images.image_id.tolist(), (fold, view)
            sets.append(rows)
    log(f'fusing {len(sets)} prediction sets per frame ({len(folds)} checkpoints x {len(TTA_VIEWS)} views) '
        f'with one WBF')
    tasks = (
        ([view_set[frame_index] for view_set in sets], h, w)
        for frame_index, (h, w) in enumerate(zip(images.height, images.width)))
    strings = []
    with ProcessPoolExecutor(args.workers) as pool:
        for frame_index, text in enumerate(pool.map(fuse_frame, tasks, chunksize=16)):
            strings.append(text)
            if (frame_index + 1) % 500 == 0 or frame_index + 1 == num_frames:
                log(f'WBF + formatting: {frame_index + 1}/{num_frames} | '
                    f'elapsed {(time.time() - started) / 60:.1f} min')
    tmp_path = SUBMISSION.with_suffix('.tmp.csv')
    pd.DataFrame({'image_id': images.image_id, 'PredictionString': strings}).to_csv(tmp_path, index=False)
    os.replace(tmp_path, SUBMISSION)

    checks, stats = validate_submission(SUBMISSION, images)
    for name, ok in checks.items():
        log(f"{'PASS' if ok else 'FAIL'}: {name}")
    log(f'statistics: {stats}')
    total_min = (time.time() - started) / 60
    report = dict(
        submission=str(SUBMISSION),
        checkpoints={fold: dict(path=f'lr_scheduler/fold{fold}/best.pth', sha256=shas[fold]) for fold in folds},
        cv_map_by_fold=cv_map, tta_views=TTA_VIEWS,
        tta_batch={str(scale): batch_size for scale, batch_size in TTA_BATCH.items()},
        fusion=f'one class-aware WBF over {len(sets)} prediction sets per frame (checkpoint x view), equal weights',
        wbf=dict(iou_thr=WBF_IOU, skip_box_thr=0.0, conf_type='avg', weights='equal', top=DET_PER_IMG),
        inference=INFERENCE, amp=AMP, device=torch.cuda.get_device_name() if DEV.type == 'cuda' else 'cpu',
        versions=dict(
            torch=torch.__version__, torchvision=torchvision.__version__, ensemble_boxes=version('ensemble-boxes'),
            python=sys.version.split()[0]),
        test_images=num_frames, checks=checks, stats=stats, inference_minutes=inference_min, total_minutes=total_min)
    (OUT / (Path(args.output).stem + '_report.json')).write_text(json.dumps(report, indent=2, default=str))
    log(f'total {total_min:.1f} min | {"ALL CHECKS PASSED" if all(checks.values()) else "CHECKS FAILED"} | {SUBMISSION}')
    sys.exit(0 if all(checks.values()) else 1)
