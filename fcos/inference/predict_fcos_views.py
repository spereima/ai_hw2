"""
FCOS prediction stage using TTA (horizontal flipping and resizing)
on three frame sets:

- tune: 56 training videos with 80 consecutive frames each. Every video is processed
  only by the checkpoint of the fold where it was held out for validation (out-of-fold).
- val: official validation set, processed by every checkpoint.
- test: test set, also processed by every checkpoint.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'shared'))
sys.path.insert(0, str(ROOT / 'final_pipeline/temporal'))
import final_evaluation
import experiment_utils

CACHE = HERE / 'cache'
DEV = torch.device('cuda')
BATCH = 2
PERMISSIVE = dict(score_thresh=0.01, detections_per_img=500)
CHUNK = 400


def parse_view(view):
    """'h@1.5' -> ('h', 1.5): horizontal flip ('h' or 'none') and resize factor."""
    flip, scale = view.split('@')
    return flip, float(scale)


def read(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def batches(paths):
    """Yield the frames as tensors in batches of BATCH; up to 3 batches are decoded ahead in threads."""
    path_batches = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(lambda batch_paths: [read(path) for path in batch_paths], batch_paths)
                   for batch_paths in path_batches[:3]]
        for k in range(len(path_batches)):
            arrays = futures[k].result()
            if k + 3 < len(path_batches):
                futures.append(pool.submit(lambda batch_paths: [read(path) for path in batch_paths], path_batches[k + 3]))
            futures[k] = None
            yield [final_evaluation.to_tensor(array) for array in arrays]


@torch.inference_mode()
def view_predictions(net, cfg, paths, view):
    """(boxes, scores, labels) per frame for one TTA view; boxes are mapped back to the original (un-flipped) frame."""
    flip, scale = parse_view(view)
    net.transform.min_size = (round(cfg['min_size'] * scale),)
    net.transform.max_size = round(cfg['max_size'] * scale)
    rows = []
    try:
        for images in batches(paths):
            images = [image.to(DEV, non_blocking=True) for image in images]
            with torch.autocast('cuda'):
                predictions = net([image.flip(-1) if flip == 'h' else image for image in images])
            for image, prediction in zip(images, predictions):
                boxes, scores, labels = prediction['boxes'].float(), prediction['scores'].float(), prediction['labels']
                if flip == 'h':
                    w = image.shape[-1]
                    boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
                keep = ((labels >= 1) & (labels <= 8)
                        & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
                rows.append((boxes[keep].cpu().numpy(), scores[keep].cpu().numpy(),
                             labels[keep].cpu().numpy().astype(np.int64)))
    finally:
        net.transform.min_size = (cfg['min_size'],)
        net.transform.max_size = cfg['max_size']
    return rows


def set_post(net, score_thresh, detections_per_img):
    """Set the post-processing score threshold and detection limit of the FCOS head."""
    net.score_thresh, net.detections_per_img = score_thresh, detections_per_img


def reproduction_check(net, cfg, config_name, fold):
    """Native settings must reproduce the saved official-validation predictions; the permissive run must contain them."""
    image_infos = final_evaluation.coco_json('val')['images'][:8]
    saved = final_evaluation.load_predictions(final_evaluation._pred_path('fcos', config_name, fold, 'official_val'))
    paths = [final_evaluation.DATA / 'val' / image_info['file_name'] for image_info in image_infos]
    set_post(net, cfg['score_thr'], cfg['det_per_img'])
    native = view_predictions(net, cfg, paths, 'none@1.0')
    set_post(net, PERMISSIVE['score_thresh'], PERMISSIVE['detections_per_img'])
    permissive = view_predictions(net, cfg, paths, 'none@1.0')
    for image_info, (boxes, scores, labels), (permissive_boxes, permissive_scores, permissive_labels) in zip(
            image_infos, native, permissive):
        reference = saved[int(image_info['id'])]
        valid = ((reference['labels'] >= 1) & (reference['labels'] <= 8)
                 & (reference['boxes'][:, 2] > reference['boxes'][:, 0])
                 & (reference['boxes'][:, 3] > reference['boxes'][:, 1]))
        assert (np.allclose(boxes, reference['boxes'][valid], atol=1e-3)
                and np.allclose(scores, reference['scores'][valid], atol=1e-5)
                and (labels == reference['labels'][valid]).all()), \
            f'{config_name} fold {fold}: native settings do not reproduce the saved predictions'
        above_threshold = permissive_scores >= cfg['score_thr']
        top = np.argsort(-permissive_scores[above_threshold], kind='stable')[:cfg['det_per_img']]
        assert np.allclose(np.sort(permissive_scores[above_threshold][top]), np.sort(scores)) and len(top) == len(scores), \
            f'{config_name} fold {fold}: permissive run is not a superset'


def jobs_for(name, folds):
    """{fold: frames to predict with that fold's checkpoint}; for 'tune' only the videos held out by the fold."""
    meta = experiment_utils.frames_meta(name)
    if name == 'tune':
        return {fold: meta[meta.fold == fold] for fold in folds}
    return {fold: meta for fold in folds}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True, choices=['tune', 'val', 'test'])
    parser.add_argument('--config', required=True, choices=['F0_baseline', 'F1_ciou', 'F2_radius_1_0'])
    parser.add_argument('--views', nargs='+', required=True)
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    started = time.time()
    for fold, meta in jobs_for(args.set, args.folds).items():
        folder = CACHE / args.set / args.config / f'fold{fold}'
        folder.mkdir(parents=True, exist_ok=True)
        n_frames, paths, ids = len(meta), meta.path.tolist(), meta.image_id.astype(str).tolist()
        todo = [(view, start) for view in args.views for start in range(0, n_frames, CHUNK)
                if not (folder / f'{view}_{start:05d}.npz').exists()]
        if not todo:
            log(f'{args.set} {args.config} fold {fold}: cached')
            continue
        net, cfg = final_evaluation.load_checkpoint('fcos', args.config, fold)
        reproduction_check(net, cfg, args.config, fold)
        set_post(net, PERMISSIVE['score_thresh'], PERMISSIVE['detections_per_img'])
        log(f'{args.set} {args.config} fold {fold}: reproduction check passed | '
            f'{n_frames} frames | {len(todo)} chunks to predict')
        for view, start in todo:
            chunk_started = time.time()
            rows = view_predictions(net, cfg, paths[start:start + CHUNK], view)
            path = folder / f'{view}_{start:05d}.npz'
            tmp_path = path.with_name(path.stem + '.tmp.npz')
            np.savez(tmp_path, ids=np.array(ids[start:start + CHUNK], dtype=str), **experiment_utils.pack(rows))
            os.replace(tmp_path, path)
            log(f'{args.set} {args.config} fold {fold} {view} frames {start + len(rows)}/{n_frames} | '
                f'{1000 * (time.time() - chunk_started) / len(rows):.0f} ms/frame '
                f'| elapsed {(time.time() - started) / 60:.1f} min')
        del net
        torch.cuda.empty_cache()
    log(f'done in {(time.time() - started) / 60:.1f} min')
