"""
Shared module for combining predictions and evaluating the inference pipelines of FCOS, Faster R-CNN v2, RetinaNet, and RT-DETR.
Evaluation is performed on three datasets: `tune` (held-out training frames), `val` (official validation), and `test`.
Performance is measured using COCO metrics.
"""
import os
import sys
import time
import experiment_utils as eu
import temporal_postprocessing as tp
from ensemble_boxes import weighted_boxes_fusion
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
HERE = ROOT / 'fcos' / 'inference'
TEMPORAL = ROOT / 'data_temporality'
for directory in (ROOT / 'shared', TEMPORAL, HERE):
    sys.path.insert(0, str(directory))

CACHE = HERE / 'cache'
FOLDS = [0, 1, 2, 3, 4]
CHUNK = 400
FASTER_RCNN_TEMPORAL = dict(s_high=0.3, s_low=0.01, iou_link=0.5, max_gap=2, min_len=3, alpha=0.65, beta=0.6, gamma=0.75)


def _chunks(folder, view, ids):
    """Read the cached predictions of one view (chunk files of CHUNK frames) and check they match `ids`."""
    rows, loaded_ids = [], []
    for start in range(0, len(ids), CHUNK):
        cached = np.load(folder / f'{view}_{start:05d}.npz')
        loaded_ids += cached['ids'].tolist()
        rows += eu.unpack(cached)
    assert loaded_ids == ids, (folder, view)
    return rows


def fcos_sets(name, configs, views):
    """{(config, fold, view): rows aligned with frames_meta(name)}; on tune fold = 'oof' (each video by its held-out checkpoint)."""
    out = {}
    for config in ([configs] if isinstance(configs, str) else configs):
        out.update({(config, *k): v for k, v in _fcos_sets(name, config, views).items()})
    return out


def _fcos_sets(name, config, views):
    meta = eu.frames_meta(name)
    out = {}
    for fold in FOLDS:
        folder = CACHE / name / config / f'fold{fold}'
        if name == 'tune':
            pos = np.where(meta.fold.values == fold)[0]
            ids = meta.image_id.astype(str).values[pos].tolist()
            for view in views:
                rows = _chunks(folder, view, ids)
                full = out.setdefault(('oof', view), [None] * len(meta))
                for i, row in zip(pos, rows):
                    full[i] = row
        else:
            for view in views:
                out[(fold, view)] = _chunks(folder, view, meta.image_id.astype(str).tolist())
    return out


def frcnn_fused(name, source=True):
    """Faster R-CNN TTA + WBF exactly as submitted: tune = out-of-fold single checkpoint, val/test = 5-checkpoint ensemble.
    source True = Faster R-CNN (v1) caches; a folder path = the same files built from other checkpoints (final_pipeline/ensemble)."""
    cache_name = {'tune': 'tune_base', 'val': 'val_ens', 'test': 'test_ens'}[name]
    folder = eu.CACHE if source is True else Path(source)
    meta, rows = eu.load_preds(folder / f'{cache_name}.npz')
    assert meta.image_id.astype(str).tolist() == eu.frames_meta(name).image_id.astype(str).tolist()
    return rows


def wbf(parts, h, w, iou, conf_type='avg', weights=None, top=500):
    """Class-aware weighted box fusion of several (boxes, scores, labels) sets of one frame; keeps `top` boxes."""
    parts = [part for part in parts if part is not None]
    size = np.array([w, h, w, h], dtype=np.float32)
    if sum(len(part[1]) for part in parts) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    boxes, scores, labels = weighted_boxes_fusion(
        [np.clip(part[0] / size, 0, 1) for part in parts], [part[1] for part in parts], [part[2] for part in parts],
        weights=weights, iou_thr=iou, skip_box_thr=0.0, conf_type=conf_type)
    order = np.argsort(-scores, kind='stable')[:top]
    return (boxes[order] * size).astype(np.float32), scores[order].astype(np.float32), labels[order].astype(np.int64)


def filt(prediction, score_min, pre_nms=None):
    """Drop boxes below `score_min`, then optionally class-wise NMS at IoU `pre_nms`."""
    confident = prediction[1] >= score_min
    boxes, scores, labels = prediction[0][confident], prediction[1][confident], prediction[2][confident]
    if pre_nms and len(scores):
        import torch
        from torchvision.ops import batched_nms
        keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(labels), pre_nms).numpy()
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    return boxes, scores, labels


def fuse_frame(fcos_parts, frcnn_part, h, w, job):
    """Fuse the FCOS sets and the Faster R-CNN set of one frame as described by `job`."""
    fcos_settings = job.get('fcos')
    out = None
    if fcos_settings and fcos_settings.get('file'):
        boxes, scores, labels = fcos_parts[0]
        out = (
            boxes,
            (scores ** fcos_settings['power']).astype(np.float32) if fcos_settings.get('power', 1.0) != 1.0 else scores,
            labels)
    elif fcos_settings:
        parts = [filt(part, fcos_settings.get('score_min', 0.01), fcos_settings.get('pre_nms')) for part in fcos_parts]
        boxes, scores, labels = parts[0] if len(parts) == 1 else wbf(
            parts, h, w, fcos_settings.get('wbf_iou', 0.55), fcos_settings.get('conf_type', 'avg'), top=job.get('top', 500))
        if fcos_settings.get('power', 1.0) != 1.0:
            scores = (scores ** fcos_settings['power']).astype(np.float32)
        out = (boxes, scores, labels)
    if job.get('frcnn'):
        if out is None:
            out = frcnn_part
        else:
            fusion_settings = job['fusion']
            out = wbf(
                [out, frcnn_part], h, w, fusion_settings['iou'], fusion_settings.get('conf_type', 'avg'),
                weights=[fusion_settings['weight_fcos'], 1.0], top=job.get('top', 500))
    boxes, scores, labels = out
    order = np.argsort(-scores, kind='stable')[:job.get('top', 500)]
    return boxes[order], scores[order], labels[order]

_WORKER_STATE = {}


def _init(name, config, views, with_frcnn):
    """Worker-process initialiser: load the frame list, cached predictions and ground truth once per worker."""
    import warnings
    warnings.filterwarnings('ignore')
    meta = eu.frames_meta(name)
    _WORKER_STATE.update(
        name=name, meta=meta.drop(columns='path'), sets=fcos_sets(name, config, views) if config else {},
        frcnn=frcnn_fused(name, with_frcnn) if with_frcnn else None, motions=eu.motion_for(name)[0])
    if name != 'test':
        _WORKER_STATE['gt'] = tp.coco_gt(eu.annotations_for(name), meta.image_id.tolist())


def _groups(name, meta):
    """Image ids per evaluation group: all frames, plus one group per fold (tune) or per video (val)."""
    groups = {'all': meta.image_id.tolist()}
    if name == 'tune':
        groups.update({f'fold{fold}': meta.image_id[meta.fold == fold].tolist() for fold in FOLDS})
    else:
        groups.update({f'video_{video}': meta.image_id[meta.video == video].tolist() for video in meta.video.unique()})
    return groups


def run_pipeline(job, meta=None, sets=None, frcnn=None, motions=None):
    """Rows (boxes, scores, labels) of every frame after fusion and optional temporal post-processing."""
    meta = _WORKER_STATE['meta'] if meta is None else meta
    sets = _WORKER_STATE['sets'] if sets is None else sets
    frcnn = _WORKER_STATE['frcnn'] if frcnn is None else frcnn
    motions = _WORKER_STATE['motions'] if motions is None else motions
    fcos_settings = job.get('fcos')
    keys = []
    if fcos_settings and fcos_settings.get('file'):
        if fcos_settings['file'] not in _WORKER_STATE.setdefault('files', {}):
            _WORKER_STATE['files'][fcos_settings['file']] = eu.load_preds(Path(fcos_settings['file']))[1]
        sets = {'file': (_WORKER_STATE['files'][fcos_settings['file']] if 'sub' not in _WORKER_STATE
                         else [_WORKER_STATE['files'][fcos_settings['file']][i] for i in _WORKER_STATE['sub']])}
        keys = ['file']
    elif fcos_settings:
        configs = fcos_settings.get('configs', [fcos_settings.get('config')])
        keys = [
            (config, fold, view)
            for (config, fold, view) in sets
            if config in configs and view in fcos_settings['views']
            and (fcos_settings.get('folds') is None or fold in fcos_settings['folds'] or fold == 'oof')]
    rows = []
    for i, (h, w) in enumerate(zip(meta.height.values, meta.width.values)):
        rows.append(fuse_frame([sets[key][i] for key in keys], frcnn[i] if frcnn is not None else None, h, w, job))
    if job.get('temporal'):
        rows = eu.apply_temporal(
            meta.reset_index(drop=True), rows, motions, {**job['temporal'], 'max_dets': job.get('top', 500)})
    return rows


def _evaluate(item):
    key, job = item
    start_time = time.perf_counter()
    rows = run_pipeline(job)
    seconds = (time.perf_counter() - start_time) / len(rows)
    metrics = tp.coco_metrics(
        _WORKER_STATE['gt'], _WORKER_STATE['meta'].image_id.tolist(), rows,
        _groups(_WORKER_STATE['name'], _WORKER_STATE['meta']), job.get('max_dets_eval', 100))
    return key, metrics, seconds


def evaluate_jobs(name, jobs, config=None, views=(), with_frcnn=False, workers=6):
    """jobs: {key: job}. Returns {key: (metrics by group, CPU seconds per frame of the fusion/temporal step)}."""
    with ProcessPoolExecutor(min(workers, len(jobs)), initializer=_init,
                             initargs=(name, config, list(views), with_frcnn)) as pool:
        return {key: (metrics, seconds) for key, metrics, seconds in pool.map(_evaluate, list(jobs.items()))}


def table(results, groups=('all',)):
    """One row per pipeline: its key, the metrics of all frames, per-fold or per-video mAP and ms per frame."""
    rows = []
    for key, (metrics, seconds) in results.items():
        row = dict(key=key, **{k: v for k, v in metrics['all'].items()}, ms_per_frame=1000 * seconds)
        for group in metrics:
            if group != 'all':
                row[f'mAP_{group}'] = metrics[group]['mAP']
        rows.append(row)
    return pd.DataFrame(rows)


def _rows_video(args):
    job, idx = args
    video_meta = _WORKER_STATE['meta'].iloc[idx]
    sets = {key: [rows[i] for i in idx] for key, rows in _WORKER_STATE['sets'].items()}
    frcnn = [_WORKER_STATE['frcnn'][i] for i in idx] if _WORKER_STATE['frcnn'] is not None else None
    _WORKER_STATE['sub'] = idx
    try:
        return idx, run_pipeline(job, video_meta, sets, frcnn, _WORKER_STATE['motions'][idx])
    finally:
        _WORKER_STATE.pop('sub')


def pipeline_rows(name, job, config, views, with_frcnn, workers=8):
    """Run one pipeline over a whole set, in parallel over videos; returns (frame list, rows)."""
    meta = eu.frames_meta(name)
    groups = [video.index.values for _, video in meta.groupby('video', sort=False)]
    out = [None] * len(meta)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(name, config, list(views), with_frcnn)) as pool:
        for idx, rows in pool.map(_rows_video, [(job, video) for video in groups]):
            for i, row in zip(idx, rows):
                out[i] = row
    return meta, out


def _fuse_one(args):
    import warnings
    warnings.filterwarnings('ignore')
    parts, h, w, job = args
    return fuse_frame(parts, None, h, w, job)


def precompute_fcos(name, fjob, workers=8):
    """Fuse all FCOS checkpoints and views of a set once and cache the result; returns the cache file path."""
    import hashlib
    import json
    key = hashlib.sha256(json.dumps(fjob, sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f'fcos_fused_{name}_{key}.npz'
    if path.exists():
        return str(path)
    start_time = time.time()
    meta = eu.frames_meta(name)
    sets = fcos_sets(name, fjob['configs'], fjob['views'])
    keys = [k for k in sets if (fjob.get('folds') is None or k[1] in fjob['folds'])]
    job = dict(fcos={k: v for k, v in fjob.items() if k != 'power'})
    tasks = (([sets[k][i] for k in keys], h, w, job) for i, (h, w) in enumerate(zip(meta.height, meta.width)))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_fuse_one, tasks, chunksize=8))
    eu.save_preds(path, meta.drop(columns='path'), rows)
    (path.with_suffix('.json')).write_text(json.dumps(dict(
        job=fjob, sets=len(keys), minutes=(time.time() - start_time) / 60,
        seconds_per_frame=(time.time() - start_time) * workers / len(meta), workers=workers)))
    print(f'{path.name}: {len(keys)} sets per frame fused in {(time.time() - start_time) / 60:.1f} min')
    return str(path)
