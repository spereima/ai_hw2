"""
A module for data preparation and evaluation of video inference experiments, used across all inference pipelines.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
HERE = ROOT / 'data_temporality'
sys.path.insert(0, str(ROOT / 'shared'))
sys.path.insert(0, str(HERE))
import temporal_postprocessing

CACHE = HERE / 'cache'
DATA = ROOT / 'datasets/multiple-object-video-detection-hw-2-ucu-ai-2026'
VAL_CACHE = ROOT / 'faster_rcnn/official_validation/cache'
TEST_CACHE = ROOT / 'kaggle_submissions/cache'
FOLDS = range(5)


def pack(rows):
    """Stack per-frame (boxes, scores, labels) rows into flat arrays plus the number of boxes of each frame."""
    return dict(
        counts=np.array([len(row[1]) for row in rows]),
        boxes=np.concatenate([row[0] for row in rows]).reshape(-1, 4).astype(np.float32),
        scores=np.concatenate([row[1] for row in rows]).astype(np.float32),
        labels=np.concatenate([row[2] for row in rows]).astype(np.int64))


def unpack(arrays, prefix=''):
    """Inverse of pack(): split the flat arrays back into per-frame (boxes, scores, labels) rows."""
    counts, boxes, scores, labels = (arrays[prefix + 'counts'], arrays[prefix + 'boxes'],
                                     arrays[prefix + 'scores'], arrays[prefix + 'labels'])
    ends = np.cumsum(counts)
    return [(boxes[i:j], scores[i:j], labels[i:j]) for i, j in zip(ends - counts, ends)]


def save_preds(path, meta, rows):
    tmp_path = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp_path, **pack(rows))
    os.replace(tmp_path, path)
    meta.to_csv(path.with_suffix('.csv'), index=False)


def load_preds(path):
    return pd.read_csv(path.with_suffix('.csv')), unpack(np.load(path))


def frames_meta(name):
    """image_id, video, frame, width, height, path, fold (tune only) in temporal order per video."""
    if name == 'tune':
        segments = pd.read_csv(CACHE / 'tune_segments.csv')
        return (segments.assign(path=[str(DATA / 'train' / file_name) for file_name in segments.file_name])
                [['image_id', 'video', 'frame_id', 'width', 'height', 'path', 'fold']]
                .rename(columns={'frame_id': 'frame'}))
    if name == 'val':
        coco = json.loads((DATA / 'val/annotations_coco.json').read_text())
        images = pd.DataFrame(coco['images'])
        return pd.DataFrame(dict(
            image_id=images.id, video=images.video, frame=images.frame_id, width=images.width, height=images.height,
            path=[str(DATA / 'val' / file_name) for file_name in images.file_name]))
    if name == 'test':
        test_images = pd.read_csv(DATA / 'test_images.csv')
        return pd.DataFrame(dict(
            image_id=test_images.image_id, video=test_images['clip'], frame=test_images['frame'],
            width=test_images.width, height=test_images.height,
            path=[str(DATA / file_name) for file_name in test_images.file_name]))
    raise ValueError(name)


def _fuse(args):
    import warnings
    warnings.filterwarnings('ignore')
    import tta_wbf
    parts, h, w = args
    return tta_wbf.tta_merge(parts, 'wbf', h, w)


def fuse_sets(sets, meta, workers=8):
    """Weighted-box-fuse the prediction sets of every frame; returns one (boxes, scores, labels) row per frame."""
    tasks = (
        ([view_set[frame_index] for view_set in sets], h, w)
        for frame_index, (h, w) in enumerate(zip(meta.height, meta.width)))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_fuse, tasks, chunksize=16))


def view_sets(name, folds):
    """The cached per-view predictions (6 TTA views per checkpoint), aligned with frames_meta(name)."""
    import tta_wbf
    meta, sets = frames_meta(name), []
    for fold in folds:
        for view in tta_wbf.TTA_VIEWS:
            if name == 'val':
                cached = np.load(VAL_CACHE / f'fold{fold}_{view}.npz')
                assert cached['ids'].tolist() == meta.image_id.tolist()
                sets.append(unpack(cached, 'full_'))
            elif name == 'test':
                ids, rows = _test_chunks(fold, view, len(meta))
                assert ids == meta.image_id.tolist()
                sets.append(rows)
    return meta, sets


def _test_chunks(fold, view, num_frames, chunk=400):
    rows, ids = [], []
    for start in range(0, num_frames, chunk):
        cached = np.load(TEST_CACHE / f'fold{fold}' / f'{view}_{start:05d}.npz', allow_pickle=True)
        ids += cached['ids'].tolist()
        rows += unpack(cached)
    return ids, rows


def tune_base():
    """Out-of-fold single-checkpoint TTA + WBF on the tuning frames: each video by the checkpoint of its held-out fold."""
    import tta_wbf
    meta = frames_meta('tune')
    rows = [None] * len(meta)
    for fold in FOLDS:
        pos = np.where(meta.fold.values == fold)[0]
        sets = []
        for view in tta_wbf.TTA_VIEWS:
            cached = np.load(CACHE / 'tune_views' / f'fold{fold}_{view}.npz')
            assert cached['ids'].tolist() == meta.image_id.values[pos].tolist()
            sets.append(unpack(cached))
        for i, row in zip(pos, fuse_sets(sets, meta.iloc[pos])):
            rows[i] = row
    return meta, rows


def _motion(args):
    video, paths = args
    start_time = time.perf_counter()
    motions, inliers = temporal_postprocessing.video_motion(paths)
    return video, motions, inliers, (time.perf_counter() - start_time) / len(paths)


def motion_for(name, workers=8):
    """Camera motion of every frame of set `name` (estimated once, then read from the cache file)."""
    path = CACHE / f'motion_{name}.npz'
    meta = frames_meta(name)
    if not path.exists():
        jobs = [(video, group.path.tolist()) for video, group in meta.groupby('video', sort=False)]
        with ProcessPoolExecutor(workers) as pool:
            results_by_video = {
                video: (video_motions, video_inliers, video_seconds)
                for video, video_motions, video_inliers, video_seconds in pool.map(_motion, jobs)}
        motions = np.concatenate([results_by_video[video][0] for video, _ in jobs])
        inliers = np.concatenate([results_by_video[video][1] for video, _ in jobs])
        seconds_per_frame = np.concatenate([np.full(len(group), results_by_video[video][2]) for video, group in jobs])
        np.savez(
            path, image_id=np.array(meta.image_id.astype(str).tolist(), dtype=str), motions=motions, inliers=inliers,
            seconds_per_frame=seconds_per_frame)
    cached = np.load(path)
    assert cached['image_id'].tolist() == meta.image_id.astype(str).tolist()
    return cached['motions'], cached['inliers'], cached['seconds_per_frame']


def apply_temporal(meta, rows, motions, params):
    """Run the temporal post-processing video by video; returns the new rows in the original order."""
    out = [None] * len(rows)
    for _, group in meta.groupby('video', sort=False):
        idx = group.index.values
        assert (np.diff(group.frame.values) == 1).all(), 'frames must be consecutive'
        new_rows = temporal_postprocessing.temporal_postprocess([rows[i] for i in idx], motions[idx], params)
        for i, row in zip(idx, new_rows):
            out[i] = row
    return out


def annotations_for(name):
    """The annotations of the frames of a set (a small file, so that every worker process does not parse the full train json)."""
    path = CACHE / f'gt_{name}.json'
    if not path.exists():
        coco = json.loads((DATA / ('val' if name == 'val' else 'train') / 'annotations_coco.json').read_text())
        ids = set(frames_meta(name).image_id.astype(int))
        path.write_text(json.dumps(dict(
            images=[image_info for image_info in coco['images'] if int(image_info['id']) in ids],
            annotations=[annotation for annotation in coco['annotations'] if int(annotation['image_id']) in ids],
            categories=coco['categories'])))
    return json.loads(path.read_text())


_WORKER_STATE = {}


def _init(name, preds_file, groups):
    meta, rows = load_preds(Path(preds_file))
    _WORKER_STATE.update(
        meta=meta, rows=rows, motions=motion_for(name)[0],
        gt=temporal_postprocessing.coco_gt(annotations_for(name), meta.image_id.tolist()), groups=groups)


def _evaluate(job):
    key, params = job
    start_time = time.perf_counter()
    rows = _WORKER_STATE['rows'] if params is None else apply_temporal(
        _WORKER_STATE['meta'], _WORKER_STATE['rows'], _WORKER_STATE['motions'], params)
    seconds = (time.perf_counter() - start_time) / len(rows)
    metrics = temporal_postprocessing.coco_metrics(
        _WORKER_STATE['gt'], _WORKER_STATE['meta'].image_id.tolist(), rows, _WORKER_STATE['groups'])
    return key, metrics, seconds, float(np.mean([len(r[1]) for r in rows]))


def evaluate_many(name, preds_file, jobs, groups, workers=6):
    """jobs: list of (key, params or None for the unchanged predictions). Returns {key: (metrics by group, s/frame, boxes/frame)}."""
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(name, str(preds_file), groups)) as pool:
        return {
            job_key: (metrics, seconds, boxes_per_frame)
            for job_key, metrics, seconds, boxes_per_frame in pool.map(_evaluate, jobs)}


def build_caches(log=print):
    """Every prediction set used by the notebook; existing files are reused."""
    jobs = [
        ('val_ens', lambda: fuse_sets(view_sets('val', FOLDS)[1], frames_meta('val')), 'val'),
        *[
            (f'val_fold{fold}', (lambda fold=fold: fuse_sets(view_sets('val', [fold])[1], frames_meta('val'))), 'val')
            for fold in FOLDS],
        ('test_ens', lambda: fuse_sets(view_sets('test', FOLDS)[1], frames_meta('test')), 'test'),
        ('tune_base', lambda: tune_base()[1], 'tune')]
    for cache_name, build, name in jobs:
        path = CACHE / f'{cache_name}.npz'
        if not path.exists():
            start_time = time.time()
            save_preds(path, frames_meta(name).drop(columns='path'), build())
            log(f'{cache_name}: built in {(time.time() - start_time) / 60:.1f} min')
    for name in ['tune', 'val', 'test']:
        if not (CACHE / f'motion_{name}.npz').exists():
            start_time = time.time()
            motion_for(name)
            log(f'motion_{name}: estimated in {(time.time() - start_time) / 60:.1f} min')


def gt_frames(name, meta):
    """Ground-truth (boxes, labels, track ids) of every frame of `meta`, without crowd and empty boxes."""
    coco = annotations_for(name)
    boxes_by_image = {}
    for annotation in coco['annotations']:
        if not annotation.get('iscrowd', 0) and annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0:
            x, y, w, h = annotation['bbox']
            boxes_by_image.setdefault(int(annotation['image_id']), []).append(
                (x, y, x + w, y + h, annotation['category_id'], annotation['track_id']))
    out = []
    for image_id in meta.image_id:
        gt_boxes = np.array(boxes_by_image.get(int(image_id), []), dtype=np.float64).reshape(-1, 6)
        out.append((gt_boxes[:, :4], gt_boxes[:, 4].astype(np.int64), gt_boxes[:, 5].astype(np.int64)))
    return out
