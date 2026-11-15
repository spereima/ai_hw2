"""
Comparison of the original and fine-tuned Faster R-CNN on one fold
"""
import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'shared'))
import tta_wbf as kaggle_submission

ROOT, DATA_DIR, DEVICE, USE_AMP = kaggle_submission.ROOT, kaggle_submission.DATA, kaggle_submission.DEV, kaggle_submission.AMP
FINETUNED_DIR = ROOT / 'rejected_experiments/faster_rcnn_high_resolution/hires_ms_ft'
CV_CACHE = ROOT / 'faster_rcnn/inference_experiments/tta/views'
VIEWS = kaggle_submission.TTA_VIEWS
SINGLE = {'single 720': 'none@1.0', 'single 900': 'none@1.25', 'single 1080': 'none@1.5'}
CONFIGS = list(SINGLE) + ['TTA + WBF']
KEYS = ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']
GROUPS = {'756-765': (700, 800), '1071-1080': (1000, 1100), '1512-1530': (1500, 1600), '2160': (2100, 2200)}
SCORE_THRESHOLD = 0.3
SEED = 0

def log(msg):
    """Print a message prefixed with the current time."""
    print(time.strftime('%H:%M:%S'), msg, flush=True)

def pack(rows):
    """Concatenate per-frame (boxes, scores, labels) rows into flat arrays plus the number of detections per frame."""
    return dict(counts=np.array([len(row[1]) for row in rows]),
                boxes=np.concatenate([row[0] for row in rows]).reshape(-1, 4).astype(np.float32),
                scores=np.concatenate([row[1] for row in rows]).astype(np.float32),
                labels=np.concatenate([row[2] for row in rows]).astype(np.int64))

def unpack(npz_data, prefix=''):
    """Inverse of pack: split the flat arrays of a loaded .npz back into per-frame (boxes, scores, labels) rows."""
    counts, boxes, scores, labels = (npz_data[prefix + 'counts'], npz_data[prefix + 'boxes'], npz_data[prefix + 'scores'],
                                     npz_data[prefix + 'labels'])
    ends = np.cumsum(counts)
    return [(boxes[i:j], scores[i:j], labels[i:j]) for i, j in zip(ends - counts, ends)]

def load_model(path):
    """Build the Faster R-CNN with the fold-0 inference config and load the weights stored at path."""
    cfg = json.loads((kaggle_submission.RUNS / 'fold0/config.json').read_text())
    model = kaggle_submission.build_model(cfg)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
    return model.to(DEVICE).eval()

def merge_frame(args):
    """Merge the TTA views of one frame with weighted box fusion (picklable worker function)."""
    parts, h, w = args
    return kaggle_submission.tta_merge(parts, 'wbf', h, w)

_GT = {}
def init_worker(datasets):
    """Worker initialiser: build the COCO ground-truth object of every evaluation set once per process."""
    for name, dataset in datasets.items():
        gt = COCO()
        gt.dataset = dataset
        with redirect_stdout(io.StringIO()):
            gt.createIndex()
        _GT[name] = gt

def eval_task(task):
    """COCO evaluation of one (set, model, config) on every frame group; returns one metrics row per group."""
    set_name, model_name, config, ids, rows, groups = task
    gt = _GT[set_name]
    detections = []
    for image_id, (boxes, scores, labels) in zip(ids, rows):
        for (x1, y1, x2, y2), score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            if x2 > x1 and y2 > y1:
                detections.append(dict(image_id=image_id, category_id=kaggle_submission.LABEL_TO_CAT[int(label)],
                                       bbox=[x1, y1, x2 - x1, y2 - y1], score=float(score)))
    with redirect_stdout(io.StringIO()):
        results = gt.loadRes(detections)
    group_rows = []
    for group, group_ids in groups.items():
        with redirect_stdout(io.StringIO()):
            evaluation = COCOeval(gt, results, 'bbox')
            evaluation.params.imgIds = sorted(group_ids)
            evaluation.params.maxDets = [1, 10, 100]
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        precision = evaluation.eval['precision'][:, :, :, 0, -1]
        scores = evaluation.stats
        row = dict(set=set_name, model=model_name, config=config, group=group, frames=len(group_ids),
                   mAP=float(precision[precision > -1].mean()), AP50=float(scores[1]),
                   AP75=float(scores[2]), APs=float(scores[3]), APm=float(scores[4]), APl=float(scores[5]), AR=float(scores[8]))
        if group == 'all':
            for class_index, class_name in enumerate(kaggle_submission.KAGGLE_CLASSES):
                class_precision = precision[:, :, class_index]
                row['AP_' + class_name] = (float(class_precision[class_precision > -1].mean())
                                           if (class_precision > -1).any() else float('nan'))
        group_rows.append(row)
    return group_rows

def slim_gt(coco, ids):
    """COCO dict restricted to the given image ids, with valid boxes only and consecutive annotation ids."""
    ids = set(ids)
    annotations = []
    for annotation in coco['annotations']:
        if int(annotation['image_id']) in ids and annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0:
            annotations.append(dict(annotation, area=float(annotation.get('area', annotation['bbox'][2] * annotation['bbox'][3])),
                                    iscrowd=int(annotation.get('iscrowd', 0)), id=len(annotations) + 1))
    return dict(info={}, licenses=[], images=[image for image in coco['images'] if int(image['id']) in ids],
                annotations=annotations, categories=coco['categories'])

def gt_boxes(anns, w, h, crowd):
    """Ground-truth boxes (x1, y1, x2, y2, clipped to the frame) and labels; crowd selects ignore regions or real objects."""
    boxes = np.array([[annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2],
                       annotation['bbox'][1] + annotation['bbox'][3]]
                      for annotation in anns if bool(annotation.get('iscrowd', 0)) == crowd], dtype=np.float32).reshape(-1, 4)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    labels = np.array([annotation['category_id'] for annotation in anns if bool(annotation.get('iscrowd', 0)) == crowd],
                      dtype=np.int64)
    return torch.from_numpy(boxes[keep]), torch.from_numpy(labels[keep])

@torch.inference_mode()
def proposal_recall(model, frames, anns_by_image, split, scale, n_list=(100, 300, 1000, 2000)):
    """Share of ground-truth objects covered by one of the top-n RPN proposals (IoU >= 0.5), per object size bucket."""
    kaggle_submission.set_scale(model, scale)
    original_top_n = dict(model.rpn._pre_nms_top_n), dict(model.rpn._post_nms_top_n)
    model.rpn._post_nms_top_n['testing'] = max(n_list)
    model.rpn._pre_nms_top_n['testing'] = max(model.rpn._pre_nms_top_n['testing'], max(n_list))
    hits, totals = {top_n: np.zeros(3) for top_n in n_list}, np.zeros(3)
    try:
        for i in np.unique(np.linspace(0, len(frames) - 1, min(300, len(frames))).astype(int)):
            frame = frames[int(i)]
            boxes, labels = gt_boxes(anns_by_image.get(int(frame['id']), []), frame['width'], frame['height'], crowd=False)
            if not len(boxes):
                continue
            area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).numpy()
            sizes = np.where(area < 32**2, 0, np.where(area < 96**2, 1, 2))
            image = kaggle_submission.to_tensor(
                kaggle_submission.read_images([DATA_DIR / split / frame['file_name']])[0]).to(DEVICE)
            inputs, transformed = model.transform([image], [dict(boxes=boxes.to(DEVICE), labels=labels.to(DEVICE))])
            proposals, _ = model.rpn(inputs, model.backbone(inputs.tensors))
            ious = box_iou(transformed[0]['boxes'], proposals[0]).cpu()
            for top_n in n_list:
                is_hit = (ious[:, :top_n].max(dim=1).values >= .5).numpy()
                for size_bucket in range(3):
                    hits[top_n][size_bucket] += np.logical_and(is_hit, sizes == size_bucket).sum()
            for size_bucket in range(3):
                totals[size_bucket] += (sizes == size_bucket).sum()
    finally:
        model.rpn._pre_nms_top_n, model.rpn._post_nms_top_n = original_top_n
        kaggle_submission.set_scale(model, 1.0)
    return [dict(size=name, n_gt=int(totals[size_bucket]),
                 **{f'top{top_n}': float(hits[top_n][size_bucket] / max(totals[size_bucket], 1)) for top_n in n_list})
            for size_bucket, name in enumerate(['small', 'medium', 'large'])]

def frame_errors(pred, anns, w, h):
    """Count true positives, false positives and missed objects of one frame; also returns the IoUs of the matches."""
    gt, gt_labels = gt_boxes(anns, w, h, crowd=False)
    ignored_boxes, _ = gt_boxes(anns, w, h, crowd=True)
    boxes, scores, labels = pred
    keep = scores >= SCORE_THRESHOLD
    boxes, labels = torch.from_numpy(boxes[keep]), torch.from_numpy(labels[keep])
    matched, true_positives, false_positives, ious = torch.zeros(len(gt), dtype=torch.bool), 0, 0, []
    iou = box_iou(boxes, gt) if len(gt) and len(boxes) else torch.zeros(len(boxes), len(gt))
    for j in range(len(boxes)):
        candidates = (iou[j] >= .5) & ~matched & (gt_labels == labels[j])
        if candidates.any():
            matched_index = torch.where(candidates)[0][iou[j][candidates].argmax()]
            matched[matched_index] = True
            true_positives += 1
            ious.append(float(iou[j, matched_index]))
        elif len(ignored_boxes) and float(((torch.min(boxes[j, 2:], ignored_boxes[:, 2:])
                                            - torch.max(boxes[j, :2], ignored_boxes[:, :2])).clamp(min=0).prod(1)
                                           / (boxes[j, 2:] - boxes[j, :2]).prod().clamp(min=1e-6)).max()) >= .5:
            pass
        else:
            false_positives += 1
    return true_positives, false_positives, len(gt) - true_positives, ious

@torch.inference_mode()
def tta_frame(model, image):
    """Test-time augmentation of one frame (3 scales x horizontal flip), merged with weighted box fusion."""
    h, w = image.shape[-2:]
    parts = []
    for scale in (1.0, 1.25, 1.5):
        kaggle_submission.set_scale(model, scale)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            predictions = model([image, image.flip(-1)])
        parts += [kaggle_submission.to_numpy(kaggle_submission.unflip(prediction['boxes'].float(), flip, h, w),
                                             prediction['scores'].float(), prediction['labels'])
                  for flip, prediction in zip(['none', 'h'], predictions)]
    kaggle_submission.set_scale(model, 1.0)
    return kaggle_submission.tta_merge(parts, 'wbf', h, w)

@torch.inference_mode()
def single_frame(model, image, scale):
    """Plain single-scale prediction of one frame."""
    kaggle_submission.set_scale(model, scale)
    with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
        output = model([image])
    kaggle_submission.set_scale(model, 1.0)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--dry', action='store_true',
                        help='CPU-only self-test of the evaluation path: the old views stand in for the new ones')
    args = parser.parse_args()
    fold = args.fold
    started = time.time()
    if args.dry:
        import tempfile
        out = Path(tempfile.mkdtemp(prefix='hires_eval_dry_'))
        result = dict(selected_checkpoint='final.pth', final_sha256='dry', cfg=dict(epochs=0), selection='dry run', history=[])
    else:
        out = FINETUNED_DIR / f'fold{fold}_eval'
        result = json.loads((FINETUNED_DIR / f'fold{fold}/result.json').read_text())
        assert (result['selected_checkpoint'] == 'final.pth'
                and kaggle_submission.sha256(FINETUNED_DIR / f'fold{fold}/final.pth') == result['final_sha256'])
    (out / 'views').mkdir(parents=True, exist_ok=True)
    old_path, new_path = kaggle_submission.RUNS / f'fold{fold}/best.pth', FINETUNED_DIR / f'fold{fold}/final.pth'
    log(f"old: {old_path.relative_to(ROOT)} | new: {new_path.relative_to(ROOT)} "
        f"(epoch {result['cfg']['epochs']}, {result['selection']})")

    coco_train = json.loads((DATA_DIR / 'train/annotations_coco.json').read_text())
    coco_val = json.loads((DATA_DIR / 'val/annotations_coco.json').read_text())
    folds = pd.read_csv(kaggle_submission.FOLDS_CSV)
    cv_ids = set(folds.loc[folds.Fold == fold, 'id'].astype(int))
    sets = {'CV held-out fold': dict(split='train', images=[image_info for image_info in coco_train['images']
                                                            if int(image_info['id']) in cv_ids], coco=coco_train),
            'official validation': dict(split='val', images=coco_val['images'], coco=coco_val)}
    for set_name, eval_set in sets.items():
        eval_set['ids'] = [int(image_info['id']) for image_info in eval_set['images']]
        eval_set['paths'] = [DATA_DIR / eval_set['split'] / image_info['file_name'] for image_info in eval_set['images']]
        eval_set['anns'], image_id_set = {}, set(eval_set['ids'])
        for coco_annotation in eval_set['coco']['annotations']:
            if int(coco_annotation['image_id']) in image_id_set:
                eval_set['anns'].setdefault(int(coco_annotation['image_id']), []).append(coco_annotation)
        eval_set['groups'] = {'all': eval_set['ids'],
                              **{group_name: [int(image_info['id']) for image_info in eval_set['images']
                                              if min_height <= image_info['height'] <= max_height]
                                 for group_name, (min_height, max_height) in GROUPS.items()}}
        eval_set['groups'] = {group_name: group_ids for group_name, group_ids in eval_set['groups'].items() if group_ids}
    del coco_train['annotations'][:]

    views = {}
    for set_name, eval_set in sets.items():
        tag = 'cv' if set_name.startswith('CV') else 'val'
        for view in VIEWS:
            if tag == 'cv':
                npz_file = np.load(CV_CACHE / f'fold{fold}_{view}.npz')
                rows = unpack(npz_file)
            else:
                npz_file = np.load(kaggle_submission.VAL_CACHE / f'fold{fold}_{view}.npz')
                rows = unpack(npz_file, 'full_')
            assert npz_file['ids'].tolist() == eval_set['ids'], (set_name, view)
            views[set_name, 'old', view] = rows
    new_model = None if args.dry else load_model(new_path)
    for set_name, eval_set in sets.items():
        tag = 'cv' if set_name.startswith('CV') else 'val'
        for view in VIEWS:
            if args.dry:
                views[set_name, 'new', view] = views[set_name, 'old', view]
                continue
            path = out / 'views' / f'{tag}_{view}.npz'
            if not path.exists():
                timer_start = time.time()
                rows = kaggle_submission.view_predictions(new_model, eval_set['paths'], view)
                np.savez(path.with_name(path.stem + '.tmp.npz'), ids=np.array(eval_set['ids']), **pack(rows))
                path.with_name(path.stem + '.tmp.npz').replace(path)
                log(f'new checkpoint, {set_name}, {view}: {len(rows)} frames in {time.time() - timer_start:.0f} s')
            npz_file = np.load(path)
            assert npz_file['ids'].tolist() == eval_set['ids']
            views[set_name, 'new', view] = unpack(npz_file)
    old_model = None if args.dry else load_model(old_path)
    for set_name, eval_set in sets.items():
        if args.dry:
            break
        fresh = kaggle_submission.view_predictions(old_model, eval_set['paths'][:8], 'h@1.25')
        assert all(kaggle_submission.same_arrays(new_row, cached_row)
                   for new_row, cached_row in zip(fresh, views[set_name, 'old', 'h@1.25'][:8])), set_name
    log('old checkpoint recomputed with this script equals its cached views (8 frames per set)')

    preds = {}
    with ProcessPoolExecutor(6) as pool:
        for set_name, eval_set in sets.items():
            dims = [(image_info['height'], image_info['width']) for image_info in eval_set['images']]
            for model_name in ['old', 'new']:
                for config_name, view in SINGLE.items():
                    preds[set_name, model_name, config_name] = views[set_name, model_name, view]
                tasks = [([views[set_name, model_name, view][k] for view in VIEWS], h, w) for k, (h, w) in enumerate(dims)]
                preds[set_name, model_name, 'TTA + WBF'] = list(pool.map(merge_frame, tasks, chunksize=32))
            log(f'{set_name}: WBF done for both checkpoints')
    datasets = {set_name: slim_gt(eval_set['coco'] if set_name.startswith('official')
                                  else dict(eval_set['coco'], annotations=[annotation for annotations in eval_set['anns'].values()
                                                                           for annotation in annotations]),
                                  eval_set['ids'])
                for set_name, eval_set in sets.items()}
    tasks = [(set_name, model_name, config_name, eval_set['ids'], preds[set_name, model_name, config_name], eval_set['groups'])
             for set_name, eval_set in sets.items() for model_name in ['old', 'new'] for config_name in CONFIGS]
    with ProcessPoolExecutor(4, initializer=init_worker, initargs=(datasets,)) as pool:
        metrics = pd.DataFrame([row for rows in pool.map(eval_task, tasks) for row in rows])
    metrics.to_csv(out / 'metrics.csv', index=False)
    log('COCO evaluation done')

    errors = []
    for set_name, eval_set in sets.items():
        for model_name in ['old', 'new']:
            true_positives = false_positives = missed_objects = 0
            ious = []
            for image_info, pred in zip(eval_set['images'], preds[set_name, model_name, 'TTA + WBF']):
                frame_true_positives, frame_false_positives, frame_missed, frame_ious = frame_errors(
                    pred, eval_set['anns'].get(int(image_info['id']), []), image_info['width'], image_info['height'])
                true_positives, false_positives, missed_objects = (
                    true_positives + frame_true_positives, false_positives + frame_false_positives,
                    missed_objects + frame_missed)
                ious += frame_ious
            errors.append(dict(set=set_name, model=model_name, config='TTA + WBF', score_thr=SCORE_THRESHOLD,
                               objects=true_positives + missed_objects, found=true_positives, missed=missed_objects,
                               false_positives=false_positives, recall=true_positives / max(true_positives + missed_objects, 1),
                               precision=true_positives / max(true_positives + false_positives, 1),
                               mean_iou_of_matches=float(np.mean(ious)),
                               share_matches_iou75=float(np.mean(np.array(ious) >= .75))))
    errors = pd.DataFrame(errors)
    errors.to_csv(out / 'errors.csv', index=False)

    if args.dry:
        old_single_720_map = metrics[(metrics.group == 'all') & (metrics.config == 'single 720')
                                     & (metrics.model == 'old')].set_index('set').mAP
        print(metrics[metrics.group == 'all'][['set', 'model', 'config', 'mAP', 'APs']].round(4).to_string(),
              errors.round(3).to_string(), sep=chr(10))
        print('DRY RUN OK | single 720 old mAP:', old_single_720_map.round(6).to_dict())
        sys.exit(0)
    cv_set = sets['CV held-out fold']
    recall = pd.DataFrame([dict(model=model_name, input=f'{round(720 * scale)} px', **recall_row)
                           for model_name, model in [('old', old_model), ('new', new_model)]
                           for scale in (1.0, 1.5)
                           for recall_row in proposal_recall(model, cv_set['images'], cv_set['anns'], 'train', scale)])
    recall.to_csv(out / 'rpn_recall.csv', index=False)
    log('RPN recall done')
    rng = np.random.default_rng(SEED + fold)
    picks = [cv_set['paths'][i] for i in rng.choice(len(cv_set['paths']), 70, replace=False)]
    frames = [kaggle_submission.to_tensor(array).to(DEVICE) for array in kaggle_submission.read_images(picks)]
    runs = {'single 720': lambda mod, frame: single_frame(mod, frame, 1.0),
            'single 1080': lambda mod, frame: single_frame(mod, frame, 1.5), 'TTA + WBF': tta_frame}
    times, memory = {(model_name, config_name): [] for model_name in ['old', 'new'] for config_name in runs}, {}
    for i, image in enumerate(frames):
        for model_name, model in [('old', old_model), ('new', new_model)]:
            for config_name, run_inference in runs.items():
                torch.cuda.synchronize()
                timer_start = time.perf_counter()
                run_inference(model, image)
                torch.cuda.synchronize()
                if i >= 10:
                    times[model_name, config_name].append(time.perf_counter() - timer_start)
    for model_name, model in [('old', old_model), ('new', new_model)]:
        for config_name, run_inference in runs.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for image in frames[:10]:
                run_inference(model, image)
            memory[model_name, config_name] = torch.cuda.max_memory_allocated() / 2**30
    speed = pd.DataFrame([dict(model=model_name, config=config_name, ms_per_frame=float(np.mean(frame_times) * 1000),
                               peak_gpu_gb=memory[model_name, config_name])
                          for (model_name, config_name), frame_times in times.items()])
    speed.to_csv(out / 'speed.csv', index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    wide = metrics.pivot_table(index=['set', 'group', 'config'], columns='model', values=KEYS, sort=False)
    delta = pd.DataFrame({metric: wide[metric]['new'] - wide[metric]['old'] for metric in KEYS})
    all_group_rows = metrics[metrics.group == 'all'].set_index(['set', 'config', 'model'])
    print('\n=== all frames: old / new / difference ===')
    for set_name in sets:
        for config_name in CONFIGS:
            old_metrics, new_metrics = (all_group_rows.loc[(set_name, config_name, 'old')],
                                        all_group_rows.loc[(set_name, config_name, 'new')])
            print(f"{set_name:20s} {config_name:12s} " + ' | '.join(
                f"{metric} {old_metrics[metric]:.4f} -> {new_metrics[metric]:.4f} "
                f"({new_metrics[metric] - old_metrics[metric]:+.4f})"
                for metric in ['mAP', 'AP50', 'APs', 'APm', 'APl', 'AR']))
    print('\n=== mAP difference (new - old) by native short side ===')
    print(delta.mAP.unstack('config')[CONFIGS].round(4).to_string())
    print('\n=== TTA + WBF: per-class AP, old -> new ===')
    for set_name in sets:
        old_metrics, new_metrics = (all_group_rows.loc[(set_name, 'TTA + WBF', 'old')],
                                    all_group_rows.loc[(set_name, 'TTA + WBF', 'new')])
        print(set_name + ': ' + ' | '.join(
            f"{class_name} {old_metrics['AP_' + class_name]:.3f}->{new_metrics['AP_' + class_name]:.3f}"
            for class_name in kaggle_submission.KAGGLE_CLASSES))
    print('\n=== errors at score >= 0.3 (TTA + WBF) ===')
    print(errors.round(4).to_string(index=False))
    print('\n=== RPN recall, IoU >= 0.5 (CV held-out fold, 300 frames) ===')
    print(recall.round(3).to_string(index=False))
    print('\n=== speed (60 frames after 10 warm-up, batch 1, AMP) ===')
    print(speed.round(2).to_string(index=False))
    map_difference = {set_name: {group_name: float(delta.mAP.loc[(set_name, group_name, 'TTA + WBF')])
                    for group_name in sets[set_name]['groups']}
         for set_name in sets}
    main = all(map_difference[set_name]['all'] >= 0.010 for set_name in sets)
    test_like = all(map_difference[set_name][group_name] > 0 for set_name in sets for group_name in ('756-765', '1071-1080')
                    if group_name in map_difference[set_name])
    verdict = ('continue: fine-tune the remaining folds' if main and test_like else
               'inconclusive: one more fold needed' if all(map_difference[set_name]['all'] >= 0.005 for set_name in sets)
               else 'stop: no meaningful improvement')
    summary = dict(fold=fold, old=str(old_path.relative_to(ROOT)), new=str(new_path.relative_to(ROOT)),
                   new_sha256=result['final_sha256'],
                   delta_mAP_tta_wbf=map_difference,
                   criteria='>= +0.010 mAP (TTA + WBF) on both sets and positive on the 756-765 and 1071-1080 groups',
                   verdict=verdict, training_history=result['history'], minutes=(time.time() - started) / 60)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nTTA + WBF mAP difference:', json.dumps(map_difference), '\nVERDICT (criteria fixed before training):', verdict)
    log(f'done in {(time.time() - started) / 60:.1f} min')
