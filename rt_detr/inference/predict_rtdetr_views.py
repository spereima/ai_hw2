"""
RT-DETR prediction stage using TTA.
The RT-DETR configuration with the highest mean mAP across 5 folds is selected for inference.
TTA includes horizontal flipping and resizing with scale factors of 0.75, 1.0, 1.25, and 1.5 relative to the training input size.
Predictions are generated on three datasets:
- tune: Consecutive frames from 56 training videos. Each fold processes only its own held-out videos (out-of-fold).
- val: Official validation set, processed by every checkpoint.
- test: Kaggle test set, processed by every checkpoint.
All TTA views are first tested on the tune set, where the best combination is selected. Only the selected TTA views and
the standard `none@1.0` view are then used on the val and test sets.
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

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
for search_path in (ROOT / 'shared', ROOT / 'data_temporality', TASK):
    sys.path.insert(0, str(search_path))

import tta_wbf
import experiment_utils
import rtdetr

VIEWS = ['none@1.0', 'h@1.0', 'none@1.25', 'h@1.25', 'none@1.5', 'h@1.5', 'none@0.75', 'h@0.75']
CACHE = HERE / 'cache' / 'views'
CHUNK = 400
BATCH = 4

def chosen_config():
    """Name of the RT-DETR configuration with the highest mean CV mAP among those with all 5 folds (also written to chosen_config.json)."""
    results = pd.read_csv(TASK / 'five_fold_results.csv')
    complete_configs = results.groupby('config').filter(lambda group: group.fold.nunique() == 5)
    mean_mAP_by_config = complete_configs.groupby('config').mAP.mean().sort_values(ascending=False)
    choice = dict(config=mean_mAP_by_config.index[0], cv_mAP=mean_mAP_by_config.to_dict())
    (HERE / 'chosen_config.json').write_text(json.dumps(choice, indent=2))
    return mean_mAP_by_config.index[0]

def load(paths):
    """Read the images of `paths` (uint8 arrays) in 4 threads."""
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(lambda path: tta_wbf.read_images([path])[0], paths))

@torch.inference_mode()
def all_views(model, cfg, paths, views):
    """Predict every requested view of every frame (images decoded once, next batch loaded while the GPU works): {view: [(boxes, scores, labels)]}."""
    predictions_by_view = {view: [] for view in views}
    path_groups = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
    with ThreadPoolExecutor(1) as prefetcher:
        pending = prefetcher.submit(load, path_groups[0])
        for group_idx in range(len(path_groups)):
            arrays = pending.result()
            if group_idx + 1 < len(path_groups):
                pending = prefetcher.submit(load, path_groups[group_idx + 1])
            for view in views:
                flip, scale = tta_wbf.parse_view(view)
                view_cfg = dict(cfg, height=int(round(cfg['height'] * scale / 32) * 32), width=int(round(cfg['width'] * scale / 32) * 32))
                rows = rtdetr.predict_arrays(model, view_cfg, arrays, flip=(flip == 'h'))
                predictions_by_view[view] += [(row['boxes'].numpy(), row['scores'].numpy(), row['labels'].numpy().astype(np.int64)) for row in rows]
    return predictions_by_view

def frames(name, fold):
    """(image ids as strings, image paths) of the frame set `name` (tune: only the held-out videos of `fold`)."""
    meta = experiment_utils.frames_meta(name)
    if name == 'tune':
        meta = meta[meta.fold == fold]
    return meta.image_id.astype(str).tolist(), meta.path.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', default=['tune', 'val', 'test'])
    parser.add_argument('--views', nargs='+', default=VIEWS, choices=VIEWS,
                        help='views to predict (val/test: only the views selected on the tuning frames, plus none@1.0)')
    args = parser.parse_args()
    config_name = chosen_config()
    tta_wbf.log(f'configuration: {config_name}')
    frames_by_set_and_fold = {(frame_set, fold): frames(frame_set, fold) for frame_set in args.sets for fold in range(5)}
    total_frames, done_frames, start_time = sum(len(image_ids) for image_ids, _ in frames_by_set_and_fold.values()), 0, time.time()
    for frame_set in args.sets:
        for fold in range(5):
            if (rtdetr.HERE / 'STOP').exists():
                tta_wbf.log('STOP file found')
                sys.exit(0)
            image_ids, paths = frames_by_set_and_fold[(frame_set, fold)]
            folder = CACHE / frame_set / f'fold{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            model = None
            for start in range(0, len(image_ids), CHUNK):
                files = {view: folder / f'{view}_{start:05d}.npz' for view in args.views}
                num_frames = len(image_ids[start:start + CHUNK])
                missing_views = [view for view, file in files.items() if not file.exists()]
                if not missing_views:
                    total_frames -= num_frames
                    continue
                if model is None:
                    model, cfg = rtdetr.load_model(config_name, fold)
                    assert cfg['fold'] == fold
                chunk_start_time = time.time()
                view_predictions = all_views(model, cfg, paths[start:start + CHUNK], missing_views)
                for view in missing_views:
                    tmp_path = files[view].with_name(files[view].stem + '.tmp.npz')
                    np.savez(tmp_path, ids=np.array(image_ids[start:start + CHUNK], dtype=str), **tta_wbf.pack(view_predictions[view]))
                    os.replace(tmp_path, files[view])
                done_frames += num_frames
                elapsed = time.time() - start_time
                tta_wbf.log(
                    f'{config_name} fold {fold} {frame_set}: {start + num_frames}/{len(image_ids)} | '
                    f'{1000 * (time.time() - chunk_start_time) / num_frames:.0f} ms/frame ({len(missing_views)} views) | '
                    f'{done_frames}/{total_frames} | ETA {elapsed / max(done_frames, 1) * (total_frames - done_frames) / 60:.0f} min')
            del model
            torch.cuda.empty_cache()
        tta_wbf.log(f'set {frame_set} complete')
    tta_wbf.log('done')
