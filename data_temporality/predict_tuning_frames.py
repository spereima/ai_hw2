"""
Faster R-CNN predictions on 80 consecutive frames from each training video using 6 TTA views.
The results are used to tune temporal post-processing.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'shared'))
import tta_wbf

OUT = Path(__file__).resolve().parent / 'cache'
SEGMENT = 80


def tuning_segments():
    """SEGMENT consecutive frames from the middle of every training video, with the CV fold that holds the video out."""
    coco = json.loads((tta_wbf.DATA / 'train/annotations_coco.json').read_text())
    images = pd.DataFrame(coco['images']).sort_values(['video', 'frame_id'])
    folds = pd.read_csv(tta_wbf.FOLDS_CSV).groupby('video').Fold.agg(['first', 'nunique'])
    assert (folds['nunique'] == 1).all() and set(folds.index) == set(images.video)
    segments = []
    for video, frames in images.groupby('video', sort=True):
        start = max(0, (len(frames) - SEGMENT) // 2)
        part = frames.iloc[start:start + SEGMENT]
        assert (np.diff(part.frame_id) == 1).all()
        segments.append(part.assign(fold=int(folds.loc[video, 'first'])))
    segment_table = (pd.concat(segments)[['id', 'video', 'frame_id', 'fold', 'file_name', 'width', 'height']]
                     .rename(columns={'id': 'image_id'}))
    return segment_table.sort_values(['fold', 'video', 'frame_id']).reset_index(drop=True)


if __name__ == '__main__':
    (OUT / 'tune_views').mkdir(parents=True, exist_ok=True)
    segment_table = tuning_segments()
    segment_table.to_csv(OUT / 'tune_segments.csv', index=False)
    tta_wbf.log(
        f'{len(segment_table)} consecutive frames from {segment_table.video.nunique()} training videos; '
        f'frames per fold: {segment_table.groupby("fold").size().to_dict()}')
    started = time.time()
    for fold in range(5):
        part = segment_table[segment_table.fold == fold]
        todo = [view for view in tta_wbf.TTA_VIEWS
                if not (OUT / 'tune_views' / f'fold{fold}_{view}.npz').exists()]
        if not todo:
            continue
        model, cfg, checkpoint = tta_wbf.load_model(fold)
        paths = [tta_wbf.DATA / 'train' / file_name for file_name in part.file_name]
        for view in todo:
            rows = tta_wbf.view_predictions(model, paths, view)
            path = OUT / 'tune_views' / f'fold{fold}_{view}.npz'
            tmp_path = path.with_name(path.stem + '.tmp.npz')
            np.savez(tmp_path, ids=part.image_id.values, **tta_wbf.pack(rows))
            os.replace(tmp_path, path)
            tta_wbf.log(
                f'fold {fold} {view}: {len(rows)} frames | elapsed {(time.time() - started) / 60:.1f} min')
        del model
    tta_wbf.log('done')
