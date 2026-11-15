"""
RetinaNet prediction stage using TTA.

TTA includes horizontal flipping and resizing with scale factors
of 0.75, 1.0, 1.25, and 1.5 relative to the training resize of 800/1333.
"""

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
for path in (ROOT / 'shared', ROOT / 'data_temporality', TASK):
    sys.path.insert(0, str(path))

import tta_wbf
import experiment_utils
import retina

VIEWS = ['none@1.0', 'h@1.0', 'none@1.25', 'h@1.25', 'none@1.5', 'h@1.5', 'none@0.75', 'h@0.75']
BATCH_SIZE_BY_SCALE = {0.75: 4, 1.0: 4, 1.25: 2, 1.5: 2}
CACHE = HERE / 'cache' / 'views'
CHUNK = 400


def final_name():
    """Run name of the final RetinaNet configuration (from screening.json)."""
    return json.loads((TASK / 'screening.json').read_text())['final']['name']


def set_scale(model, cfg, scale):
    """Set the model's resize to `scale` times the trained min_size / max_size."""
    model.transform.min_size = (round(cfg['min_size'] * scale),)
    model.transform.max_size = round(cfg['max_size'] * scale)


def load(paths):
    """Decode the frames of `paths` with 4 threads."""
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(lambda path: tta_wbf.read_images([path])[0], paths))


@torch.inference_mode()
def all_views(model, cfg, paths, views):
    """Predictions of every view for the frames of `paths`; the next group of 4 frames is decoded while the GPU runs."""
    predictions = {view: [] for view in views}
    path_groups = [paths[i:i + 4] for i in range(0, len(paths), 4)]
    with ThreadPoolExecutor(1) as reader:
        pending = reader.submit(load, path_groups[0])
        for group_index in range(len(path_groups)):
            arrays = pending.result()
            if group_index + 1 < len(path_groups):
                pending = reader.submit(load, path_groups[group_index + 1])
            images = [tta_wbf.to_tensor(array).to(retina.DEV, non_blocking=True) for array in arrays]
            for view in views:
                flip, scale = tta_wbf.parse_view(view)
                set_scale(model, cfg, scale)
                batch_size = BATCH_SIZE_BY_SCALE[scale]
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size]
                    with torch.autocast(device_type='cuda'):
                        preds = model([tta_wbf.flip_image(image, flip) for image in batch])
                    predictions[view] += [
                        (tta_wbf.unflip(pred['boxes'].float(), flip, *image.shape[-2:]).cpu().numpy(),
                         pred['scores'].float().cpu().numpy(),
                         pred['labels'].cpu().numpy().astype(np.int64) + 1)
                        for image, pred in zip(batch, preds)]
            set_scale(model, cfg, 1.0)
    return predictions


def frames(name, fold):
    """Image ids and file paths of a frame set; for 'tune' only the videos of the held-out fold."""
    meta = experiment_utils.frames_meta(name)
    if name == 'tune':
        meta = meta[meta.fold == fold]
    return meta.image_id.astype(str).tolist(), meta.path.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', default=['tune', 'val', 'test'])
    parser.add_argument('--views', nargs='+', default=VIEWS)
    args = parser.parse_args()
    run_name = final_name()
    frames_by_set_fold = {(set_name, fold): frames(set_name, fold) for set_name in args.sets for fold in range(5)}
    total = sum(len(frame_ids) for frame_ids, _ in frames_by_set_fold.values())
    done, started = 0, time.time()
    for set_name in args.sets:
        for fold in range(5):
            ids, paths = frames_by_set_fold[(set_name, fold)]
            folder = CACHE / set_name / f'fold{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            model = None
            for start in range(0, len(ids), CHUNK):
                view_files = {view: folder / f'{view}_{start:05d}.npz' for view in args.views}
                n_frames = len(ids[start:start + CHUNK])
                missing_views = [view for view, file in view_files.items() if not file.exists()]
                if not missing_views:
                    total -= n_frames
                    continue
                if model is None:
                    model, cfg = retina.load_model(run_name, fold)
                    assert cfg['fold'] == fold and cfg['folds'] == retina.FOLDS_SHA
                chunk_started = time.time()
                view_predictions = all_views(model, cfg, paths[start:start + CHUNK], missing_views)
                for view in missing_views:
                    tmp_file = view_files[view].with_name(view_files[view].stem + '.tmp.npz')
                    np.savez(tmp_file, ids=np.array(ids[start:start + CHUNK], dtype=str),
                             **tta_wbf.pack(view_predictions[view]))
                    os.replace(tmp_file, view_files[view])
                done += n_frames
                elapsed = time.time() - started
                tta_wbf.log(
                    f'{run_name} fold {fold} {set_name}: {start + n_frames}/{len(ids)} | '
                    f'{1000 * (time.time() - chunk_started) / n_frames:.0f} ms/frame ({len(missing_views)} views) | '
                    f'{done}/{total} | elapsed {elapsed / 60:.0f} min | ETA {elapsed / max(done, 1) * (total - done) / 60:.0f} min')
            del model
            torch.cuda.empty_cache()
        tta_wbf.log(f'set {set_name} complete')
    tta_wbf.log('done')
