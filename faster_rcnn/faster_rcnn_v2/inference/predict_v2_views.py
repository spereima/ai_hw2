import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[3]))
for import_dir in (ROOT / 'shared', ROOT / 'data_temporality', ROOT / 'faster_rcnn/faster_rcnn_v2'):
    sys.path.insert(0, str(import_dir))
import tta_wbf
import experiment_utils
import faster_rcnn_v2

RUN = 'v2_lr_scheduler_scale_aug'
CACHE = HERE / 'cache' / 'views'
CHUNK = 400
BATCH = 4
assert all(BATCH % batch_size == 0 for batch_size in tta_wbf.TTA_BATCH.values()) and CHUNK % BATCH == 0


def load(paths):
    """Decode frames as uint8 arrays in 4 threads (float tensors are made in the main thread)."""
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(lambda path: tta_wbf.read_images([path])[0], paths))


@torch.inference_mode()
def all_views(model, paths):
    """{view: rows} for a list of frames; identical batching to
    tta_wbf.view_predictions(model, paths, view) for every view."""
    predictions_by_view = {view: [] for view in tta_wbf.TTA_VIEWS}
    batches = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
    with ThreadPoolExecutor(1) as loader:
        pending = loader.submit(load, batches[0])
        for batch_index in range(len(batches)):
            arrays = pending.result()
            if batch_index + 1 < len(batches):
                pending = loader.submit(load, batches[batch_index + 1])
            images = [tta_wbf.to_tensor(array).to(tta_wbf.DEV, non_blocking=True)
                      for array in arrays]
            for view in tta_wbf.TTA_VIEWS:
                flip, scale = tta_wbf.parse_view(view)
                tta_wbf.set_scale(model, scale)
                batch_size = tta_wbf.TTA_BATCH[scale]
                for i in range(0, len(images), batch_size):
                    sub_batch = images[i:i + batch_size]
                    with torch.autocast(device_type=tta_wbf.DEV.type,
                                        enabled=tta_wbf.AMP):
                        predictions = model([tta_wbf.flip_image(image, flip) for image in sub_batch])
                    predictions_by_view[view] += [
                        tta_wbf.to_numpy(
                            tta_wbf.unflip(prediction['boxes'].float(), flip, *image.shape[-2:]),
                            prediction['scores'].float(), prediction['labels'])
                        for image, prediction in zip(sub_batch, predictions)]
            tta_wbf.set_scale(model, 1.0)
    return predictions_by_view


def frames(name, fold):
    """Image ids and paths of frame set `name`; for 'tune' only the videos held out by `fold`."""
    meta = experiment_utils.frames_meta(name)
    if name == 'tune':
        meta = meta[meta.fold == fold]
    return meta.image_id.astype(str).tolist(), meta.path.tolist()


def reproduction_check(model, fold):
    """all_views must give exactly the arrays of tta_wbf.view_predictions on 8 validation frames."""
    paths = experiment_utils.frames_meta('val').path.tolist()[:8]
    combined = all_views(model, paths)
    for view in tta_wbf.TTA_VIEWS:
        reference = tta_wbf.view_predictions(model, paths, view)
        assert all(tta_wbf.same_arrays(row, reference_row)
                   for row, reference_row in zip(combined[view], reference)), \
            f'fold {fold} {view}: differs from ks.view_predictions'
    tta_wbf.log(f'fold {fold}: reproduction check passed (6 views x 8 frames equal ks.view_predictions)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--sets', nargs='+', default=['tune', 'val', 'test'])
    args = parser.parse_args()
    todo = {(name, fold): frames(name, fold) for fold in args.folds for name in args.sets}
    total = sum(len(ids) for ids, _ in todo.values())
    done, started, checked = 0, time.time(), set()
    for name in args.sets:
        for fold in args.folds:
            model = None
            ids, paths = todo[(name, fold)]
            folder = CACHE / name / f'fold{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(ids), CHUNK):
                view_files = {view: folder / f'{view}_{start:05d}.npz' for view in tta_wbf.TTA_VIEWS}
                n_frames = len(ids[start:start + CHUNK])
                if all(path.exists() for path in view_files.values()):
                    total -= n_frames
                    continue
                if model is None:
                    model, cfg = faster_rcnn_v2.load_model(RUN, fold)
                    assert cfg['fold'] == fold and cfg['folds'] == tta_wbf.FOLDS_SHA
                    assert all(cfg[key] == tta_wbf.INFERENCE[key]
                               for key in tta_wbf.INFERENCE_KEYS), 'inference settings differ from Faster R-CNN'
                    if fold not in checked:
                        reproduction_check(model, fold)
                        checked.add(fold)
                chunk_started = time.time()
                predictions_by_view = all_views(model, paths[start:start + CHUNK])
                for view, path in view_files.items():
                    tmp_path = path.with_name(path.stem + '.tmp.npz')
                    np.savez(tmp_path, ids=np.array(ids[start:start + CHUNK], dtype=str),
                             **tta_wbf.pack(predictions_by_view[view]))
                    os.replace(tmp_path, path)
                done += n_frames
                elapsed = time.time() - started
                tta_wbf.log(
                    f'fold {fold} {name}: {start + n_frames}/{len(ids)} | '
                    f'{1000 * (time.time() - chunk_started) / n_frames:.0f} ms/frame (6 views) | '
                    f'{done}/{total} frames | elapsed {elapsed / 60:.0f} min | '
                    f'ETA {elapsed / done * (total - done) / 60:.0f} min')
            del model
            torch.cuda.empty_cache()
        tta_wbf.log(f'set {name} complete')
    tta_wbf.log('done')
