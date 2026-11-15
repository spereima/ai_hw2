"""
Preparing RT-DETR results for the final comparison with Faster R-CNN and FCOS using the same metrics and evaluation conditions.
"""
import datetime
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rtdetr
sys.path.insert(0, str(rtdetr.ROOT))
import final_evaluation

MODEL = 'rt_detr'
SPEC = final_evaluation.COMPARED_MODELS[MODEL]
ALL_RUNS = ['recipe_warmup_ema', 'queries_200', 'queries_450', 'matching_bbox7_giou3', 'no_denoising', 'lr_backbone_2e-5',
            'no_denoising_640x1152']
OUT = final_evaluation.FINAL / MODEL


def official_metrics(experiment, fold):
    """Official-validation metrics of one checkpoint (official_val.json, evaluated once after training)."""
    return json.loads((final_evaluation.run_dir(MODEL, experiment, fold) / 'official_val.json').read_text())['metrics']


def preprocess_on_gpu(images, height, width):
    """uint8 CHW frames of one resolution on the GPU -> pixel_values as the RT-DETR image processor makes them."""
    batch = torch.stack(images).float().div(255)
    return F.interpolate(batch, size=(height, width), mode='bilinear', align_corners=False, antialias=True)


def check_preprocessing(cfg):
    """Mean absolute difference between the GPU resize used for timing and the RT-DETR image processor, on one speed frame."""
    _, image = final_evaluation._speed_images()[0]
    reference = rtdetr.processor(cfg)(images=[image.permute(1, 2, 0).cpu().numpy()], return_tensors='pt')['pixel_values']
    timed = preprocess_on_gpu([image], cfg['height'], cfg['width']).cpu()
    return dict(mean_abs_difference=float((reference - timed).abs().mean()), max_abs_difference=float((reference - timed).abs().max()),
                pixel_value_range='0..1')


@torch.inference_mode()
def benchmark(model, cfg, batch_size, amp=True, warmup=20):
    """final_evaluation.benchmark for RT-DETR: same frames, batching, warm-up and synchronisation; times per image."""
    image_processor = rtdetr.processor(cfg)
    groups = {}
    for _, image in final_evaluation._speed_images():
        groups.setdefault(tuple(image.shape), []).append(image)
    batches = [group[i:i + batch_size] for group in groups.values() for i in range(0, len(group), batch_size)]

    def run(batch):
        sizes = torch.tensor([image.shape[1:] for image in batch], device=rtdetr.DEV)
        with torch.autocast('cuda', enabled=amp):
            output = model(pixel_values=preprocess_on_gpu(batch, cfg['height'], cfg['width']))
        return image_processor.post_process_object_detection(output, threshold=cfg['score_thr'], target_sizes=sizes)

    for step in range(warmup):
        run(batches[step % len(batches)])
    per_image = []
    for batch in batches:
        torch.cuda.synchronize()
        start = time.perf_counter()
        run(batch)
        torch.cuda.synchronize()
        per_image += [(time.perf_counter() - start) * 1000 / len(batch)] * len(batch)
    times = np.asarray(per_image)
    return dict(ms_per_image=float(times.mean()), ms_median=float(np.median(times)), ms_p90=float(np.percentile(times, 90)),
                fps=float(1000 / times.mean()), n_images=len(times), n_batches=len(batches))


def measure_speed(experiment, fold, batch_size=1, amp=True):
    """Speed of one checkpoint, cached per checkpoint hash and protocol (never overwritten)."""
    path = OUT / 'speed' / f'{experiment}_fold{fold}_bs{batch_size}_{"amp" if amp else "fp32"}.json'
    record = final_evaluation.checkpoint_record(MODEL, experiment, fold)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['speed_id'] == final_evaluation.SPEED_ID and saved['checkpoint_sha256'] == record['checkpoint_sha256']:
            return saved
        raise RuntimeError(f'{path} exists for a different checkpoint/protocol; move it away (not overwritten)')
    model, cfg = rtdetr.load_model(experiment, fold)
    result = benchmark(model, cfg, batch_size, amp)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    out = {**{key: value for key, value in record.items() if key != 'cfg'}, 'speed_id': final_evaluation.SPEED_ID,
           'batch_size': batch_size, 'precision': 'amp_fp16' if amp else 'fp32', 'gpu': torch.cuda.get_device_name(0),
           'input_resize': f"{cfg['height']} x {cfg['width']} (bilinear on the GPU, as the RT-DETR image processor)", **result,
           'created': datetime.datetime.now().isoformat(timespec='seconds')}
    final_evaluation._atomic(path, json.dumps(out, indent=2, default=final_evaluation._json_default))
    return out


def fold_table(speed):
    """Rows of rt_detr_folds.csv: every compared experiment, fold and split."""
    official_frames = len(final_evaluation.coco_json('val')['images'])
    latency = speed[(speed.batch_size == 1) & (speed.precision == 'amp_fp16')].set_index(['experiment', 'fold'])
    rows = []
    for experiment in SPEC['experiments']:
        for fold in final_evaluation.FOLDS:
            result = final_evaluation.load_result(MODEL, experiment, fold)
            timing = latency.loc[(experiment, fold)]
            base = dict(model=SPEC['label'], experiment=experiment, fold=fold, best_epoch=int(result['best_epoch']),
                        ms_per_image=float(timing.ms_per_image), fps=float(timing.fps))
            cv_frames = len(final_evaluation.cv_fold_ids(fold))
            assert sorted(map(int, rtdetr.fold_ids(fold)[1])) == sorted(final_evaluation.cv_fold_ids(fold))
            rows.append(dict(base, evaluation_split='cv_out_of_fold',
                             **{final_evaluation.COLUMNS[key]: result['metrics'][key] for key in final_evaluation.METRICS},
                             n_images=cv_frames, metrics_source=f"{SPEC['notebook']} result.json"))
            official = official_metrics(experiment, fold)
            rows.append(dict(base, evaluation_split='official_val',
                             **{final_evaluation.COLUMNS[key]: official[key] for key in final_evaluation.METRICS},
                             n_images=official_frames, metrics_source=f"{SPEC['notebook']} official_val.json"))
    table = pd.DataFrame(rows)
    order = ['model', 'experiment', 'fold', 'best_epoch', 'evaluation_split'] + list(final_evaluation.COLUMNS.values()) + \
            ['ms_per_image', 'fps', 'n_images', 'metrics_source']
    table = table[order].sort_values(['evaluation_split', 'experiment', 'fold']).reset_index(drop=True)
    table['speed_protocol'] = 'standardised: RTX 5070, batch 1, AMP, 200 official-val frames (see inference_speed.csv)'
    return table


def selection_record():
    """Selection by mean CV mAP over all 7 RT-DETR runs; official validation only as an independent check."""
    cv_means = {run: float(np.mean([final_evaluation.load_result(MODEL, run, fold)['metrics']['mAP']
                                    for fold in final_evaluation.FOLDS])) for run in ALL_RUNS}
    official_means = {run: float(np.mean([official_metrics(run, fold)['mAP'] for fold in final_evaluation.FOLDS]))
                      for run in ALL_RUNS}
    best_cv, best_official = max(cv_means, key=cv_means.get), max(official_means, key=official_means.get)
    assert best_cv in SPEC['experiments']
    official = pd.DataFrame([dict(experiment=best_cv, fold=fold, mAP=official_metrics(best_cv, fold)['mAP'])
                             for fold in final_evaluation.FOLDS])
    representative = final_evaluation.representative_fold(official, best_cv)
    run_dir = final_evaluation.run_dir
    return dict(model=SPEC['label'], best_by_cv=best_cv, cv_mean_map=cv_means, best_by_official_val=best_official,
                official_val_mean_map=official_means, agree=best_cv == best_official, representative_fold=representative,
                representative_checkpoint=str(run_dir(MODEL, best_cv, representative) / 'best.pth'),
                representative_rule='fold with the median official-validation mAP of the CV-selected experiment',
                fold_checkpoints=[str(run_dir(MODEL, best_cv, fold) / 'best.pth') for fold in final_evaluation.FOLDS],
                compared_experiments=SPEC['experiments'],
                rule='experiment selected by mean CV (out-of-fold) mAP over all 7 RT-DETR runs; official validation only as '
                     'an independent check')


def main():
    selection = selection_record()
    rows = [measure_speed(experiment, fold, batch_size) for experiment in SPEC['experiments']
            for fold in final_evaluation.FOLDS for batch_size in (1, 4)]
    rows.append(measure_speed(selection['best_by_cv'], selection['representative_fold'], batch_size=1, amp=False))
    keep = ['model', 'experiment', 'fold', 'batch_size', 'precision', 'ms_per_image', 'ms_median', 'ms_p90', 'fps',
            'n_images', 'n_batches', 'gpu', 'input_resize']
    speed = pd.DataFrame(rows)[keep]
    speed['original_training_ms_per_img'] = [final_evaluation.load_result(MODEL, experiment, fold)['ms_per_img']
                                             for experiment, fold in zip(speed.experiment, speed.fold)]
    folds = fold_table(speed)
    _, cfg = rtdetr.load_model(selection['best_by_cv'], selection['representative_fold'])
    selection['timing_preprocessing_check'] = check_preprocessing(cfg)
    final_evaluation.save_csv(folds, OUT / f'{MODEL}_folds.csv')
    final_evaluation.save_csv(final_evaluation.summarize(folds), OUT / f'{MODEL}_summary_mean_std.csv')
    final_evaluation.save_csv(speed, OUT / f'{MODEL}_inference_speed.csv')
    final_evaluation.save_json(selection, OUT / 'selection.json')
    print(final_evaluation.summarize(folds)[['experiment', 'evaluation_split', 'map_mean', 'map_std', 'ms_per_image_mean']].round(4))
    print('selection:', selection['best_by_cv'], '| representative fold', selection['representative_fold'],
          '| preprocessing check', selection['timing_preprocessing_check'])


if __name__ == '__main__':
    main()
