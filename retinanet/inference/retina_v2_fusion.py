"""
Additional experiment combining RetinaNet and Faster R-CNN v2 without FCOS.

The RetinaNet ensemble (5 folds + TTA + WBF) is combined with the Faster R-CNN v2 ensemble
(5 folds + TTA + WBF), followed by temporal post-processing.

- RetinaNet weights: 0.25, 0.5, 0.75, 1.0.
- WBF IoU: 0.55, 0.65.
- Tuning of temporal post-processing parameters for Faster R-CNN.
"""

import json, os, sys, time
from pathlib import Path
import pandas as pd
import retina_pipeline
from retina_pipeline import detector_fusion, pipeline_tuning, experiment_utils, tta_wbf, ensemble_pipeline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TUNING = retina_pipeline.TUNING
OFFICIAL_RAW_JSON = HERE / 'official_validation_retina_v2_raw.json'
A_KEY = 'A: v2 + temporal (tuned)'
FUSION_KEY = 'RetinaNet + Faster R-CNN v2 fusion + temporal (tuned)'
OUT_CSV = HERE / 'kaggle_submissions' / 'retinanet_v2_5fold_tta_wbf_fusion_temporal_submission.csv'


def run(stage, jobs, workers=6):
    """Evaluate the jobs of one stage on the tuning frames, Faster R-CNN v2 as second input (cached as tuning/<stage>.csv)."""
    path = TUNING / f'{stage}.csv'
    if path.exists():
        return pd.read_csv(path)
    started = time.time()
    table = detector_fusion.table(detector_fusion.evaluate_jobs('tune', jobs, None, (), ensemble_pipeline.V2, workers))
    table.insert(1, 'job', [json.dumps(jobs[key]) for key in table.key])
    table.to_csv(path, index=False)
    print(f'{stage}: {len(jobs)} jobs in {(time.time() - started) / 60:.1f} min', flush=True)
    print(table.sort_values('mAP', ascending=False)[['key', 'mAP', 'AP50', 'APs']].head(6).round(4).to_string(index=False),
          flush=True)
    return table


def tune():
    """Fusion weight / WBF IoU, then the temporal grid, on the tuning frames; the choices go to tuning/selected.json."""
    retina_input = dict(file=retina_pipeline.retina_file('tune', retina_pipeline.retina_fusion_settings()))
    jobs = {'v2 only': dict(frcnn=True), 'RetinaNet only': dict(fcos=retina_input)}
    for weight in (0.25, 0.5, 0.75, 1.0):
        for iou in (0.55, 0.65):
            jobs[f'w_retina {weight} iou {iou}'] = dict(fcos=retina_input, frcnn=True, fusion=dict(weight_fcos=weight, iou=iou))
    table = run('g_weight_v2', jobs)
    fusion = retina_pipeline.best_job(table, ('w_retina',))['fusion']
    fusion_job = dict(fcos=retina_input, frcnn=True, fusion=fusion)
    jobs = {'fusion, no temporal': fusion_job,
            'v2 + its own temporal (pipeline A, reference)': dict(frcnn=True, temporal=ensemble_pipeline.settings()['v2_temporal'])}
    jobs.update({key: {**fusion_job, 'temporal': params} for key, params in pipeline_tuning.temporal_grid().items()})
    table = run('g_temporal_v2', jobs)
    retina_pipeline.select(v2_fusion=fusion,
                           v2_fusion_temporal=retina_pipeline.best_job(table, ('Faster R-CNN parameters', 's_high'))['temporal'])


def tuned_job(name):
    """The tuned RetinaNet + v2 fusion + temporal pipeline as a job on a frame set."""
    selected = retina_pipeline.settings()
    return dict(fcos=dict(file=retina_pipeline.retina_file(name, retina_pipeline.retina_fusion_settings())), frcnn=True,
                fusion=selected['v2_fusion'], temporal=selected['v2_fusion_temporal'])


def official():
    """One evaluation on the official validation set (cached) and the decision rule against pipeline A."""
    if not OFFICIAL_RAW_JSON.exists():
        results = detector_fusion.evaluate_jobs('val', {FUSION_KEY: tuned_job('val')}, None, (), ensemble_pipeline.V2, 1)
        OFFICIAL_RAW_JSON.write_text(json.dumps(dict(
            job=tuned_job('val'), selected=retina_pipeline.settings(),
            results={key: dict(metrics=metrics, seconds_per_frame=seconds) for key, (metrics, seconds) in results.items()}),
            indent=2))
    fusion_metrics = json.loads(OFFICIAL_RAW_JSON.read_text())['results'][FUSION_KEY]['metrics']
    a_metrics = json.loads(ensemble_pipeline.RAW.read_text())['results'][A_KEY]['metrics']
    videos = [group for group in fusion_metrics if group.startswith('video_')]
    wins = int(sum(fusion_metrics[video]['mAP'] > a_metrics[video]['mAP'] for video in videos))
    gain = fusion_metrics['all']['mAP'] - a_metrics['all']['mAP']
    fusion_decision = dict(mAP=fusion_metrics['all']['mAP'], reference_A=a_metrics['all']['mAP'], gain=gain,
                           videos_improved=wins, videos=len(videos),
                           per_video={video: (round(fusion_metrics[video]['mAP'], 4), round(a_metrics[video]['mAP'], 4))
                                      for video in videos},
                           adopt=bool(gain >= 0.002 and wins >= 5))
    (HERE / 'retina_v2_fusion_decision.json').write_text(json.dumps(fusion_decision, indent=2))
    return fusion_decision


if __name__ == '__main__':
    tune()
    print(json.dumps({key: value for key, value in retina_pipeline.settings().items() if key.startswith('v2_')}, indent=2))
    fusion_decision = official()
    print(json.dumps(fusion_decision, indent=2))
    if fusion_decision['adopt']:
        images = pd.read_csv(experiment_utils.DATA / 'test_images.csv')
        meta, rows = detector_fusion.pipeline_rows('test', tuned_job('test'), None, [], ensemble_pipeline.V2)
        assert meta.image_id.astype(str).tolist() == images.image_id.astype(str).tolist()
        strings = [tta_wbf.prediction_string(boxes, scores, labels, h, w)
                   for (boxes, scores, labels), h, w in zip(rows, meta.height, meta.width)]
        tmp_path = OUT_CSV.with_suffix('.tmp.csv')
        pd.DataFrame({'image_id': images.image_id, 'PredictionString': strings}).to_csv(tmp_path, index=False)
        os.replace(tmp_path, OUT_CSV)
        checks, stats = tta_wbf.validate_submission(OUT_CSV, images)
        print('CSV:', OUT_CSV.name, 'ALL PASS' if all(checks.values()) else checks, stats)
    else:
        print('no CSV: rule not met')
