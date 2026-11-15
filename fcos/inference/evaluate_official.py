"""
One evaluation of the FCOS inference pipelines on the official validation set.

The following pipelines are compared:
- Each FCOS checkpoint with its original predictions and optimised TTA.
- The 5-fold FCOS ensemble without TTA, with TTA, and with temporal post-processing
  (using Faster R-CNN parameters or parameters tuned specifically for FCOS).
- Faster R-CNN + FCOS fusion with and without temporal post-processing.
  An additional evaluation is performed with maxDets = 500.
"""

import json
import sys
import time
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'shared'))
import detector_fusion
import pipeline_tuning

RAW = detector_fusion.HERE / 'official_validation_raw.json'
CSV = detector_fusion.HERE / 'official_validation_results.csv'


def jobs():
    """{row name: pipeline job} for every evaluated pipeline, plus the FCOS config and the views passed to evaluate_jobs."""
    selected = pipeline_tuning.settings()
    config, views = selected['config'], selected['views']
    fcos_ensemble_file = detector_fusion.precompute_fcos('val', pipeline_tuning.fcos_job(views))
    fcos_ensemble = dict(file=fcos_ensemble_file)
    fused = dict(fcos={**fcos_ensemble, 'power': selected.get('fusion_power', 1.0)}, frcnn=True, fusion=selected['fusion'])
    job_specs = {}
    for fold in detector_fusion.FOLDS:
        job_specs[f'FCOS {config} fold {fold}: original (none@1.0, score>=0.02, 300 det)'] = dict(
            fcos=dict(configs=[config], views=['none@1.0'], score_min=0.02, folds=[fold]), top=300)
        job_specs[f'FCOS {config} fold {fold}: optimized TTA'] = dict(
            fcos=dict(file=detector_fusion.precompute_fcos('val', {**pipeline_tuning.fcos_job(views), 'folds': [fold]})))
    job_specs['FCOS 5 checkpoints, no TTA (WBF)'] = dict(
        fcos=dict(file=detector_fusion.precompute_fcos('val', pipeline_tuning.fcos_job(['none@1.0']))))
    job_specs['FCOS 5 checkpoints, optimized TTA'] = dict(fcos=fcos_ensemble)
    job_specs['FCOS 5 checkpoints, optimized TTA + temporal (Faster R-CNN parameters)'] = dict(
        fcos=fcos_ensemble, temporal=detector_fusion.FASTER_RCNN_TEMPORAL)
    job_specs['FCOS 5 checkpoints, optimized TTA + temporal (tuned for FCOS)'] = dict(
        fcos=fcos_ensemble, temporal=selected['fcos_temporal'])
    job_specs['Faster R-CNN 5 checkpoints TTA (0.2587 pipeline)'] = dict(frcnn=True)
    job_specs['Faster R-CNN + temporal (0.2810 pipeline)'] = dict(frcnn=True, temporal=detector_fusion.FASTER_RCNN_TEMPORAL)
    job_specs['Faster R-CNN + FCOS fusion'] = fused
    job_specs['Faster R-CNN + FCOS fusion + temporal (Faster R-CNN parameters)'] = {**fused, 'temporal': detector_fusion.FASTER_RCNN_TEMPORAL}
    job_specs['Faster R-CNN + FCOS fusion + temporal (tuned)'] = {**fused, 'temporal': selected['fusion_temporal']}
    job_specs['Faster R-CNN + FCOS fusion + temporal (tuned), maxDets 500'] = {
        **fused, 'temporal': selected['fusion_temporal'], 'max_dets_eval': 500}
    job_specs['Faster R-CNN + temporal (0.2810 pipeline), maxDets 500'] = dict(
        frcnn=True, temporal=detector_fusion.FASTER_RCNN_TEMPORAL, max_dets_eval=500)
    return job_specs, config, ['none@1.0']


def evaluate(workers=4):
    """Results table of all pipelines (evaluated once, then read from official_validation_raw.json) and the raw results."""
    if RAW.exists():
        raw = json.loads(RAW.read_text())
    else:
        job_specs, config, views = jobs()
        start = time.time()
        results = detector_fusion.evaluate_jobs('val', job_specs, config, views, True, workers)
        raw = dict(jobs=job_specs,
                   results={key: dict(metrics=metrics, seconds_per_frame=seconds)
                            for key, (metrics, seconds) in results.items()},
                   minutes=(time.time() - start) / 60,
                   selected=pipeline_tuning.settings())
        RAW.write_text(json.dumps(raw, indent=2))
    table = detector_fusion.table({key: (result['metrics'], result['seconds_per_frame']) for key, result in raw['results'].items()})
    table.to_csv(CSV, index=False)
    return table, raw


def decision(df):
    """Apply the decision rule above to the results table `df` returned by evaluate()."""
    reference = df.set_index('key').loc['Faster R-CNN + temporal (0.2810 pipeline)']
    fusion = df.set_index('key').loc['Faster R-CNN + FCOS fusion + temporal (tuned)']
    videos = [column for column in df.columns if column.startswith('mAP_video_')]
    videos_improved = int(sum(fusion[video] > reference[video] for video in videos))
    gain = float(fusion.mAP - reference.mAP)
    return dict(gain=gain, videos_improved=videos_improved, videos=len(videos),
                adopt=bool(gain >= 0.002 and videos_improved >= 5))


if __name__ == '__main__':
    table, raw = evaluate()
    print(table[['key', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR', 'ms_per_frame']].round(4).to_string(index=False))
    print(decision(table))
