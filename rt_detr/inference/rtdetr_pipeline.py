"""
RT-DETR tuning and official evaluation using saved predictions.

Main stages:
1. WBF: Tune IoU thresholds (0.55, 0.6, 0.65) and test NMS.
2. TTA: Select the best combination of scales and horizontal flipping.
3. Score: Tune confidence thresholds (0.01, 0.02, 0.05) and score aggregation methods.
4. Temporal processing: Tune temporal post-processing for RT-DETR predictions.
5. Fusion: Combine RT-DETR with the best Faster R-CNN v2 + FCOS ensemble and tune model weights.
6. Fusion temporal: Tune temporal post-processing for the combined ensemble.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
FCOS_DIR = ROOT / 'fcos/inference'
V2_DIR = ROOT / 'final_pipeline/ensemble'
sys.path.insert(0, str(ROOT / 'shared'))
sys.path.insert(0, str(V2_DIR))

import detector_fusion
import pipeline_tuning
import experiment_utils
import tta_wbf
import ensemble_pipeline


VIEWS_DIR = HERE / 'cache' / 'views'
FUSED = HERE / 'cache' / 'fused'
B_DIR = HERE / 'cache' / 'b_fused'
TUNING = HERE / 'tuning'
RAW = HERE / 'official_validation_raw.json'
ALL_VIEWS = ['none@1.0', 'h@1.0', 'none@1.25', 'h@1.25', 'none@1.5', 'h@1.5', 'none@0.75', 'h@0.75']
B_KEY = 'B: v2 + FCOS fusion + temporal (tuned)'
MARGIN, MIN_VIDEOS = 0.002, 5
FUSION = os.environ.get('RTDETR_FUSION') == '1' or (HERE / 'FUSION_ON').exists()

def _fuse_one(task_args):
    """Fuse the views of one frame (run in a worker process)."""
    import warnings
    warnings.filterwarnings('ignore')
    view_parts, height, width, job = task_args
    return detector_fusion.fuse_frame(view_parts, None, height, width, job)

def model_file(name, rtdetr_job, workers=8):
    """rtdetr_job: dict(views, score_min, pre_nms, wbf_iou, conf_type). tune: 1 checkpoint x views (OOF); val/test: 5 x views."""
    job_hash = hashlib.sha256(json.dumps(rtdetr_job, sort_keys=True).encode()).hexdigest()[:12]
    path = FUSED / f'rtdetr_{name}_{job_hash}.npz'
    if path.exists():
        return str(path)
    FUSED.mkdir(parents=True, exist_ok=True)
    meta = experiment_utils.frames_meta(name)
    image_ids = meta.image_id.astype(str)
    job = dict(fcos={key: value for key, value in rtdetr_job.items() if key != 'views'})
    start_time = time.time()
    if name == 'tune':
        view_sets = [[None] * len(meta) for _ in rtdetr_job['views']]
        for fold in detector_fusion.FOLDS:
            frame_positions = np.where(meta.fold.values == fold)[0]
            for view_idx, view in enumerate(rtdetr_job['views']):
                for frame_idx, view_result in zip(frame_positions, detector_fusion._chunks(VIEWS_DIR / name / f'fold{fold}', view, image_ids.values[frame_positions].tolist())):
                    view_sets[view_idx][frame_idx] = view_result
    else:
        view_sets = [detector_fusion._chunks(VIEWS_DIR / name / f'fold{fold}', view, image_ids.tolist()) for fold in detector_fusion.FOLDS for view in rtdetr_job['views']]
    tasks = (([view_set[frame_idx] for view_set in view_sets], height, width, job) for frame_idx, (height, width) in enumerate(zip(meta.height, meta.width)))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_fuse_one, tasks, chunksize=8))
    experiment_utils.save_preds(path, meta.drop(columns='path'), rows)
    path.with_suffix('.json').write_text(json.dumps(dict(job=rtdetr_job, sets=len(view_sets), minutes=(time.time() - start_time) / 60)))
    print(f'{path.name}: {len(view_sets)} sets per frame in {(time.time() - start_time) / 60:.1f} min', flush=True)
    return str(path)

def make_rtdetr_job(**overrides):
    """Job settings from tuning/selected.json (defaults where nothing is selected yet), updated with the keyword overrides."""
    selected = settings()
    job = dict(views=selected.get('views', ALL_VIEWS), score_min=selected.get('score_min', 0.01), pre_nms=selected.get('pre_nms'),
               wbf_iou=selected.get('wbf_iou', 0.55), conf_type=selected.get('conf_type', 'avg'))
    job.update(overrides)
    return job

def build_b(name):
    """Cache the v2 + FCOS ensemble fusion without its temporal step ("B") for the frame set `name`."""
    path = B_DIR / {'tune': 'tune_base', 'val': 'val_ens', 'test': 'test_ens'}[name]
    if path.with_suffix('.npz').exists():
        return
    B_DIR.mkdir(parents=True, exist_ok=True)
    v2_settings = ensemble_pipeline.settings()
    job = dict(fcos=dict(file=ensemble_pipeline.fcos_file(name), power=v2_settings['fusion_power']), frcnn=True, fusion=v2_settings['fusion'])
    meta, rows = detector_fusion.pipeline_rows(name, job, None, [], ensemble_pipeline.V2)
    experiment_utils.save_preds(path.with_suffix('.npz'), meta.drop(columns='path'), rows)
    print(f'v2/FCOS ensemble fusion ({name}) cached', flush=True)

def settings():
    """Settings selected so far (tuning/selected.json), {} if none."""
    selected_path = TUNING / 'selected.json'
    return json.loads(selected_path.read_text()) if selected_path.exists() else {}

def select(**changes):
    """Store the selected settings (merged into tuning/selected.json)."""
    TUNING.mkdir(exist_ok=True)
    (TUNING / 'selected.json').write_text(json.dumps({**settings(), **changes}, indent=2))

def run(stage, jobs, with_b=False, workers=6):
    """Evaluate the jobs on the tuning frames (cached as tuning/<stage>.csv) and return the results table."""
    path = TUNING / f'{stage}.csv'
    if path.exists():
        return pd.read_csv(path)
    TUNING.mkdir(exist_ok=True)
    start_time = time.time()
    table = detector_fusion.table(detector_fusion.evaluate_jobs('tune', jobs, None, (), str(B_DIR) if with_b else False, workers))
    table.insert(1, 'job', [json.dumps(jobs[job_key]) for job_key in table.key])
    table.to_csv(path, index=False)
    print(f'{stage}: {len(jobs)} jobs in {(time.time() - start_time) / 60:.1f} min', flush=True)
    print(table.sort_values('mAP', ascending=False)[['key', 'mAP', 'AP50', 'APs']].head(6).round(4).to_string(index=False), flush=True)
    return table

def best_job(table, prefixes=None):
    """The job with the highest mAP in the results table (optionally only rows whose key starts with `prefixes`)."""
    subset = table if prefixes is None else table[table.key.str.startswith(prefixes)]
    return json.loads(subset.loc[subset.mAP.idxmax(), 'job'])

def file_job(rtdetr_job, **extra):
    """Pipeline job whose detections are the fused RT-DETR views of the tuning frames."""
    return dict(fcos=dict(file=model_file('tune', rtdetr_job)), **extra)

def tune():
    """All tuning stages (r_wbf, r_views, r_score, r_temporal and, with RTDETR_FUSION=1, f_weight, f_temporal), each selecting its best setting."""
    jobs, files = {}, {}
    jobs['single view none@1.0 (trained setting)'] = file_job(make_rtdetr_job(views=['none@1.0']))
    for wbf_iou in (0.55, 0.6, 0.65):
        jobs[f'all views: WBF {wbf_iou}'] = file_job(make_rtdetr_job(views=ALL_VIEWS, wbf_iou=wbf_iou))
        jobs[f'all views: NMS 0.5 per view + WBF {wbf_iou}'] = file_job(make_rtdetr_job(views=ALL_VIEWS, wbf_iou=wbf_iou, pre_nms=0.5))
    table = run('r_wbf', jobs)
    fused_entry = json.loads(table[table.key.str.startswith('all views')].sort_values('mAP').iloc[-1].job)['fcos']
    saved_job = json.loads(Path(fused_entry['file']).with_suffix('.json').read_text())['job']
    select(wbf_iou=saved_job['wbf_iou'], pre_nms=saved_job['pre_nms'])

    scale_sets = [['1.0'], ['0.75', '1.0'], ['1.0', '1.25'], ['1.0', '1.5'], ['0.75', '1.0', '1.25'], ['1.0', '1.25', '1.5'],
                  ['0.75', '1.0', '1.25', '1.5']]
    jobs = {}
    for scales in scale_sets:
        jobs['flip+none @ ' + '+'.join(scales)] = file_job(make_rtdetr_job(views=[f'{flip}@{scale}' for scale in scales for flip in ('none', 'h')]))
    jobs['none @ 0.75+1.0+1.25+1.5'] = file_job(make_rtdetr_job(views=[f'none@{scale}' for scale in ('0.75', '1.0', '1.25', '1.5')]))
    table = run('r_views', jobs)
    best_saved_job = json.loads(Path(best_job(table)['fcos']['file']).with_suffix('.json').read_text())['job']
    select(views=best_saved_job['views'])

    jobs = {f'score>={score_floor}': file_job(make_rtdetr_job(score_min=score_floor)) for score_floor in (0.01, 0.02, 0.05)}
    jobs['conf max'] = file_job(make_rtdetr_job(conf_type='max'))
    table = run('r_score', jobs)
    best_saved_job = json.loads(Path(best_job(table)['fcos']['file']).with_suffix('.json').read_text())['job']
    select(score_min=best_saved_job['score_min'], conf_type=best_saved_job['conf_type'])

    fused_entry = dict(file=model_file('tune', make_rtdetr_job()))
    grid = pipeline_tuning.temporal_grid()
    jobs = {'RT-DETR TTA, no temporal': dict(fcos=fused_entry), **{grid_key: dict(fcos=fused_entry, temporal=temporal_params) for grid_key, temporal_params in grid.items()}}
    table = run('r_temporal', jobs)
    select(model_temporal=best_job(table, ('Faster R-CNN parameters', 's_high'))['temporal'])
    if not FUSION:
        print(json.dumps(settings(), indent=2))
        return

    build_b('tune')
    jobs = {'v2/FCOS ensemble only': dict(frcnn=True), 'RT-DETR only': dict(fcos=fused_entry)}
    for rtdetr_weight in (0.25, 0.5, 0.75, 1.0):
        for fusion_iou in (0.55, 0.65):
            jobs[f'w_rtdetr {rtdetr_weight} iou {fusion_iou}'] = dict(fcos=fused_entry, frcnn=True, fusion=dict(weight_fcos=rtdetr_weight, iou=fusion_iou))
    table = run('f_weight', jobs, with_b=True)
    select(fusion=best_job(table, ('w_rtdetr',))['fusion'])

    fusion_base = dict(fcos=fused_entry, frcnn=True, fusion=settings()['fusion'])
    b_temporal = ensemble_pipeline.settings()['fusion_temporal']
    jobs = {'fusion, no temporal': fusion_base, 'v2/FCOS ensemble + its own temporal (reference)': dict(frcnn=True, temporal=b_temporal)}
    jobs.update({grid_key: {**fusion_base, 'temporal': temporal_params} for grid_key, temporal_params in grid.items()})
    table = run('f_temporal', jobs, with_b=True)
    select(fusion_temporal=best_job(table, ('Faster R-CNN parameters', 's_high'))['temporal'])
    print(json.dumps(settings(), indent=2))

def candidates(name):
    """The final pipelines (RT-DETR alone; with RTDETR_FUSION=1 also the fusion with the v2 + FCOS ensemble) for the frame set `name`."""
    selected = settings()
    fused_entry = dict(file=model_file(name, make_rtdetr_job()))
    candidate_jobs = {'RT-DETR 5-fold TTA + WBF + temporal (tuned)': dict(fcos=fused_entry, temporal=selected['model_temporal'])}
    if FUSION:
        candidate_jobs['RT-DETR + v2/FCOS ensemble fusion + temporal (tuned)'] = dict(fcos=fused_entry, frcnn=True, fusion=selected['fusion'], temporal=selected['fusion_temporal'])
    return candidate_jobs

def official(workers=4):
    """Evaluate the pipelines on the official validation set once (raw results cached in official_validation_raw.json); returns the table."""
    if not RAW.exists():
        if FUSION:
            build_b('val')
        selected = settings()
        fused_entry = dict(file=model_file('val', make_rtdetr_job()))
        jobs = {'RT-DETR 5 checkpoints, single view (trained setting)': dict(fcos=dict(file=model_file('val', make_rtdetr_job(views=['none@1.0'])))),
                'RT-DETR 5 checkpoints TTA + WBF': dict(fcos=fused_entry),
                **({'RT-DETR + v2/FCOS ensemble fusion (no temporal)': dict(fcos=fused_entry, frcnn=True, fusion=selected['fusion'])} if FUSION else {}),
                **candidates('val')}
        start_time = time.time()
        evaluation_results = detector_fusion.evaluate_jobs('val', jobs, None, (), str(B_DIR) if FUSION else False, workers)
        RAW.write_text(json.dumps(dict(jobs=jobs, selected=selected, minutes=(time.time() - start_time) / 60,
                                       results={job_name: dict(metrics=metrics, seconds_per_frame=seconds) for job_name, (metrics, seconds) in evaluation_results.items()}), indent=2))
    raw_results = json.loads(RAW.read_text())
    reference_raw = json.loads(ensemble_pipeline.RAW.read_text())['results']
    reference_labels = {'A: v2 + temporal (tuned)': 'Faster R-CNN v2 + temporal (Kaggle 0.3188)', B_KEY: 'Faster R-CNN v2 + FCOS + temporal (best validated so far)'}
    evaluation_results = {reference_labels[reference_key]: (reference_raw[reference_key]['metrics'], reference_raw[reference_key]['seconds_per_frame']) for reference_key in reference_labels}
    evaluation_results.update({job_name: (result['metrics'], result['seconds_per_frame']) for job_name, result in raw_results['results'].items()})
    table = detector_fusion.table(evaluation_results)
    table.to_csv(HERE / 'official_validation_results.csv', index=False)
    return table

def decision(official_table):
    """Decision rule: adopt the fusion if its official mAP is >= MARGIN above B and it wins on >= MIN_VIDEOS videos."""
    if not FUSION:
        return dict(adopt=False, note='fusion not evaluated (RTDETR_FUSION not set)')
    by_key = official_table.set_index('key')
    reference_row = by_key.loc['Faster R-CNN v2 + FCOS + temporal (best validated so far)']
    videos = [column for column in official_table.columns if column.startswith('mAP_video_')]
    fusion_row = by_key.loc['RT-DETR + v2/FCOS ensemble fusion + temporal (tuned)']
    wins = int(sum(fusion_row[video] > reference_row[video] for video in videos))
    return dict(mAP=float(fusion_row.mAP), reference_B=float(reference_row.mAP), gain=float(fusion_row.mAP - reference_row.mAP), videos_improved=wins, videos=len(videos),
                adopt=bool(fusion_row.mAP - reference_row.mAP >= MARGIN and wins >= MIN_VIDEOS))

def submit():
    """Write the Kaggle CSVs (RT-DETR alone; the fusion only if the decision rule is met), validate them; nothing is uploaded."""
    official_table = official()
    decision_result = decision(official_table)
    test_images = pd.read_csv(experiment_utils.DATA / 'test_images.csv')
    if FUSION:
        build_b('test')
    jobs = candidates('test')
    submission_files = {'RT-DETR 5-fold TTA + WBF + temporal (tuned)': 'rtdetr_5fold_tta_wbf_temporal_submission.csv'}
    if decision_result['adopt']:
        submission_files['RT-DETR + v2/FCOS ensemble fusion + temporal (tuned)'] = 'rtdetr_v2_fcos_5fold_tta_wbf_fusion_temporal_submission.csv'
    else:
        print('no CSV for the ensemble fusion (rule not met):', decision_result)
    reports = {}
    for key, fname in submission_files.items():
        path = ROOT / 'kaggle_submissions' / fname
        meta, rows = detector_fusion.pipeline_rows('test', jobs[key], None, [], str(B_DIR) if 'ensemble fusion' in key else False)
        assert meta.image_id.astype(str).tolist() == test_images.image_id.astype(str).tolist()
        strings = [tta_wbf.prediction_string(boxes, scores, labels, height, width)
                   for (boxes, scores, labels), height, width in zip(rows, meta.height, meta.width)]
        tmp_path = path.with_suffix('.tmp.csv')
        pd.DataFrame({'image_id': test_images.image_id, 'PredictionString': strings}).to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        checks, stats = tta_wbf.validate_submission(path, test_images)
        reports[key] = dict(submission=str(path), pipeline=jobs[key], checks=checks, stats=stats)
        print(key, '->', fname, '| checks:', 'ALL PASS' if all(checks.values()) else checks, flush=True)
    (ROOT / 'kaggle_submissions' / 'rtdetr_submission_report.json').write_text(json.dumps(dict(decision_ensemble_fusion=decision_result, submissions=reports,
        note='validated against test_images.csv and the Data-page format; nothing uploaded'), indent=2, default=str))

if __name__ == '__main__':
    for step in sys.argv[1:]:
        if step == 'tune':
            tune()
        elif step == 'official':
            table = official()
            columns = ['key', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR'] + [column for column in table.columns if column.startswith('mAP_video_')]
            print(table[columns].round(4).to_string(index=False))
            print(json.dumps(decision(table), indent=2))
        elif step == 'submit':
            submit()
