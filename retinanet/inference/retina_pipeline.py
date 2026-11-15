"""
Tuning, evaluation, and Kaggle CSV generation for RetinaNet using predictions
from retinanet/inference/predict_retina_views.py.

Stages:
- tune: tune all parameters on out-of-fold frames (56 videos, 80 frames each).
- official: evaluate once on the official validation set.
- submit: generate Kaggle CSV files without automatically uploading them.

The best configuration is selected at each stage:
- r_wbf: tune WBF IoU (0.55, 0.6, 0.65) and test class-wise NMS 0.5
  for each TTA view.
- r_views: select the best combination of resizing scales and horizontal flipping.
- r_score: tune the confidence score threshold (0.01, 0.02, 0.05)
  and confidence aggregation method (average or maximum).
- r_temporal: tune temporal post-processing parameters for RetinaNet.
- f_weight: tune the RetinaNet weight (0.25, 0.5, 0.75, 1.0) and WBF IoU
  (0.55, 0.65) when combining it with Faster R-CNN v2 + FCOS
  before temporal post-processing.
- f_temporal: tune temporal post-processing parameters for the best ensemble.
"""

import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[2]))
FCOS_DIR = ROOT / 'fcos/inference'
V2_DIR = ROOT / 'faster_rcnn/faster_rcnn_v2/inference'
sys.path.insert(0, str(ROOT / 'shared'))
sys.path.insert(0, str(V2_DIR))

import detector_fusion
import pipeline_tuning
import experiment_utils
import tta_wbf
import ensemble_pipeline

VIEWS_DIR = HERE / 'cache' / 'views'
FUSED_DIR = HERE / 'cache' / 'fused'
B_DIR = HERE / 'cache' / 'b_fused'
TUNING = HERE / 'tuning'
OFFICIAL_RAW_JSON = HERE / 'official_validation_raw.json'
ALL_VIEWS = ['none@1.0', 'h@1.0', 'none@1.25', 'h@1.25', 'none@1.5', 'h@1.5', 'none@0.75', 'h@0.75']
B_KEY = 'B: v2 + FCOS fusion + temporal (tuned)'
MARGIN, MIN_VIDEOS = 0.002, 5


def _fuse_one(task):
    """Fuse the predictions of all (checkpoint, view) sets of one frame (runs in a worker process)."""
    import warnings
    warnings.filterwarnings('ignore')
    parts, h, w, job = task
    return detector_fusion.fuse_frame(parts, None, h, w, job)


def retina_file(name, fusion_settings, workers=8):
    """fusion_settings: dict(views, score_min, pre_nms, wbf_iou, conf_type).
    tune: 1 checkpoint x views (OOF); val/test: 5 x views."""
    settings_hash = hashlib.sha256(json.dumps(fusion_settings, sort_keys=True).encode()).hexdigest()[:12]
    path = FUSED_DIR / f'retina_{name}_{settings_hash}.npz'
    if path.exists():
        return str(path)
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    meta = experiment_utils.frames_meta(name)
    ids = meta.image_id.astype(str)
    job = dict(fcos={key: value for key, value in fusion_settings.items() if key != 'views'})
    started = time.time()
    if name == 'tune':
        prediction_sets = [[None] * len(meta) for _ in fusion_settings['views']]
        for fold in detector_fusion.FOLDS:
            positions = np.where(meta.fold.values == fold)[0]
            for j, view in enumerate(fusion_settings['views']):
                for i, pred in zip(positions,
                                   detector_fusion._chunks(VIEWS_DIR / name / f'fold{fold}', view, ids.values[positions].tolist())):
                    prediction_sets[j][i] = pred
    else:
        prediction_sets = [detector_fusion._chunks(VIEWS_DIR / name / f'fold{fold}', view, ids.tolist())
                           for fold in detector_fusion.FOLDS for view in fusion_settings['views']]
    tasks = (([prediction_set[i] for prediction_set in prediction_sets], h, w, job)
             for i, (h, w) in enumerate(zip(meta.height, meta.width)))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_fuse_one, tasks, chunksize=8))
    experiment_utils.save_preds(path, meta.drop(columns='path'), rows)
    path.with_suffix('.json').write_text(json.dumps(dict(job=fusion_settings, sets=len(prediction_sets),
                                                         minutes=(time.time() - started) / 60)))
    print(f'{path.name}: {len(prediction_sets)} sets per frame in {(time.time() - started) / 60:.1f} min', flush=True)
    return str(path)


def retina_fusion_settings(**overrides):
    """RetinaNet view-fusion settings: the ones selected so far (tuning/selected.json, else the defaults), then `overrides`."""
    selected = settings()
    fusion_settings = dict(views=selected.get('views', ALL_VIEWS), score_min=selected.get('score_min', 0.01),
                           pre_nms=selected.get('pre_nms'), wbf_iou=selected.get('wbf_iou', 0.55),
                           conf_type=selected.get('conf_type', 'avg'))
    fusion_settings.update(overrides)
    return fusion_settings


def build_b_fusion(name):
    """Cache the v2 + FCOS ensemble fusion (pipeline B without its temporal step) of a frame set."""
    path = B_DIR / {'tune': 'tune_base', 'val': 'val_ens', 'test': 'test_ens'}[name]
    if path.with_suffix('.npz').exists():
        return
    B_DIR.mkdir(parents=True, exist_ok=True)
    b_settings = ensemble_pipeline.settings()
    job = dict(fcos=dict(file=ensemble_pipeline.fcos_file(name), power=b_settings['fusion_power']), frcnn=True,
               fusion=b_settings['fusion'])
    meta, rows = detector_fusion.pipeline_rows(name, job, None, [], ensemble_pipeline.V2)
    experiment_utils.save_preds(path.with_suffix('.npz'), meta.drop(columns='path'), rows)
    print(f'v2/FCOS ensemble fusion ({name}) cached', flush=True)


def settings():
    """The choices made so far on the tuning frames (tuning/selected.json)."""
    path = TUNING / 'selected.json'
    return json.loads(path.read_text()) if path.exists() else {}


def select(**choices):
    """Add `choices` to tuning/selected.json."""
    TUNING.mkdir(exist_ok=True)
    (TUNING / 'selected.json').write_text(json.dumps({**settings(), **choices}, indent=2))


def run(stage, jobs, with_b=False, workers=6):
    """Evaluate the jobs of one stage on the tuning frames (cached as tuning/<stage>.csv)."""
    path = TUNING / f'{stage}.csv'
    if path.exists():
        return pd.read_csv(path)
    TUNING.mkdir(exist_ok=True)
    started = time.time()
    table = detector_fusion.table(detector_fusion.evaluate_jobs('tune', jobs, None, (), str(B_DIR) if with_b else False, workers))
    table.insert(1, 'job', [json.dumps(jobs[key]) for key in table.key])
    table.to_csv(path, index=False)
    print(f'{stage}: {len(jobs)} jobs in {(time.time() - started) / 60:.1f} min', flush=True)
    print(table.sort_values('mAP', ascending=False)[['key', 'mAP', 'AP50', 'APs']].head(6).round(4).to_string(index=False),
          flush=True)
    return table


def best_job(table, prefixes=None):
    """The job with the highest tuning mAP (optionally only among keys starting with `prefixes`)."""
    subset = table if prefixes is None else table[table.key.str.startswith(prefixes)]
    return json.loads(subset.loc[subset.mAP.idxmax(), 'job'])


def file_job(fusion_settings, **extra):
    """A pipeline job whose RetinaNet input is the fused file of `fusion_settings` on the tuning frames."""
    return dict(fcos=dict(file=retina_file('tune', fusion_settings)), **extra)


def tune():
    """All tuning stages in order; each stage adds its choice to tuning/selected.json."""
    jobs, files = {}, {}
    jobs['single view none@1.0 (trained setting)'] = file_job(retina_fusion_settings(views=['none@1.0']))
    for wbf_iou in (0.55, 0.6, 0.65):
        jobs[f'all views: WBF {wbf_iou}'] = file_job(retina_fusion_settings(views=ALL_VIEWS, wbf_iou=wbf_iou))
        jobs[f'all views: NMS 0.5 per view + WBF {wbf_iou}'] = file_job(
            retina_fusion_settings(views=ALL_VIEWS, wbf_iou=wbf_iou, pre_nms=0.5))
    table = run('r_wbf', jobs)
    best_input = json.loads(table[table.key.str.startswith('all views')].sort_values('mAP').iloc[-1].job)['fcos']
    best_settings = json.loads(Path(best_input['file']).with_suffix('.json').read_text())['job']
    select(wbf_iou=best_settings['wbf_iou'], pre_nms=best_settings['pre_nms'])

    scale_sets = [['1.0'], ['0.75', '1.0'], ['1.0', '1.25'], ['1.0', '1.5'], ['0.75', '1.0', '1.25'], ['1.0', '1.25', '1.5'],
                  ['0.75', '1.0', '1.25', '1.5']]
    jobs = {}
    for scales in scale_sets:
        jobs['flip+none @ ' + '+'.join(scales)] = file_job(
            retina_fusion_settings(views=[f'{flip}@{scale}' for scale in scales for flip in ('none', 'h')]))
    jobs['none @ 0.75+1.0+1.25+1.5'] = file_job(
        retina_fusion_settings(views=[f'none@{scale}' for scale in ('0.75', '1.0', '1.25', '1.5')]))
    table = run('r_views', jobs)
    best = json.loads(Path(best_job(table)['fcos']['file']).with_suffix('.json').read_text())['job']
    select(views=best['views'])

    jobs = {f'score>={score_min}': file_job(retina_fusion_settings(score_min=score_min)) for score_min in (0.01, 0.02, 0.05)}
    jobs['conf max'] = file_job(retina_fusion_settings(conf_type='max'))
    table = run('r_score', jobs)
    best = json.loads(Path(best_job(table)['fcos']['file']).with_suffix('.json').read_text())['job']
    select(score_min=best['score_min'], conf_type=best['conf_type'])

    retina_input = dict(file=retina_file('tune', retina_fusion_settings()))
    temporal_grid = pipeline_tuning.temporal_grid()
    jobs = {'RetinaNet TTA, no temporal': dict(fcos=retina_input),
            **{key: dict(fcos=retina_input, temporal=params) for key, params in temporal_grid.items()}}
    table = run('r_temporal', jobs)
    select(retina_temporal=best_job(table, ('Faster R-CNN parameters', 's_high'))['temporal'])

    build_b_fusion('tune')
    jobs = {'v2/FCOS ensemble only': dict(frcnn=True), 'RetinaNet only': dict(fcos=retina_input)}
    for weight in (0.25, 0.5, 0.75, 1.0):
        for iou in (0.55, 0.65):
            jobs[f'w_retina {weight} iou {iou}'] = dict(fcos=retina_input, frcnn=True, fusion=dict(weight_fcos=weight, iou=iou))
    table = run('f_weight', jobs, with_b=True)
    select(fusion=best_job(table, ('w_retina',))['fusion'])

    fusion_job = dict(fcos=retina_input, frcnn=True, fusion=settings()['fusion'])
    b_temporal = ensemble_pipeline.settings()['fusion_temporal']
    jobs = {'fusion, no temporal': fusion_job,
            'v2/FCOS ensemble + its own temporal (reference)': dict(frcnn=True, temporal=b_temporal)}
    jobs.update({key: {**fusion_job, 'temporal': params} for key, params in temporal_grid.items()})
    table = run('f_temporal', jobs, with_b=True)
    select(fusion_temporal=best_job(table, ('Faster R-CNN parameters', 's_high'))['temporal'])
    print(json.dumps(settings(), indent=2))


def candidates(name):
    """The two tuned pipelines (RetinaNet alone, RetinaNet + v2/FCOS fusion) as jobs on a frame set."""
    selected = settings()
    retina_input = dict(file=retina_file(name, retina_fusion_settings()))
    return {'RetinaNet 5-fold TTA + WBF + temporal (tuned)': dict(fcos=retina_input, temporal=selected['retina_temporal']),
            'RetinaNet + v2/FCOS ensemble fusion + temporal (tuned)': dict(fcos=retina_input, frcnn=True, fusion=selected['fusion'],
                                                                           temporal=selected['fusion_temporal'])}


def official(workers=4):
    """Evaluate the pipelines once on the official validation set (cached) and tabulate them next to pipelines A and B."""
    if not OFFICIAL_RAW_JSON.exists():
        build_b_fusion('val')
        selected = settings()
        retina_input = dict(file=retina_file('val', retina_fusion_settings()))
        jobs = {'RetinaNet 5 checkpoints, single view (trained setting)':
                    dict(fcos=dict(file=retina_file('val', retina_fusion_settings(views=['none@1.0'])))),
                'RetinaNet 5 checkpoints TTA + WBF': dict(fcos=retina_input),
                'RetinaNet + v2/FCOS ensemble fusion (no temporal)': dict(fcos=retina_input, frcnn=True, fusion=selected['fusion']),
                **candidates('val')}
        started = time.time()
        results = detector_fusion.evaluate_jobs('val', jobs, None, (), str(B_DIR), workers)
        OFFICIAL_RAW_JSON.write_text(json.dumps(dict(
            jobs=jobs, selected=selected, minutes=(time.time() - started) / 60,
            results={key: dict(metrics=metrics, seconds_per_frame=seconds) for key, (metrics, seconds) in results.items()}),
            indent=2))
    raw = json.loads(OFFICIAL_RAW_JSON.read_text())
    v2_results = json.loads(ensemble_pipeline.RAW.read_text())['results']
    labels = {'A: v2 + temporal (tuned)': 'Faster R-CNN v2 + temporal (Kaggle 0.3188)',
              B_KEY: 'Faster R-CNN v2 + FCOS + temporal (best validated so far)'}
    results = {labels[key]: (v2_results[key]['metrics'], v2_results[key]['seconds_per_frame']) for key in labels}
    results.update({key: (entry['metrics'], entry['seconds_per_frame']) for key, entry in raw['results'].items()})
    table = detector_fusion.table(results)
    table.to_csv(HERE / 'official_validation_results.csv', index=False)
    return table


def decision(table):
    """Decision rule for the three-model fusion: mAP gain >= MARGIN over B and better on >= MIN_VIDEOS videos."""
    by_key = table.set_index('key')
    reference = by_key.loc['Faster R-CNN v2 + FCOS + temporal (best validated so far)']
    video_columns = [column for column in table.columns if column.startswith('mAP_video_')]
    fusion = by_key.loc['RetinaNet + v2/FCOS ensemble fusion + temporal (tuned)']
    wins = int(sum(fusion[video] > reference[video] for video in video_columns))
    return dict(mAP=float(fusion.mAP), reference_B=float(reference.mAP), gain=float(fusion.mAP - reference.mAP),
                videos_improved=wins, videos=len(video_columns),
                adopt=bool(fusion.mAP - reference.mAP >= MARGIN and wins >= MIN_VIDEOS))


def submit():
    """Write the Kaggle CSVs (RetinaNet alone always; the fusion only if the decision rule is met) and validate them."""
    table = official()
    fusion_decision = decision(table)
    images = pd.read_csv(experiment_utils.DATA / 'test_images.csv')
    build_b_fusion('test')
    jobs = candidates('test')
    csv_names = {'RetinaNet 5-fold TTA + WBF + temporal (tuned)': 'retinanet_5fold_tta_wbf_temporal_submission.csv'}
    if fusion_decision['adopt']:
        csv_names['RetinaNet + v2/FCOS ensemble fusion + temporal (tuned)'] = \
            'retinanet_v2_fcos_5fold_tta_wbf_fusion_temporal_submission.csv'
    else:
        print('no CSV for the ensemble fusion (rule not met):', fusion_decision)
    reports = {}
    for key, csv_name in csv_names.items():
        path = HERE / 'kaggle_submissions' / csv_name
        meta, rows = detector_fusion.pipeline_rows('test', jobs[key], None, [], str(B_DIR) if 'ensemble fusion' in key else False)
        assert meta.image_id.astype(str).tolist() == images.image_id.astype(str).tolist()
        strings = [tta_wbf.prediction_string(boxes, scores, labels, h, w)
                   for (boxes, scores, labels), h, w in zip(rows, meta.height, meta.width)]
        tmp_path = path.with_suffix('.tmp.csv')
        pd.DataFrame({'image_id': images.image_id, 'PredictionString': strings}).to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        checks, stats = tta_wbf.validate_submission(path, images)
        reports[key] = dict(submission=str(path), pipeline=jobs[key], checks=checks, stats=stats)
        print(key, '->', csv_name, '| checks:', 'ALL PASS' if all(checks.values()) else checks, flush=True)
    (HERE / 'kaggle_submissions' / 'retinanet_submission_report.json').write_text(json.dumps(dict(
        decision_ensemble_fusion=fusion_decision, submissions=reports,
        note='validated against test_images.csv and the Data-page format; nothing uploaded'), indent=2, default=str))


if __name__ == '__main__':
    for step in sys.argv[1:]:
        if step == 'tune':
            tune()
        elif step == 'official':
            table = official()
            columns = (['key', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR'] +
                       [column for column in table.columns if column.startswith('mAP_video_')])
            print(table[columns].round(4).to_string(index=False))
            print(json.dumps(decision(table), indent=2))
        elif step == 'submit':
            submit()
