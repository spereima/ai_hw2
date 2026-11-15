"""
Runs the Faster R-CNN v2 pipeline using cached TTA predictions.
Applies WBF, tunes fusion and temporal post-processing parameters,
compares the results with v1 on official validation.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('UCU_HW2_ROOT', Path(__file__).resolve().parents[3]))
FCOS_DIR = ROOT / 'fcos/inference'
sys.path.insert(0, str(ROOT / 'shared'))
import detector_fusion
import pipeline_tuning
import experiment_utils
import tta_wbf

VIEWS = HERE / 'cache' / 'views'
FUSED = HERE / 'cache' / 'fused'
V2 = str(FUSED)
TUNING = HERE / 'tuning'
RAW = HERE / 'official_validation_raw.json'
V1_RAW = FCOS_DIR / 'official_validation_raw.json'
REF_KEY = 'Faster R-CNN + FCOS fusion + temporal (tuned)'
MARGIN, MIN_VIDEOS = 0.002, 5
CANDIDATES = ['A: v2 + temporal (tuned)', 'B: v2 + FCOS fusion + temporal (tuned)']


def view_rows(name, fold, ids):
    """Cached v2 predictions of every TTA view of one fold, aligned with the image ids `ids`."""
    return [
        detector_fusion._chunks(VIEWS / name / f'fold{fold}', view, ids)
        for view in tta_wbf.TTA_VIEWS]


def build_fused(name):
    """TTA + WBF of the v2 views of a set (tune / val / test), written once to the `fused` cache."""
    path = FUSED / {'tune': 'tune_base', 'val': 'val_ens', 'test': 'test_ens'}[name]
    path = path.with_suffix('.npz')
    if path.exists():
        return
    FUSED.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    meta = experiment_utils.frames_meta(name)
    ids = meta.image_id.astype(str)
    if name == 'tune':
        rows = [None] * len(meta)
        for fold in detector_fusion.FOLDS:
            pos = np.where(meta.fold.values == fold)[0]
            for i, row in zip(
                pos,
                experiment_utils.fuse_sets(view_rows(name, fold, ids.values[pos].tolist()), meta.iloc[pos])):
                rows[i] = row
    else:
        sets = [view_set for fold in detector_fusion.FOLDS for view_set in view_rows(name, fold, ids.tolist())]
        rows = experiment_utils.fuse_sets(sets, meta)
    experiment_utils.save_preds(path, meta.drop(columns='path'), rows)
    print(f'{path.name}: built in {(time.time() - start_time) / 60:.1f} min', flush=True)


def fcos_file(name):
    return detector_fusion.precompute_fcos(name, pipeline_tuning.fcos_job(pipeline_tuning.settings()['views']))


def settings():
    path = TUNING / 'selected.json'
    return json.loads(path.read_text()) if path.exists() else {}


def select(**new_settings):
    TUNING.mkdir(exist_ok=True)
    (TUNING / 'selected.json').write_text(json.dumps({**settings(), **new_settings}, indent=2))


def run(stage, jobs, workers=6):
    """Evaluate the pipelines of `jobs` on the tuning frames, or read the finished stage from its csv."""
    path = TUNING / f'{stage}.csv'
    if path.exists():
        return pd.read_csv(path)
    TUNING.mkdir(exist_ok=True)
    start_time = time.time()
    table = detector_fusion.table(detector_fusion.evaluate_jobs('tune', jobs, None, (), V2, workers))
    table.insert(1, 'job', [json.dumps(jobs[key]) for key in table.key])
    table.to_csv(path, index=False)
    print(f'{stage}: {len(jobs)} jobs in {(time.time() - start_time) / 60:.1f} min', flush=True)
    return table


def best_of(table, prefixes):
    """Job of the best-mAP row among the keys that start with `prefixes`."""
    sub = table[table.key.str.startswith(prefixes)]
    return json.loads(sub.loc[sub.mAP.idxmax(), 'job'])


def tune():
    """Select the temporal and fusion parameters of the two candidates on the out-of-fold frames."""
    grid = pipeline_tuning.temporal_grid()
    jobs = {
        'v2 TTA, no temporal': dict(frcnn=True),
        **{name: dict(frcnn=True, temporal=params) for name, params in grid.items()}}
    table = run('v2_temporal', jobs)
    select(v2_temporal=best_of(table, ('Faster R-CNN parameters', 's_high'))['temporal'])

    fcos_cache = dict(file=fcos_file('tune'))
    jobs = {'v2 only': dict(frcnn=True), 'FCOS only': dict(fcos=fcos_cache)}
    for weight_fcos in (0.25, 0.5, 0.75, 1.0, 1.5):
        for iou in (0.55, 0.65):
            jobs[f'w_fcos {weight_fcos} iou {iou}'] = dict(
                fcos=fcos_cache,
                frcnn=True,
                fusion=dict(weight_fcos=weight_fcos, iou=iou))
    for score_power in (0.75, 1.5):
        jobs[f'w_fcos 0.75 iou 0.55 fcos score^{score_power}'] = dict(
            fcos={**fcos_cache, 'power': score_power},
            frcnn=True,
            fusion=dict(weight_fcos=0.75, iou=0.55))
    table = run('v2_fusion', jobs)
    best_job = best_of(table, ('w_fcos',))
    select(fusion=best_job['fusion'], fusion_power=best_job['fcos'].get('power', 1.0))

    selected = settings()
    base = dict(fcos={**fcos_cache, 'power': selected['fusion_power']}, frcnn=True, fusion=selected['fusion'])
    jobs = {'fusion, no temporal': base, **{name: {**base, 'temporal': params} for name, params in grid.items()}}
    table = run('v2_fusion_temporal', jobs)
    select(fusion_temporal=best_of(table, ('Faster R-CNN parameters', 's_high'))['temporal'])
    print(json.dumps(settings(), indent=2))


def candidates(name):
    """The two candidate pipelines (A, B) with the tuned parameters, for the set `name`."""
    selected = settings()
    fused = dict(fcos=dict(file=fcos_file(name), power=selected['fusion_power']), frcnn=True, fusion=selected['fusion'])
    return {'A: v2 + temporal (tuned)': dict(frcnn=True, temporal=selected['v2_temporal']),
            'B: v2 + FCOS fusion + temporal (tuned)': {**fused, 'temporal': selected['fusion_temporal']}}


def official(workers=4):
    """Evaluate the pipelines once on the official validation set (cached) next to the v1 results."""
    if not RAW.exists():
        selected = settings()
        jobs = {
            'v2 5 checkpoints TTA': dict(frcnn=True),
            'v2 + temporal (Faster R-CNN parameters)': dict(frcnn=True, temporal=detector_fusion.FASTER_RCNN_TEMPORAL),
            'v2 + FCOS fusion': dict(
                fcos=dict(file=fcos_file('val'), power=selected['fusion_power']),
                frcnn=True,
                fusion=selected['fusion']),
            **candidates('val')}
        start_time = time.time()
        results = detector_fusion.evaluate_jobs('val', jobs, None, (), V2, workers)
        RAW.write_text(json.dumps(dict(
            jobs=jobs, selected=selected, minutes=(time.time() - start_time) / 60,
            results={key: dict(metrics=metrics, seconds_per_frame=seconds) for key, (metrics, seconds) in results.items()}),
            indent=2))
    raw = json.loads(RAW.read_text())
    v1_results = json.loads(V1_RAW.read_text())['results']
    keep = [
        'Faster R-CNN 5 checkpoints TTA (0.2587 pipeline)', 'Faster R-CNN + temporal (0.2810 pipeline)',
        'Faster R-CNN + FCOS fusion', REF_KEY]
    results = {f'v1: {key}': (v1_results[key]['metrics'], v1_results[key]['seconds_per_frame']) for key in keep}
    results.update({key: (result['metrics'], result['seconds_per_frame']) for key, result in raw['results'].items()})
    table = detector_fusion.table(results)
    table.to_csv(HERE / 'official_validation_results.csv', index=False)
    return table


def decision(results_table):
    """Apply the decision rule from the module docstring to each candidate."""
    table_by_key = results_table.set_index('key')
    reference = table_by_key.loc[f'v1: {REF_KEY}']
    videos = [column for column in results_table.columns if column.startswith('mAP_video_')]
    out = {}
    for candidate in CANDIDATES:
        row = table_by_key.loc[candidate]
        wins = int(sum(row[video] > reference[video] for video in videos))
        out[candidate] = dict(
            mAP=float(row.mAP), gain=float(row.mAP - reference.mAP), videos_improved=wins, videos=len(videos),
            adopt=bool(row.mAP - reference.mAP >= MARGIN and wins >= MIN_VIDEOS))
    return out


def submit():
    """Write a Kaggle CSV for every candidate that passes the decision rule (nothing is uploaded)."""
    verdicts = decision(official())
    names = {
        'A: v2 + temporal (tuned)': 'faster_rcnn_updated_submission.csv',
        'B: v2 + FCOS fusion + temporal (tuned)': 'faster_rcnn_updated_fcos_submission.csv'}
    images = pd.read_csv(experiment_utils.DATA / 'test_images.csv')
    jobs = candidates('test')
    reports = {}
    for key, verdict in verdicts.items():
        if not verdict['adopt']:
            print('no CSV (rule not met):', key, verdict)
            continue
        csv_path = ROOT / 'kaggle_submissions' / names[key]
        start_time = time.time()
        meta, rows = detector_fusion.pipeline_rows('test', jobs[key], None, [], V2)
        assert meta.image_id.astype(str).tolist() == images.image_id.astype(str).tolist()
        strings = [
            tta_wbf.prediction_string(boxes, scores, labels, h, w)
            for (boxes, scores, labels), h, w in zip(rows, meta.height, meta.width)]
        tmp_path = csv_path.with_suffix('.tmp.csv')
        pd.DataFrame({'image_id': images.image_id, 'PredictionString': strings}).to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
        checks, stats = tta_wbf.validate_submission(csv_path, images)
        reports[key] = dict(
            submission=str(csv_path), pipeline=jobs[key], official_validation=verdict, checks=checks, stats=stats,
            minutes=(time.time() - start_time) / 60)
        print(key, '->', csv_path.name, '| checks:', 'ALL PASS' if all(checks.values()) else checks, '|', stats)
    report_names = {'A: v2 + temporal (tuned)': 'faster_rcnn_updated_submission_report.json',
                    'B: v2 + FCOS fusion + temporal (tuned)': 'faster_rcnn_updated_fcos_submission_report.json'}
    for key, report in reports.items():
        (ROOT / 'kaggle_submissions' / report_names[key]).write_text(json.dumps(
            dict(decision=verdicts, submissions={key: report},
                 note='sample_submission.csv is not available locally; validated against test_images.csv and the Data-page format'),
            indent=2, default=str))

if __name__ == '__main__':
    for step in sys.argv[1:]:
        if step == 'fuse':
            for name in ('tune', 'val', 'test'):
                if all((VIEWS / name / f'fold{fold}').exists() for fold in detector_fusion.FOLDS):
                    try:
                        build_fused(name)
                    except (FileNotFoundError, AssertionError) as error:
                        print(f'{name}: views not complete yet ({type(error).__name__})')
        elif step == 'tune':
            tune()
        elif step == 'official':
            table = official()
            cols = ['key', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR'] + [
                column
                for column in table.columns
                if column.startswith('mAP_video_')]
            print(table[cols].round(4).to_string(index=False))
            print(json.dumps(decision(table), indent=2))
        elif step == 'submit':
            submit()
