"""
Tuning the FCOS inference pipeline and its fusion with Faster R-CNN using held-out training frames, without using the official validation set.
"""
import json
import sys
import time
from pathlib import Path
import pandas as pd
import detector_fusion

OUT = detector_fusion.HERE / 'tuning'
CONFIGS = ['F0_baseline', 'F1_ciou', 'F2_radius_1_0']
ALL_VIEWS = [f'{flip}@{scale}' for scale in ('1.0', '1.25', '1.5', '1.75') for flip in ('none', 'h')]


def best(stage):
    """Row with the highest mAP of the results table of `stage`."""
    table = pd.read_csv(OUT / f'{stage}.csv')
    return table.loc[table.mAP.idxmax()]


def settings():
    """The settings selected so far by the earlier stages (empty before the first stage)."""
    path = OUT / 'selected.json'
    return json.loads(path.read_text()) if path.exists() else {}


def select(**new_settings):
    """Store new selected settings next to the old ones and return all of them."""
    selected = {**settings(), **new_settings}
    (OUT / 'selected.json').write_text(json.dumps(selected, indent=2))
    return selected


def run(stage, jobs, configs=None, views=(), with_frcnn=False, workers=6):
    """Evaluate the pipelines of `jobs` on the tuning frames, or read the finished stage from its csv."""
    path = OUT / f'{stage}.csv'
    if path.exists():
        return pd.read_csv(path)
    OUT.mkdir(exist_ok=True)
    start_time = time.time()
    results = detector_fusion.evaluate_jobs('tune', jobs, configs, views, with_frcnn, workers)
    table = detector_fusion.table(results)
    table.insert(1, 'job', [json.dumps(jobs[k]) for k in table.key])
    table.to_csv(path, index=False)
    print(f'{stage}: {len(jobs)} pipelines in {(time.time() - start_time) / 60:.1f} min')
    return table


def fcos_job(views, configs=None, **overrides):
    """FCOS part of a pipeline: the selected settings, with `overrides` applied on top."""
    selected = settings()
    fcos_settings = dict(
        configs=configs or [selected['config']], views=list(views), score_min=selected.get('score_min', 0.01),
        wbf_iou=selected.get('wbf_iou', 0.55), conf_type=selected.get('conf_type', 'avg'),
        pre_nms=selected.get('pre_nms'))
    fcos_settings.update(overrides)
    return fcos_settings


def stage_configs():
    jobs = {}
    for config in CONFIGS:
        jobs[f'{config} native (score>=0.02, 300 det)'] = dict(
            fcos=dict(configs=[config], views=['none@1.0'], score_min=0.02),
            top=300)
        jobs[f'{config} score>=0.01, 500 det'] = dict(
            fcos=dict(configs=[config], views=['none@1.0'], score_min=0.01),
            top=500)
    jobs['3 configs WBF (none@1.0)'] = dict(
        fcos=dict(configs=CONFIGS, views=['none@1.0'], score_min=0.01, wbf_iou=0.55))
    jobs['Faster R-CNN TTA+WBF (reference)'] = dict(frcnn=True)
    table = run('configs', jobs, CONFIGS, ['none@1.0'], with_frcnn=True)
    single = table[table.key.str.contains('score>=0.01')]
    choice = single.loc[single.mAP.idxmax(), 'key'].split(' ')[0]
    select(config=choice)
    return table


def stage_wbf_probe():
    """FCOS keeps boxes up to IoU 0.6 (native NMS), so WBF at the Faster R-CNN setting (0.55) also merges neighbouring objects of
    one model. Probe the fusion of two simple view pairs before selecting views."""
    pair, three = ['none@1.0', 'h@1.0'], ['none@1.0', 'none@1.25', 'none@1.5']
    jobs = {'F2 none@1.0 alone': dict(fcos=dict(configs=['F2_radius_1_0'], views=['none@1.0'], score_min=0.01))}
    for name, views in (('flip pair @1.0', pair), ('none @1.0+1.25+1.5', three)):
        for wbf_iou in (0.55, 0.6, 0.65, 0.7, 0.8):
            jobs[f'{name}: WBF {wbf_iou}'] = dict(
                fcos=dict(configs=['F2_radius_1_0'], views=views, score_min=0.01, wbf_iou=wbf_iou))
        for pre_nms in (0.5,):
            for wbf_iou in (0.55, 0.6):
                jobs[f'{name}: NMS {pre_nms} per view + WBF {wbf_iou}'] = dict(
                    fcos=dict(configs=['F2_radius_1_0'], views=views, score_min=0.01, pre_nms=pre_nms, wbf_iou=wbf_iou))
    table = run('wbf_probe', jobs, 'F2_radius_1_0', sorted(set(pair + three)), workers=7)
    best_fcos_job = json.loads(best('wbf_probe').job)['fcos']
    select(config='F2_radius_1_0', wbf_iou=best_fcos_job['wbf_iou'], pre_nms=best_fcos_job.get('pre_nms'))
    return table


def stage_views():
    scale_sets = [
        ['1.0'], ['1.25'], ['1.5'], ['1.75'], ['1.0', '1.25'], ['1.0', '1.5'], ['1.25', '1.5'], ['1.5', '1.75'],
        ['1.0', '1.25', '1.5'], ['1.25', '1.5', '1.75'], ['1.0', '1.5', '1.75'], ['1.0', '1.25', '1.5', '1.75']]
    jobs = {}
    for scales in scale_sets:
        jobs['flip+none @ ' + '+'.join(scales)] = dict(
            fcos=fcos_job([f'{flip}@{scale}' for scale in scales for flip in ('none', 'h')]))
    for scales in (['1.0', '1.25', '1.5'], ['1.25', '1.5', '1.75'], ['1.0', '1.25', '1.5', '1.75']):
        jobs['none @ ' + '+'.join(scales)] = dict(fcos=fcos_job([f'none@{scale}' for scale in scales]))
    table = run('views', jobs, settings()['config'], ALL_VIEWS)
    select(views=json.loads(best('views').job)['fcos']['views'])
    return table


def stage_wbf():
    views = settings()['views']
    base_wbf_iou = settings()['wbf_iou']
    jobs = {
        f'iou {wbf_iou}': dict(fcos=fcos_job(views, wbf_iou=wbf_iou))
        for wbf_iou in sorted({round(base_wbf_iou - 0.05, 2), base_wbf_iou, round(base_wbf_iou + 0.05, 2)})}
    jobs.update({f'score>={score_min}': dict(fcos=fcos_job(views, score_min=score_min)) for score_min in (0.02, 0.05)})
    jobs['conf box_and_model_avg'] = dict(fcos=fcos_job(views, conf_type='box_and_model_avg'))
    jobs['conf max'] = dict(fcos=fcos_job(views, conf_type='max'))
    table = run('wbf', jobs, settings()['config'], views)
    best_fcos_job = json.loads(best('wbf').job)['fcos']
    select(wbf_iou=best_fcos_job['wbf_iou'], score_min=best_fcos_job['score_min'], conf_type=best_fcos_job['conf_type'])
    return table


def temporal_grid():
    """The temporal-parameter grid searched around the frozen Faster R-CNN parameters, keyed by a readable name."""
    base = detector_fusion.FASTER_RCNN_TEMPORAL
    grid = {'Faster R-CNN parameters (frozen)': base}
    for s_high in (0.2, 0.3, 0.4):
        for alpha in (0.5, 0.65, 0.8):
            for beta in (0.6, 1.0):
                if (s_high, alpha, beta) != (base['s_high'], base['alpha'], base['beta']):
                    grid[f's_high {s_high} alpha {alpha} beta {beta}'] = {
                        **base, 's_high': s_high, 'alpha': alpha, 'beta': beta}
    return grid


def stage_temporal():
    selected = settings()
    fcos_cache = dict(file=detector_fusion.precompute_fcos('tune', fcos_job(selected['views'])))
    jobs = {'FCOS TTA, no temporal': dict(fcos=fcos_cache)}
    jobs.update({name: dict(fcos=fcos_cache, temporal=params) for name, params in temporal_grid().items()})
    table = run('temporal', jobs)
    sub = table[table.key != 'FCOS TTA, no temporal']
    select(fcos_temporal=json.loads(sub.loc[sub.mAP.idxmax(), 'job'])['temporal'])
    return table


def stage_fusion():
    selected = settings()
    fcos_settings = fcos_job(selected['views'])
    jobs = {'Faster R-CNN only': dict(frcnn=True), 'FCOS only': dict(fcos=fcos_settings)}
    for weight_fcos in (0.25, 0.5, 0.75, 1.0, 1.5):
        for iou in (0.55, 0.65):
            jobs[f'w_fcos {weight_fcos} iou {iou}'] = dict(
                fcos=fcos_settings,
                frcnn=True,
                fusion=dict(weight_fcos=weight_fcos, iou=iou))
    for score_power in (0.75, 1.5):
        jobs[f'w_fcos 0.75 iou 0.55 fcos score^{score_power}'] = dict(
            fcos={**fcos_settings, 'power': score_power},
            frcnn=True,
            fusion=dict(weight_fcos=0.75, iou=0.55))
    table = run('fusion', jobs, selected['config'], selected['views'], with_frcnn=True)
    sub = table[table.key.str.startswith('w_fcos')]
    best_job = json.loads(sub.loc[sub.mAP.idxmax(), 'job'])
    select(fusion=best_job['fusion'], fusion_power=best_job['fcos'].get('power', 1.0))
    return table


def stage_fusion_temporal():
    selected = settings()
    fcos_settings = {**fcos_job(selected['views']), 'power': selected.get('fusion_power', 1.0)}
    base = dict(fcos=fcos_settings, frcnn=True, fusion=selected['fusion'])
    jobs = {
        'fusion, no temporal': base,
        'Faster R-CNN + Faster R-CNN temporal (the 0.2810 pipeline)': dict(frcnn=True, temporal=detector_fusion.FASTER_RCNN_TEMPORAL)}
    jobs.update({name: {**base, 'temporal': params} for name, params in temporal_grid().items()})
    table = run('fusion_temporal', jobs, selected['config'], selected['views'], with_frcnn=True)
    sub = table[table.key.str.startswith(('Faster R-CNN parameters', 's_high'))]
    select(fusion_temporal=json.loads(sub.loc[sub.mAP.idxmax(), 'job'])['temporal'])
    return table

STAGES = dict(
    configs=stage_configs, wbf_probe=stage_wbf_probe, views=stage_views, wbf=stage_wbf, temporal=stage_temporal,
    fusion=stage_fusion, fusion_temporal=stage_fusion_temporal)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        table = STAGES[name]()
        cols = ['key', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR'] + [
            column
            for column in table.columns
            if column.startswith('mAP_fold')]
        print(table[cols].sort_values('mAP', ascending=False).head(12).round(4).to_string(index=False))
        print('selected:', settings())
