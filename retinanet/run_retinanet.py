"""
RetinaNet experiment plan.
1. Baseline: Train the baseline model on all 5 folds.
2. Configuration screening on Fold 0: Test three parameter groups:
   - Anchors: Scales and aspect ratios based on data analysis.
   - Focal loss: alpha = 0.5, gamma = 1.5.
   - IoU: Thresholds of 0.4/0.3 instead of 0.5/0.4.
A new configuration is accepted only if it improves mAP by at least 0.005.
3. Final configuration: Train the selected configuration on all 5 folds.
4. Official validation: Evaluate each checkpoint once.

Anchor selection: At least 99% coverage of medium and large objects at IoU ≥ 0.5, prioritizing the highest overall coverage.
"""
import json, sys, time
from pathlib import Path
import pandas as pd
import retina

MARGIN = 0.005
ANCHOR_SCAN_CSV = retina.ROOT / 'retinanet/anchor_analysis/anchor_scan.csv'
RATIO_SETS = {
    '0.5,1,2': (0.5, 1.0, 2.0),
    '0.5,1,2,3': (0.5, 1.0, 2.0, 3.0),
    '0.33,0.5,1,2,3': (1 / 3, 0.5, 1.0, 2.0, 3.0),
}


def anchor_sizes(mult):
    """torchvision anchor sizes (x, x*2^(1/3), x*2^(2/3)) per FPN level, with x = mult * default base size."""
    return tuple((int(base_size * mult), int(base_size * mult * 2 ** (1 / 3)), int(base_size * mult * 2 ** (2 / 3)))
                 for base_size in (32, 64, 128, 256, 512))


def anchor_choice():
    """Apply the anchor rule to anchor_scan.csv: the chosen scale (ratios 0.5,1,2) and, if it covers more objects, the same
    scale with more aspect ratios; also returns the scan as a wide table."""
    scan = pd.read_csv(ANCHOR_SCAN_CSV)
    scan_table = scan.pivot_table(index=['mult', 'ratios'], columns='objects',
                                  values=['covered_iou50', 'mean_best_iou']).reset_index()
    scan_table.columns = ['_'.join(column_parts).strip('_') for column_parts in scan_table.columns]
    eligible = scan_table[(scan_table.covered_iou50_medium >= 0.99) & (scan_table.covered_iou50_large >= 0.99)]
    default_ratios = eligible[eligible.ratios == '0.5,1,2']
    best_coverage = default_ratios.covered_iou50_all.max()
    scale_choice = (default_ratios[default_ratios.covered_iou50_all >= best_coverage - 0.005]
                    .sort_values('mean_best_iou_all', ascending=False)
                    .iloc[0])
    same_scale = (eligible[(eligible.mult == scale_choice.mult) & (eligible.ratios != '0.5,1,2')]
                  .sort_values('covered_iou50_all', ascending=False))
    ratio_choice = (same_scale.iloc[0]
                    if len(same_scale) and same_scale.iloc[0].covered_iou50_all > scale_choice.covered_iou50_all
                    else None)
    return scale_choice, ratio_choice, scan_table


def stage_candidates(best, scale_choice, ratio_choice):
    """The anchor-stage candidates: the chosen scale, plus the same scale with extended ratios if the rule selected one."""
    yield 'anchors', ([dict(best, name=f'anchors_x{scale_choice.mult:g}', anchor_sizes=anchor_sizes(scale_choice.mult))] +
                      ([dict(best, name=f'anchors_x{ratio_choice.mult:g}_ratios{len(RATIO_SETS[ratio_choice.ratios])}',
                             anchor_sizes=anchor_sizes(ratio_choice.mult),
                             ratios=RATIO_SETS[ratio_choice.ratios])]
                       if ratio_choice is not None else []))


def fold0_mAP(overrides):
    """Held-out fold-0 mAP of a configuration (trained once, then read from its result.json)."""
    return retina.run_experiment(overrides, 0)['metrics']['mAP']


def log(*parts):
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


if __name__ == '__main__':
    decisions = []
    best = dict(name='baseline')
    best_map = fold0_mAP(best)
    log('baseline fold 0 mAP', round(best_map, 4))
    while not ANCHOR_SCAN_CSV.exists():
        time.sleep(30)
    scale_choice, ratio_choice, scan_table = anchor_choice()
    log('anchor rule ->', f'anchors_x{scale_choice.mult:g} (coverage {scale_choice.covered_iou50_all:.3f})',
        '| with ratios',
        None if ratio_choice is None
        else f'x{ratio_choice.mult:g} {ratio_choice.ratios} (coverage {ratio_choice.covered_iou50_all:.3f})')

    def screen(stage, candidates):
        """Train the candidates on fold 0 and adopt the best one only if it beats the current best by MARGIN."""
        global best, best_map
        scores = {candidate['name']: fold0_mAP(candidate) for candidate in candidates}
        top_name = max(scores, key=scores.get)
        adopt = scores[top_name] >= best_map + MARGIN
        decisions.append(dict(stage=stage, base=best['name'], base_fold0=best_map, candidates=scores,
                              adopted=top_name if adopt else None))
        log(stage, scores, '-> adopt' if adopt else '-> keep', top_name if adopt else best['name'])
        if adopt:
            best, best_map = next(candidate for candidate in candidates if candidate['name'] == top_name), scores[top_name]

    screen('anchors', next(stage_candidates(best, scale_choice, ratio_choice))[1])
    run_name = lambda tag: tag if best['name'] == 'baseline' else f"{best['name']}_{tag}"
    screen('focal', [dict(best, name=run_name('focal_a0.5_g1.5'), focal_alpha=0.5, focal_gamma=1.5)])
    screen('iou', [dict(best, name=run_name('iou_0.4_0.3'), fg_iou=0.4, bg_iou=0.3)])
    final = best
    (retina.HERE / 'screening.json').write_text(json.dumps(
        dict(margin=MARGIN, decisions=decisions, final=final,
             anchor_rule=dict(anchors_scale=scale_choice.to_dict(),
                              anchors_scale_and_ratios=None if ratio_choice is None else ratio_choice.to_dict())),
        indent=2, default=str))
    log('final configuration:', final['name'])

    rows = []
    for config in ([dict(name='baseline')] + ([final] if final['name'] != 'baseline' else [])):
        for fold in range(5):
            result = retina.run_experiment(config, fold)
            official = retina.official_validation(config['name'], fold)['metrics']
            rows.append(dict(config=config['name'], fold=fold, best_epoch=result['best_epoch'],
                             **{metric: result['metrics'][metric] for metric in ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']},
                             ms_per_img=result['ms_per_img'],
                             official_mAP=official['mAP'], official_AP50=official['AP50'],
                             official_AP75=official['AP75'], official_APs=official['APs']))
            log(config['name'], 'fold', fold, 'CV mAP', round(result['metrics']['mAP'], 4), '| official', round(official['mAP'], 4))
    table = pd.DataFrame(rows)
    table.to_csv(retina.HERE / 'five_fold_results.csv', index=False)
    print(table.groupby('config')[['mAP', 'AP50', 'AP75', 'APs', 'official_mAP', 'ms_per_img']].mean().round(4).to_string())
