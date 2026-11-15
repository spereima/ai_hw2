"""
RT-DETR experiment plan
1. Base configuration selection on Fold 0: Compare the original settings with a version using warm-up and EMA.
2. Experiments on all 5 folds:
   - Queries: 200 or 450 instead of 300.
   - Matching loss: Change L1/GIoU weights from 5/2 to 7/3.
   - Denoising: Disable denoising queries.
   - Learning rate: Increase the backbone LR from 1e-5 to 2e-5.
   - Resolution: Increase the input size from 640 × 640 to 640 × 1152.
The resolution experiment is based on `no_denoising`, as this configuration performed better across all folds.
3. Evaluation: Train each configuration for 6 epochs.
"""
import json
import sys
import time
import pandas as pd
import rtdetr

MARGIN = 0.005
RECIPES = {'recipe_notebook': dict(), 'recipe_warmup_ema': dict(warmup_iters=500, ema=True)}
EXPERIMENTS = {'queries_200': dict(num_queries=200), 'queries_450': dict(num_queries=450),
               'matching_bbox7_giou3': dict(matcher_bbox_cost=7.0, matcher_giou_cost=3.0, weight_loss_bbox=7.0, weight_loss_giou=3.0),
               'no_denoising': dict(num_denoising=0), 'lr_backbone_2e-5': dict(lr_backbone=2e-5),
               'no_denoising_640x1152': dict(num_denoising=0, height=640, width=1152)}

def log(*messages):
    """Print the message with a time stamp."""
    print(time.strftime('%H:%M:%S'), *messages, flush=True)

def stop():
    """Exit before the next run if a file named STOP exists next to rtdetr.py."""
    if (rtdetr.HERE / 'STOP').exists():
        log('STOP file found: stopping before the next run')
        sys.exit(0)

def run(name, overrides, fold):
    """Train (or read) one run on one fold, evaluate it on the official validation set, return one row of the results table."""
    stop()
    result = rtdetr.run_experiment(dict(name=name, **overrides), fold)
    official_metrics = rtdetr.official_validation(name, fold)['metrics']
    log(name, 'fold', fold, 'CV mAP', round(result['metrics']['mAP'], 4), '| best epoch', result['best_epoch'],
        '| official', round(official_metrics['mAP'], 4))
    return dict(config=name, fold=fold, best_epoch=result['best_epoch'],
                **{metric: result['metrics'][metric] for metric in ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']},
                train_minutes=result['train_minutes'], ms_per_img=result['ms_per_img'], official_mAP=official_metrics['mAP'],
                official_AP50=official_metrics['AP50'], official_AP75=official_metrics['AP75'], official_APs=official_metrics['APs'],
                official_APm=official_metrics['APm'], official_APl=official_metrics['APl'])

if __name__ == '__main__':
    fold0_mAP = {recipe_name: rtdetr.run_experiment(dict(name=recipe_name, **recipe), 0)['metrics']['mAP']
                 for recipe_name, recipe in RECIPES.items()}
    adopt_warmup_ema = fold0_mAP['recipe_warmup_ema'] >= fold0_mAP['recipe_notebook'] + MARGIN
    base_name = 'recipe_warmup_ema' if adopt_warmup_ema else 'recipe_notebook'
    base_recipe = RECIPES[base_name]
    (rtdetr.HERE / 'recipe_decision.json').write_text(json.dumps(dict(fold0_mAP=fold0_mAP, margin=MARGIN, adopted=base_name), indent=2))
    log('recipe decision:', fold0_mAP, '->', base_name)
    rows = []
    rows += [run(base_name, base_recipe, fold) for fold in range(5)]
    other_name = next(recipe_name for recipe_name in RECIPES if recipe_name != base_name)
    rows.append(run(other_name, RECIPES[other_name], 0))
    pd.DataFrame(rows).to_csv(rtdetr.HERE / 'five_fold_results.csv', index=False)
    for name, overrides in EXPERIMENTS.items():
        rows += [run(name, {**base_recipe, **overrides}, fold) for fold in range(5)]
        pd.DataFrame(rows).to_csv(rtdetr.HERE / 'five_fold_results.csv', index=False)
    results_table = pd.DataFrame(rows)
    print(results_table.groupby('config')[['mAP', 'AP50', 'APs', 'official_mAP', 'train_minutes', 'ms_per_img']].mean().round(4).to_string())
    log('PLAN_DONE')
