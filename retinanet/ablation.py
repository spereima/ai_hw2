"""
Additional RetinaNet ablation experiment conducted after obtaining results from all 5 folds.

The RetinaNet baseline (COCO-pretrained weights + ignore-aware loss) achieved an average mAP
0.008 lower than the original notebook version (ImageNet-pretrained backbone, without ignore-region handling).

To separately evaluate the impact of these two changes, two additional experiments are
conducted on Fold 1, where the performance gap was the largest:

- ablation_coco_no_ignore: COCO initialization without ignore-aware loss.
  Comparison with the baseline shows the effect of ignore-aware loss.
- ablation_imagenet_no_ignore: ImageNet initialization without ignore-aware loss.
  Comparison with ablation_coco_no_ignore shows the effect of COCO initialization
  and checks whether the original training setup can be reproduced.

All other settings, data, training procedure, best epoch selection based on held-out fold mAP,
and evaluation remain unchanged.
"""
import pandas as pd
import retina

FOLD = 1
RUNS = [
    dict(name='ablation_coco_no_ignore', ignore_ioa=0.0),
    dict(name='ablation_imagenet_no_ignore', init='imagenet', ignore_ioa=0.0),
]

if __name__ == '__main__':
    rows = []
    for overrides in [dict(name='baseline')] + RUNS:
        result = retina.run_experiment(overrides, FOLD)
        rows.append(dict(config=overrides['name'], fold=FOLD, best_epoch=result['best_epoch'],
                         last_epoch_mAP=result['history'][-1]['mAP'],
                         **{metric: result['metrics'][metric] for metric in ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']}))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(retina.HERE / 'ablation.csv', index=False)
