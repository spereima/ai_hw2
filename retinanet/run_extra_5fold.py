"""
Additional RetinaNet experiment conducted after the initial configuration screening.
The goal is to compare the selected configuration with the baseline on the same 5 held-out folds.
"""
import json, sys
import pandas as pd
import retina

if __name__ == '__main__':
    name = sys.argv[1]
    config = json.loads((retina.OUT / name / 'fold0' / 'config.json').read_text())
    overrides = {key: value for key, value in config.items()
                 if (key not in ('fold', 'folds') and retina.BASE.get(key) != value) or key == 'name'}
    print('5-fold run of', name, '| differs from the baseline in',
          {key: value for key, value in overrides.items() if key != 'name'}, flush=True)
    rows = []
    for fold in range(5):
        result = retina.run_experiment(overrides, fold)
        official = retina.official_validation(name, fold)['metrics']
        rows.append(dict(config=name, fold=fold, best_epoch=result['best_epoch'],
                         **{metric: result['metrics'][metric] for metric in ['mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR']},
                         ms_per_img=result['ms_per_img'],
                         official_mAP=official['mAP'], official_AP50=official['AP50'],
                         official_AP75=official['AP75'], official_APs=official['APs']))
        print(name, 'fold', fold, 'CV mAP', round(result['metrics']['mAP'], 4), '| official', round(official['mAP'], 4), flush=True)
    pd.DataFrame(rows).to_csv(retina.HERE / f'five_fold_results_{name}.csv', index=False)
