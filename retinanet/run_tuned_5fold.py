import json
import pandas as pd
import retina

if __name__ == '__main__':
    screening = json.loads((retina.HERE / 'screening.json').read_text())
    anchor_stage = next(decision for decision in screening['decisions'] if decision['stage'] == 'anchors')
    name = max(anchor_stage['candidates'], key=anchor_stage['candidates'].get)
    config = json.loads((retina.OUT / name / 'fold0' / 'config.json').read_text())
    overrides = {key: config[key] for key in ('name', 'anchor_sizes', 'ratios')}
    print('tuned configuration trained on 5 folds:', overrides, flush=True)
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
    pd.DataFrame(rows).to_csv(retina.HERE / 'five_fold_results_tuned.csv', index=False)
