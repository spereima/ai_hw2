"""
Second stage of Faster R-CNN v2 training, run only if the pilot experiment meets the required mAP threshold.
The best-performing pilot configuration is trained on folds 1–4, while the results from Fold 0 are reused.
"""

import json
import time
import pandas as pd
import faster_rcnn_v2
import train_pilot


def chosen():
    """The pilot experiment with the higher fold-0 mAP (asserts the decision rule) and the fold-0 mAP of every pilot run."""
    reference_map = json.loads(train_pilot.REFERENCE.read_text())['metrics']['mAP']
    scores = {experiment['name']:
              json.loads((faster_rcnn_v2.OUT / experiment['name'] / 'fold0/result.json').read_text())['metrics']['mAP']
              for experiment in faster_rcnn_v2.EXPERIMENTS}
    name = max(scores, key=scores.get)
    assert scores[name] >= reference_map + train_pilot.MARGIN, 'pilot decision rule not met: do not train the other folds'
    return next(experiment for experiment in faster_rcnn_v2.EXPERIMENTS if experiment['name'] == name), scores


def official(name, fold):
    """Official-validation metrics of one checkpoint of run `name`; computed once, then read from disk."""
    path = faster_rcnn_v2.OUT / name / f'fold{fold}' / 'official_val.json'
    if path.exists():
        return json.loads(path.read_text())
    import torch
    import final_evaluation
    model, _ = faster_rcnn_v2.load_model(name, fold)
    image_ids = [int(image['id']) for image in faster_rcnn_v2.coco_val['images']]
    predictions = final_evaluation.predict(model, 'val', image_ids, 4)
    rows = [(image_id, {key: torch.from_numpy(array) for key, array in predictions[image_id].items()}) for image_id in image_ids]
    result = dict(metrics=faster_rcnn_v2.coco_metrics(rows, faster_rcnn_v2.make_gt(faster_rcnn_v2.coco_val, image_ids)))
    faster_rcnn_v2.save_atomic(result, path, keep_backup=False)
    del model
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    experiment, scores = chosen()
    print('pilot fold-0 mAP:', scores, '-> training', experiment['name'], flush=True)
    rows = []
    for fold in range(5):
        start = time.time()
        result = faster_rcnn_v2.run_experiment(experiment, fold)
        official_metrics = official(experiment['name'], fold)['metrics']
        v1_map = json.loads((faster_rcnn_v2.ROOT / f'faster_rcnn/lr_scheduler/fold{fold}/result.json')
                            .read_text())['metrics']['mAP']
        rows.append(dict(fold=fold, cv_mAP=result['metrics']['mAP'], v1_cv_mAP=v1_map, official_mAP=official_metrics['mAP'],
                         best_epoch=result['best_epoch']))
        print(f"fold {fold}: CV mAP {result['metrics']['mAP']:.4f} (v1 {v1_map:.4f}) | "
              f"official val {official_metrics['mAP']:.4f} | {(time.time() - start) / 60:.1f} min", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(faster_rcnn_v2.HERE / f"{experiment['name']}_5fold.csv", index=False)
    print(table.round(4).to_string(index=False))
    print('mean CV mAP', round(table.cv_mAP.mean(), 4), 'vs v1', round(table.v1_cv_mAP.mean(), 4),
          '| official', round(table.official_mAP.mean(), 4))
