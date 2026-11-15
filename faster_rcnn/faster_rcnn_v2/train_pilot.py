"""
Pilot experiment of Faster R-CNN v2 on fold 0.

Two configurations from faster_rcnn_v2.py are trained: with and without scale augmentation.
After training, the best checkpoint of each configuration is evaluated once on official validation
using the standard Faster R-CNN protocol (single checkpoint, 720/1280 resize,
no TTA, WBF, or temporal processing).
"""
import json
import time
import torch
import faster_rcnn_v2

REFERENCE = faster_rcnn_v2.ROOT / 'faster_rcnn/lr_scheduler/fold0/result.json'
MARGIN = 0.010


def official_validation(name):
    """Official-validation metrics (and speed) of the fold-0 checkpoint of run `name`; computed once, then read from disk."""
    path = faster_rcnn_v2.OUT / name / 'fold0' / 'official_val.json'
    if path.exists():
        return json.loads(path.read_text())
    model, cfg = faster_rcnn_v2.load_model(name, faster_rcnn_v2.PILOT_FOLD)
    import final_evaluation
    image_ids = [int(image['id']) for image in faster_rcnn_v2.coco_val['images']]
    dataset = faster_rcnn_v2.VisDrone(faster_rcnn_v2.DATA, 'val', faster_rcnn_v2.coco_val, image_ids)
    start = time.time()
    predictions = final_evaluation.predict(model, 'val', image_ids, 4)
    rows = [(image_id, {key: torch.from_numpy(array) for key, array in predictions[image_id].items()})
            for image_id in image_ids]
    metrics = faster_rcnn_v2.coco_metrics(rows, faster_rcnn_v2.make_gt(faster_rcnn_v2.coco_val, image_ids))
    result = dict(metrics=metrics, seconds=time.time() - start, ms_per_img=faster_rcnn_v2.measure_speed(model, dataset))
    faster_rcnn_v2.save_atomic(result, path, keep_backup=False)
    del model
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    reference_map = json.loads(REFERENCE.read_text())['metrics']['mAP']
    for experiment in faster_rcnn_v2.EXPERIMENTS:
        start = time.time()
        result = faster_rcnn_v2.run_experiment(experiment, faster_rcnn_v2.PILOT_FOLD)
        official = official_validation(experiment['name'])
        print(f"{experiment['name']}: fold-0 mAP {result['metrics']['mAP']:.4f} "
              f"(reference {reference_map:.4f}, {result['metrics']['mAP'] - reference_map:+.4f}) | "
              f"official val {official['metrics']['mAP']:.4f} | {(time.time() - start) / 60:.1f} min", flush=True)
    best_map = max(json.loads((faster_rcnn_v2.OUT / experiment['name'] / 'fold0/result.json').read_text())['metrics']['mAP']
                   for experiment in faster_rcnn_v2.EXPERIMENTS)
    print('decision:',
          'continue with the other folds' if best_map >= reference_map + MARGIN else 'stop (gain below the fixed margin)')
