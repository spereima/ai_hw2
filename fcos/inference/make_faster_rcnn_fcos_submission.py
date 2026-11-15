"""
Generating a Kaggle CSV for the final Faster R-CNN + FCOS ensemble.
Predictions from both models across 5 folds, using TTA and WBF, are combined through weighted box fusion, followed by temporal post-processing.
"""
import json
import os
import sys
import time
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'shared'))
import detector_fusion
import pipeline_tuning
import tta_wbf

OUT = detector_fusion.ROOT / 'kaggle_submissions' / 'faster_rcnn_fcos_submission.csv'


def final_job():
    """The frozen pipeline as a job for detector_fusion.pipeline_rows on the test set, and the selected settings."""
    selected = pipeline_tuning.settings()
    fcos_ensemble_file = detector_fusion.precompute_fcos('test', pipeline_tuning.fcos_job(selected['views']))
    return dict(fcos=dict(file=fcos_ensemble_file, power=selected.get('fusion_power', 1.0)), frcnn=True,
                fusion=selected['fusion'], temporal=selected['fusion_temporal']), selected


if __name__ == '__main__':
    start = time.time()
    job, selected = final_job()
    meta, rows = detector_fusion.pipeline_rows('test', job, None, [], True)
    images = pd.read_csv(detector_fusion.eu.DATA / 'test_images.csv')
    assert meta.image_id.astype(str).tolist() == images.image_id.astype(str).tolist()
    prediction_strings = [tta_wbf.prediction_string(boxes, scores, labels, h, w)
                          for (boxes, scores, labels), h, w in zip(rows, meta.height, meta.width)]
    tmp_path = OUT.with_suffix('.tmp.csv')
    pd.DataFrame({'image_id': images.image_id, 'PredictionString': prediction_strings}).to_csv(tmp_path, index=False)
    os.replace(tmp_path, OUT)
    checks, stats = tta_wbf.validate_submission(OUT, images)
    report = dict(submission=str(OUT), pipeline=job, checks=checks, stats=stats, minutes=(time.time() - start) / 60,
                  note='sample_submission.csv is not available locally; validated against test_images.csv and the Data-page format')
    (detector_fusion.ROOT / 'kaggle_submissions' / 'faster_rcnn_fcos_submission_report.json').write_text(json.dumps(report, indent=2, default=str))
    for check, passed in checks.items():
        print('PASS' if passed else 'FAIL', check)
    print(stats)
    sys.exit(0 if all(checks.values()) else 1)
