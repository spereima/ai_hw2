"""Temporal post-processing of per-frame detections for video object detection.
"""
import io
from contextlib import redirect_stdout
import numpy as np
import cv2

IDENTITY = np.eye(3)
DEFAULTS = dict(s_high=0.5, s_low=0.05, iou_link=0.3, max_gap=2, min_len=3, alpha=0.5, beta=1.0, gamma=0.0,
                motion=True, max_dets=500)


def load_gray(path, width=960):
    """Grayscale image downscaled to at most `width` pixels wide, and the scale factor that was applied."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = min(1.0, width / w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def estimate_motion(prev, cur, scale):
    """3x3 similarity transform from pixel coordinates of the previous frame to the current frame (original resolution).
    Moving objects are outliers of the camera motion and are rejected by RANSAC. Returns (matrix, inlier count)."""
    points = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    if points is None or len(points) < 12:
        return IDENTITY.copy(), 0
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, points, None, winSize=(21, 21), maxLevel=3)
    tracked = status.ravel() == 1
    if tracked.sum() < 12:
        return IDENTITY.copy(), 0
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[tracked], next_points[tracked], method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=2000,
        confidence=0.995)
    if matrix is None or inliers is None or inliers.sum() < 12:
        return IDENTITY.copy(), 0
    out = np.eye(3)
    out[:2] = matrix
    out[:2, 2] /= scale
    return out, int(inliers.sum())


def video_motion(paths, width=960):
    """motion[t] maps frame t-1 to frame t (motion[0] = identity) for the frames of one video in temporal order."""
    motions, inliers = [IDENTITY.copy()], [0]
    prev, scale = load_gray(paths[0], width)
    for path in paths[1:]:
        cur, _ = load_gray(path, width)
        if cur.shape != prev.shape:
            cur = cv2.resize(cur, prev.shape[::-1], interpolation=cv2.INTER_AREA)
        matrix, inlier_count = estimate_motion(prev, cur, scale)
        motions.append(matrix)
        inliers.append(inlier_count)
        prev = cur
    return np.stack(motions), np.array(inliers)


def cumulative(motions):
    """C[t] maps frame 0 to frame t; frame a -> frame b is C[b] @ inv(C[a])."""
    cumulative_motion = [IDENTITY.copy()]
    for matrix in motions[1:]:
        cumulative_motion.append(matrix @ cumulative_motion[-1])
    return np.stack(cumulative_motion)


def warp_boxes(boxes, matrix):
    """Move xyxy boxes with a 3x3 similarity transform (axis-aligned box around the warped corners)."""
    if len(boxes) == 0:
        return boxes.reshape(-1, 4)
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x1, y2], 1), np.stack([x2, y2], 1)], 1)
    warped = corners @ matrix[:2, :2].T + matrix[:2, 2]
    return np.concatenate([warped.min(1), warped.max(1)], 1)


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU of two xyxy box arrays, shape (len(boxes_a), len(boxes_b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(2)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clip(min=0).prod(1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clip(min=0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def center(boxes):
    """Centre points of xyxy boxes."""
    return (boxes[..., :2] + boxes[..., 2:]) / 2


def track_video(frames, cumulative_motion, params):
    """frames: list of (boxes, scores, labels) of consecutive frames. Returns a track id per detection (-1 = untracked)."""
    track_ids = [np.full(len(frame[1]), -1, dtype=np.int64) for frame in frames]
    tracks = []
    for frame_index, (boxes, scores, labels) in enumerate(frames):
        active = [
            track_index
            for track_index, track in enumerate(tracks)
            if frame_index - track['last_t'] <= params['max_gap'] + 1]
        if active:
            predicted_boxes = np.stack(
                [predict(tracks[track_index], frame_index, cumulative_motion) for track_index in active])
            predicted_labels = np.array([tracks[track_index]['label'] for track_index in active])
        free = np.ones(len(active), dtype=bool)
        new_tracks = []
        for score_low, score_high in [(params['s_high'], np.inf), (params['s_low'], params['s_high'])]:
            candidates = np.where((scores >= score_low) & (scores < score_high))[0]
            candidates = candidates[np.argsort(-scores[candidates], kind='stable')]
            if len(candidates) and len(active):
                iou = iou_matrix(boxes[candidates], predicted_boxes)
                iou[labels[candidates][:, None] != predicted_labels[None, :]] = 0
                for i, detection in enumerate(candidates):
                    row = np.where(free, iou[i], 0)
                    j = int(row.argmax())
                    if row[j] >= params['iou_link']:
                        free[j] = False
                        track_index = active[j]
                        update(tracks[track_index], frame_index, boxes[detection], cumulative_motion)
                        track_ids[frame_index][detection] = track_index
            if score_low == params['s_high']:
                new_tracks = [detection for detection in candidates if track_ids[frame_index][detection] < 0]
        for detection in new_tracks:
            tracks.append(dict(label=int(labels[detection]), last_t=frame_index,
                               box=boxes[detection].astype(np.float64), vel=np.zeros(2), n=1))
            track_ids[frame_index][detection] = len(tracks) - 1
    return track_ids


def predict(track, frame_index, cumulative_motion):
    """Where the track's box should be in frame `frame_index` (camera motion + its own velocity)."""
    matrix = cumulative_motion[frame_index] @ np.linalg.inv(cumulative_motion[track['last_t']])
    box = warp_boxes(track['box'][None], matrix)[0]
    return box + np.r_[track['vel'], track['vel']] * (frame_index - track['last_t'])


def update(track, frame_index, box, cumulative_motion):
    """Move the track to the matched detection `box` and update its velocity estimate."""
    seen = warp_boxes(
        track['box'][None],
        cumulative_motion[frame_index] @ np.linalg.inv(cumulative_motion[track['last_t']]))[0]
    velocity = (center(box) - center(seen)) / (frame_index - track['last_t'])
    track['vel'] = velocity if track['n'] == 1 else 0.5 * track['vel'] + 0.5 * velocity
    track['box'], track['last_t'], track['n'] = box.astype(np.float64), frame_index, track['n'] + 1


def temporal_postprocess(frames, motions, params):
    """frames: list of (boxes, scores, labels) for the consecutive frames of one video, motions: video_motion()[0].
    Returns new (boxes, scores, labels) per frame, sorted by score and capped at p['max_dets']."""
    params = {**DEFAULTS, **params}
    cumulative_motion = cumulative(motions if params['motion'] else np.stack([IDENTITY] * len(frames)))
    track_ids = track_video(frames, cumulative_motion, params)
    members = {}
    for frame_index, frame_track_ids in enumerate(track_ids):
        for i, track_id in enumerate(frame_track_ids):
            if track_id >= 0:
                members.setdefault(int(track_id), []).append((frame_index, i))
    track_mean_score = {
        track_id: float(np.mean([frames[frame_index][1][i] for frame_index, i in track_members]))
        for track_id, track_members in members.items()}
    persistent = {track_id for track_id, track_members in members.items() if len(track_members) >= params['min_len']}
    out = []
    for frame_index, (boxes, scores, labels) in enumerate(frames):
        new_scores = scores.astype(np.float64).copy()
        in_track = np.array([track_id in persistent for track_id in track_ids[frame_index]], dtype=bool)
        for i in np.where(in_track)[0]:
            new_scores[i] = ((1 - params['alpha']) * new_scores[i]
                             + params['alpha'] * track_mean_score[int(track_ids[frame_index][i])])
        new_scores[~in_track] *= params['beta']
        out.append([boxes, new_scores, labels])
    if params['gamma'] > 0:
        for track_id in persistent:
            track_members = members[track_id]
            for (start_frame, start_index), (end_frame, end_index) in zip(track_members[:-1], track_members[1:]):
                if 1 < end_frame - start_frame <= params['max_gap'] + 1:
                    start_box, end_box = (frames[start_frame][0][start_index].astype(np.float64),
                                          frames[end_frame][0][end_index].astype(np.float64))
                    score = params['gamma'] * min(out[start_frame][1][start_index], out[end_frame][1][end_index])
                    label = frames[start_frame][2][start_index]
                    for frame_index in range(start_frame + 1, end_frame):
                        weight = (frame_index - start_frame) / (end_frame - start_frame)
                        fwd = warp_boxes(
                            start_box[None],
                            cumulative_motion[frame_index] @ np.linalg.inv(cumulative_motion[start_frame]))[0]
                        bwd = warp_boxes(
                            end_box[None],
                            cumulative_motion[frame_index] @ np.linalg.inv(cumulative_motion[end_frame]))[0]
                        box = (1 - weight) * fwd + weight * bwd
                        same_label = out[frame_index][2] == label
                        if same_label.any() and iou_matrix(
                                box[None], out[frame_index][0][same_label].astype(np.float64)).max() >= 0.5:
                            continue
                        out[frame_index] = [
                            np.vstack([out[frame_index][0], box[None].astype(np.float32)]),
                            np.append(out[frame_index][1], score),
                            np.append(out[frame_index][2], label)]
    final = []
    for boxes, scores, labels in out:
        order = np.argsort(-scores, kind='stable')[:params['max_dets']]
        final.append(
            (boxes[order].astype(np.float32), scores[order].astype(np.float32), labels[order].astype(np.int64)))
    return final


def coco_gt(coco, ids):
    """COCO ground-truth object for the given image ids (boxes with zero width or height are dropped)."""
    from pycocotools.coco import COCO
    ids = set(map(int, ids))
    annotations = []
    for annotation in coco['annotations']:
        if int(annotation['image_id']) in ids and annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0:
            annotations.append(dict(
                annotation,
                area=float(annotation.get('area', annotation['bbox'][2] * annotation['bbox'][3])),
                iscrowd=int(annotation.get('iscrowd', 0)),
                id=len(annotations) + 1))
    gt = COCO()
    gt.dataset = dict(
        info={}, licenses=[], images=[image_info for image_info in coco['images'] if int(image_info['id']) in ids],
        annotations=annotations, categories=coco['categories'])
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def coco_metrics(gt, ids, preds, groups=None, max_dets=100):
    """preds: list of (boxes, scores, labels) aligned with ids; labels 1-8 = COCO category ids. groups: name -> image ids."""
    from pycocotools.cocoeval import COCOeval
    dets = []
    for image_id, (boxes, scores, labels) in zip(ids, preds):
        for (x1, y1, x2, y2), score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            if x2 > x1 and y2 > y1:
                dets.append(dict(image_id=int(image_id), category_id=int(label),
                                 bbox=[x1, y1, x2 - x1, y2 - y1], score=float(score)))
    with redirect_stdout(io.StringIO()):
        results = gt.loadRes(dets)
    out = {}
    for group_name, group_image_ids in (groups or {'all': list(ids)}).items():
        with redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(gt, results, 'bbox')
            coco_eval.params.imgIds = sorted(int(i) for i in group_image_ids)
            coco_eval.params.maxDets = [1, 10, max_dets]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        precision = coco_eval.eval['precision'][:, :, :, 0, -1]
        stats = coco_eval.stats
        out[group_name] = dict(
            mAP=float(precision[precision > -1].mean()), AP50=float(stats[1]), AP75=float(stats[2]),
            APs=float(stats[3]), APm=float(stats[4]), APl=float(stats[5]), AR=float(stats[8]), ARs=float(stats[9]))
    return out


def recovery_potential(frames, gt_frames, tau=0.3, window=2):
    """frames: list of (boxes, scores, labels); gt_frames: list of (boxes, labels, track_ids) of the same consecutive frames.
    For every object that is NOT detected at score >= tau (IoU >= 0.5, same class), checks whether
      - a lower-score box of the same class overlaps it in the same frame (rescoring could recover it),
      - the same ground-truth track is detected at score >= tau within +-window frames (temporal propagation could),
      - there is no box of its class at all (only gap filling could)."""
    found = []
    for (boxes, scores, labels), (gt_boxes, gt_labels, gt_track_ids) in zip(frames, gt_frames):
        iou = iou_matrix(gt_boxes.astype(np.float64), boxes.astype(np.float64))
        same_class = gt_labels[:, None] == labels[None, :]
        best_iou_high_score = (np.where(same_class & (scores[None, :] >= tau), iou, 0).max(1)
                               if len(boxes) else np.zeros(len(gt_boxes)))
        best_iou_any_score = np.where(same_class, iou, 0).max(1) if len(boxes) else np.zeros(len(gt_boxes))
        found.append((best_iou_high_score >= 0.5, best_iou_any_score >= 0.5, gt_track_ids))
    detected_tracks = [set(gt_track_ids[hit_high_score].tolist()) for hit_high_score, _, gt_track_ids in found]
    rows = []
    for frame_index, (hit_high_score, hit_any_score, gt_track_ids) in enumerate(found):
        near = set().union(*[
            detected_tracks[other_frame]
            for other_frame in range(max(0, frame_index - window), min(len(found), frame_index + window + 1))
            if other_frame != frame_index])
        for j in np.where(~hit_high_score)[0]:
            rows.append((bool(hit_any_score[j]), gt_track_ids[j] in near))
    rows = np.array(rows, dtype=bool).reshape(-1, 2)
    num_objects = sum(len(found_frame[0]) for found_frame in found)
    return dict(objects=num_objects, missed=len(rows), missed_share=len(rows) / max(num_objects, 1),
                low_score_box_present=float(rows[:, 0].mean()) if len(rows) else 0.,
                track_detected_nearby=float(rows[:, 1].mean()) if len(rows) else 0.,
                no_box_but_track_nearby=float((~rows[:, 0] & rows[:, 1]).mean()) if len(rows) else 0.,
                low_score_and_track_nearby=float((rows[:, 0] & rows[:, 1]).mean()) if len(rows) else 0.)
