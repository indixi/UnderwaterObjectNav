"""不依赖 MMDetection 的目标类别 precision/recall 计算函数。

本模块只处理普通 Python 列表，因此既能被训练期自定义 Metric 调用，也能
被训练后的阈值校准脚本复用。当前目标类别是海胆，判定规则为预测框与同图
未匹配真值框的 IoU 达到阈值（默认 0.5）。
"""

from __future__ import annotations


def iou_xyxy(left, right):
    """计算两个 ``[x1, y1, x2, y2]`` 框的交并比。

    对退化框使用 0 面积处理；当并集为 0 时返回 0，避免除零异常。
    """
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _prediction_outcomes(records, target_label, iou_threshold):
    """按置信度从高到低，把目标类预测依次标记为 TP 或 FP。

    每个真值框最多匹配一次。对于同一预测，选择 IoU 最大且尚未被占用的
    真值框；达到阈值记为 TP，否则记为 FP。全局按分数降序处理，使之后
    截取任意置信度阈值时都能直接得到对应的累计 PR 工作点。
    """
    ground_truth = {}
    predictions = []
    total_ground_truth = 0

    # image_index 只在本次评估中充当图像键，不要求等于 COCO image_id。
    for image_index, record in enumerate(records):
        boxes = [
            box for box, label in zip(record['gt_boxes'], record['gt_labels'])
            if int(label) == target_label
        ]
        ground_truth[image_index] = boxes
        total_ground_truth += len(boxes)
        for box, score, label in zip(
                record['pred_boxes'], record['pred_scores'], record['pred_labels']):
            if int(label) == target_label:
                predictions.append((float(score), image_index, box))

    # 必须先按 score 降序再做一对一匹配，这与检测 PR 曲线的生成顺序一致。
    predictions.sort(key=lambda item: item[0], reverse=True)
    matched = {image_index: set() for image_index in ground_truth}
    outcomes = []
    for score, image_index, predicted_box in predictions:
        best_iou = 0.0
        best_index = None
        for gt_index, gt_box in enumerate(ground_truth[image_index]):
            if gt_index in matched[image_index]:
                continue
            overlap = iou_xyxy(predicted_box, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        is_true_positive = best_index is not None and best_iou >= iou_threshold
        if is_true_positive:
            matched[image_index].add(best_index)
        outcomes.append((score, int(is_true_positive)))
    return outcomes, total_ground_truth


def _point(outcomes, total_ground_truth, threshold):
    """计算一个固定置信度阈值下的 TP/FP/FN、precision、recall 和 F1。"""
    selected = [is_tp for score, is_tp in outcomes if score >= threshold]
    true_positive = sum(selected)
    false_positive = len(selected) - true_positive
    false_negative = total_ground_truth - true_positive
    # 没有任何预测时 precision 在数学上无分母。这里约定为 1.0，但同时
    # recall=0；最终报告还会明确标记是否真的检出了至少一个海胆。
    precision = true_positive / (true_positive + false_positive) if selected else 1.0
    recall = true_positive / total_ground_truth if total_ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'threshold': float(threshold),
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_target(records, target_label=0, iou_threshold=0.5,
                    min_precision=0.95, reference_threshold=0.5):
    """在满足最低 precision 的候选阈值中选择 recall 最高的工作点。

    参数 ``records`` 中每个元素代表一张图，包含预测/真值框及对应标签。
    除推荐工作点外，还报告固定 0.5 分数阈值的参考结果，便于观察阈值校准
    带来的 precision/recall 变化。
    """
    outcomes, total_ground_truth = _prediction_outcomes(
        records, target_label, iou_threshold)
    reference = _point(outcomes, total_ground_truth, reference_threshold)

    # 阈值只需考虑实际预测分数的不同取值。用累计计数实现 O(N)，避免每个
    # 阈值都重新扫描所有预测造成 O(N²) 的验证开销。
    candidates = [_point(outcomes, total_ground_truth, 1.0)]
    true_positive = 0
    for index, (score, is_true_positive) in enumerate(outcomes):
        true_positive += is_true_positive
        next_score = outcomes[index + 1][0] if index + 1 < len(outcomes) else None
        if next_score == score:
            continue
        selected_count = index + 1
        precision = true_positive / selected_count
        recall = true_positive / total_ground_truth if total_ground_truth else 0.0
        candidates.append({
            'threshold': score,
            'true_positive': true_positive,
            'false_positive': selected_count - true_positive,
            'false_negative': total_ground_truth - true_positive,
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
        })
    # 空预测点始终存在，因此 feasible 不会为空；若模型完全无法在约束下
    # 正确检出海胆，constraint_satisfied_with_detection 会明确返回 False。
    feasible = [point for point in candidates if point['precision'] >= min_precision]
    selected = max(
        feasible,
        # 首先最大化 recall，其次选择 precision 更高、阈值更保守的点。
        key=lambda point: (point['recall'], point['precision'], point['threshold']),
    )
    return {
        'target_label': target_label,
        'iou_threshold': iou_threshold,
        'minimum_precision': min_precision,
        'constraint_satisfied_with_detection': selected['true_positive'] > 0,
        'ground_truth_instances': total_ground_truth,
        'predictions_considered': len(outcomes),
        'reference': reference,
        'recommended': selected,
    }
