"""训练/验证阶段使用的海胆 precision 优先指标。

该类注册到 MMDetection 的 METRICS 注册表，由配置文件按名称构建。它把
每个 batch 的预测与真值暂存为普通列表，epoch 结束后统一寻找满足最低
precision（默认 95%）且 recall 最高的置信度阈值。
"""

from __future__ import annotations

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS

from tools.target_metrics import evaluate_target


@METRICS.register_module()
class TargetPrecisionMetric(BaseMetric):
    """报告指定目标类别在 precision 约束下的最佳 recall。

    ``target_label`` 使用模型内部从 0 开始的连续标签；本数据中海胆为 0。
    ``prefix`` 默认使用 echinus，最终 MMEngine 指标名形如
    ``echinus/recall_at_precision_95``。
    """

    default_prefix = 'echinus'

    def __init__(self, target_label=0, target_name='echinus', iou_threshold=0.5,
                 min_precision=0.95, reference_threshold=0.5,
                 collect_device='cpu', prefix=None):
        # BaseMetric 负责多卡结果收集；collect_device='cpu' 可降低显存占用。
        super().__init__(collect_device=collect_device, prefix=prefix or target_name)
        self.target_label = target_label
        self.iou_threshold = iou_threshold
        self.min_precision = min_precision
        self.reference_threshold = reference_threshold

    def process(self, data_batch, data_samples):
        """接收一个验证 batch，并保存计算指标所需的最小字段。"""
        for sample in data_samples:
            pred = sample.pred_instances.cpu()
            gt = sample.gt_instances.cpu()
            # MMDetection 输出的预测框已恢复到原图坐标，但经过 Resize 的 GT
            # 仍是缩放后坐标。必须用 scale_factor 反缩放，否则二者 IoU 错位，
            # 得到的 precision 与推荐阈值将完全不可信。
            scale_factor = gt.bboxes.new_tensor(sample.scale_factor).flatten()
            if scale_factor.numel() == 2:
                scale_factor = scale_factor.repeat(2)
            if scale_factor.numel() != 4:
                raise ValueError(f'unexpected scale_factor: {sample.scale_factor}')
            gt_boxes = gt.bboxes / scale_factor
            self.results.append({
                'pred_boxes': pred.bboxes.numpy().tolist(),
                'pred_scores': pred.scores.numpy().tolist(),
                'pred_labels': pred.labels.numpy().tolist(),
                'gt_boxes': gt_boxes.numpy().tolist(),
                'gt_labels': gt.labels.numpy().tolist(),
            })

    def compute_metrics(self, results):
        """在整个数据集结果收集完毕后计算并返回标量指标字典。"""
        report = evaluate_target(
            results,
            target_label=self.target_label,
            iou_threshold=self.iou_threshold,
            min_precision=self.min_precision,
            reference_threshold=self.reference_threshold,
        )
        reference = report['reference']
        selected = report['recommended']
        # 把 0.95 转成指标名后缀 95，使 CheckpointHook 能引用稳定的 key。
        suffix = int(round(self.min_precision * 100))
        return {
            'precision_at_score_0_50': reference['precision'],
            'recall_at_score_0_50': reference['recall'],
            'f1_at_score_0_50': reference['f1'],
            f'recall_at_precision_{suffix}': selected['recall'],
            f'precision_at_selected_{suffix}': selected['precision'],
            f'threshold_at_precision_{suffix}': selected['threshold'],
        }
