"""训练/验证阶段使用的海胆 precision 优先指标。

该类注册到 MMDetection 的 METRICS 注册表，由配置文件按名称构建。它把
每个 batch 的预测与真值暂存为普通列表，epoch 结束后统一寻找满足最低
precision（默认 95%）且 recall 最高的置信度阈值。
"""

from __future__ import annotations

from collections.abc import Mapping

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS

from tools.target_metrics import evaluate_target


def _field(container, name):
    """同时读取 MMEngine 字典和 ``BaseDataElement`` 对象中的字段。

    MMEngine 的 ``Evaluator.process`` 会在调用各个 metric 前，把模型返回的
    ``DetDataSample`` 递归转换成普通字典；但直接单独调用本 metric 时也可能
    收到尚未转换的对象。统一通过这个小函数读取，可以兼容这两种调用路径。
    """
    if isinstance(container, Mapping):
        return container[name]
    return getattr(container, name)


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
            # Evaluator 正常运行时 sample、pred 和 gt 都是普通字典；兼容读取
            # 函数也允许在调试时直接传入 DetDataSample/InstanceData 对象。
            pred = _field(sample, 'pred_instances')
            gt = _field(sample, 'gt_instances')
            pred_boxes = _field(pred, 'bboxes').detach().cpu()
            pred_scores = _field(pred, 'scores').detach().cpu()
            pred_labels = _field(pred, 'labels').detach().cpu()
            gt_boxes = _field(gt, 'bboxes').detach().cpu()
            gt_labels = _field(gt, 'labels').detach().cpu()
            # MMDetection 输出的预测框已恢复到原图坐标，但经过 Resize 的 GT
            # 仍是缩放后坐标。必须用 scale_factor 反缩放，否则二者 IoU 错位，
            # 得到的 precision 与推荐阈值将完全不可信。
            sample_scale_factor = _field(sample, 'scale_factor')
            scale_factor = gt_boxes.new_tensor(sample_scale_factor).flatten()
            if scale_factor.numel() == 2:
                scale_factor = scale_factor.repeat(2)
            if scale_factor.numel() != 4:
                raise ValueError(
                    f'unexpected scale_factor: {sample_scale_factor}')
            gt_boxes = gt_boxes / scale_factor
            self.results.append({
                'pred_boxes': pred_boxes.numpy().tolist(),
                'pred_scores': pred_scores.numpy().tolist(),
                'pred_labels': pred_labels.numpy().tolist(),
                'gt_boxes': gt_boxes.numpy().tolist(),
                'gt_labels': gt_labels.numpy().tolist(),
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
