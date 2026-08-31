"""Adapter between the existing DUO GFL detector and semantic mapping.

The mapper only depends on ``detect(image)``; importing MMDetection is lazy so
dataset preparation and policy unit tests work without a GPU environment.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Detection:
    """一个检测结果：像素坐标 bbox、置信度和语义类别。"""
    bbox: tuple[float, float, float, float]     #目标检测框，(x1, y1, x2, y2)
    score: float                                #置信度，表示检测框的可信度
    class_name: str                             ##语义类别，表示检测框对应的物体类别


class MMDetSemanticDetector:
    """将 duo_gfl_project 的 MMDetection 推理器封装为统一 detect 接口。"""
    #config是配置文件，也就是介绍网络结构，checkpoint是权重文件，score_threshold是置信度阈值，target_classes是目标类别
    def __init__(self, config: str, checkpoint: str, score_threshold: float = 0.30,
                 target_classes: tuple[str, ...] = ("echinus", "holothurian", "scallop", "starfish")):
        # 延迟导入：没有安装 MMDetection 时，仍可使用无检测器的基础地图流程。
        from mmdet.apis import DetInferencer
        self.inferencer = DetInferencer(model=config, weights=checkpoint)   #创建MMDetection推理器对象，使用指定的配置文件和权重文件
        self.score_threshold = score_threshold
        self.target_classes = target_classes

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:    #输入可以是图像路径或者图像数组，返回一个检测结果列表
        """对一张 RGB 图像推理，并转换为地图模块所需的 Detection 列表。"""
        result = self.inferencer(image, pred_score_thr=self.score_threshold,
                                 no_save_pred=True, return_datasamples=True)
        pred = result["predictions"][0].pred_instances.cpu()    #[0]表示第一张图预测结果，这里只有一张图，获取预测结果的实例预测，并将其从GPU tensor移动到 CPU tensor上
        detections = []
        for box, score, label in zip(pred.bboxes.numpy(), pred.scores.numpy(), pred.labels.numpy()):    #zip把三个列表相同位置的东西捆到一起
            name = self.target_classes[int(label)]  #根据标签索引获取对应的类别名称
            detections.append(Detection(tuple(map(float, box)), float(score), name))
        return detections


class JsonDetectionDetector:
    """读取 duo_gfl_project/tools/infer.py 生成的缓存 JSON，避免重复推理。也就是一个图片已经推理过了，就不需要再推理了，直接读取缓存的结果。"""
    def __init__(self, records: dict, score_threshold: float = 0.30):   #records是一个字典，存储了图像路径和对应的检测结果，score_threshold是置信度阈值
        self.records, self.score_threshold = records, score_threshold

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        """按图像路径查找缓存记录；找不到时返回空检测列表。"""
        key = str(Path(image).resolve()) if not isinstance(image, np.ndarray) else ""   #如果image不是numpy数组，则将其转换为绝对路径字符串作为key，否则key为空字符串
        record = self.records.get(key, self.records.get(str(image), {}))
        return [Detection(tuple(box), float(score), name)
                for box, score, name in zip(record.get("boxes_xyxy", []), record.get("scores", []),
                                            record.get("class_names", [])) if score >= self.score_threshold]
