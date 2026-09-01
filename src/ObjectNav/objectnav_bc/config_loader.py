"""统一加载 ObjectNav 的 YAML 配置并解析外部模型路径。

所有相对路径都以 YAML 文件所在目录为基准，而不是以命令启动目录为
基准。这样无论从哪里执行 Python，检测器配置和 checkpoint 都能定位到
同一个文件。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectorConfig:
    """经过路径解析和类型转换的检测器配置。"""

    enabled: bool
    config_path: Path | None
    checkpoint_path: Path | None
    score_threshold: float
    classes: tuple[str, ...]

    def validate(self) -> None:
        """启用检测器时，尽早报告缺失或错误的模型文件。"""
        if not self.enabled:
            return
        if self.config_path is None:
            raise ValueError("perception.detector.config_path 不能为空")
        if self.checkpoint_path is None:
            raise ValueError("perception.detector.checkpoint_path 不能为空")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"找不到检测器 config：{self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到检测器 checkpoint：{self.checkpoint_path}")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("perception.detector.score_threshold 必须在 [0, 1] 内")
        if not self.classes:
            raise ValueError("perception.detector.classes 不能为空")


@dataclass(frozen=True)
class BCConfig:
    """ObjectNav 预处理和训练共用的配置内容。"""

    source_path: Path
    dataset: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    detector: DetectorConfig


def _resolve_path(yaml_dir: Path, value: str | None) -> Path | None:
    """把 YAML 中的绝对/相对路径统一转换为规范的绝对路径。"""
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (yaml_dir / path).resolve()


def load_bc_config(path: str | Path, validate_detector: bool = True) -> BCConfig:
    """读取 bc.yaml；可选择是否立即检查检测器文件存在性。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 bc.yaml 需要 PyYAML，请安装 requirements.txt") from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 BC 配置文件：{source}")
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"BC 配置顶层必须是字典：{source}")

    detector_doc = document.get("perception", {}).get("detector", {})
    classes = tuple(detector_doc.get(
        "classes", ("holothurian", "echinus", "scallop", "starfish")))
    detector = DetectorConfig(
        enabled=bool(detector_doc.get("enabled", False)),
        config_path=_resolve_path(source.parent, detector_doc.get("config_path")),
        checkpoint_path=_resolve_path(source.parent, detector_doc.get("checkpoint_path")),
        score_threshold=float(detector_doc.get("score_threshold", 0.30)),
        classes=classes,
    )
    if validate_detector:
        detector.validate()

    return BCConfig(
        source_path=source,
        dataset=dict(document.get("dataset", {})),
        model=dict(document.get("model", {})),
        training=dict(document.get("training", {})),
        detector=detector,
    )
