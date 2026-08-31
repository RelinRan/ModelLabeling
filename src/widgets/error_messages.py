from __future__ import annotations

"""Chinese translations for service-layer error messages.

Services raise English errors (developer-facing); this module converts the
known patterns to Chinese right before they are shown in a prompt, so a
Chinese UI never surfaces English text. Unknown messages pass through
unchanged.
"""

_PATTERNS: list[tuple[str, str]] = [
    # dataset_detector
    ("dataset directory does not exist: ", "数据集目录不存在: "),
    ("multiple datasets found under: ", "该目录下存在多个数据集: "),
    ("unsupported dataset format: ", "不支持的数据集格式: "),
    ("YOLO metadata must be a mapping: ", "YOLO data.yaml 格式错误: "),
    # annotation_service
    ("label is not in presets: ", "标签不在类别列表中: "),
    ("label is not in COCO categories: ", "标签不在 COCO 类别中: "),
    ("missing object name in ", "VOC 缺少 object/name: "),
    ("missing bndbox for ", "VOC 缺少 bndbox: "),
    ("YOLO Pose requires keypoint annotations", "YOLO 关键点格式需要关键点标注"),
    ("YOLO Pose requires an outer bounding box", "YOLO 关键点格式需要外接框"),
    ("YOLO Pose kpt_shape must use [count, 3]", "kpt_shape 必须为 [数量, 3] 格式"),
    ("YOLO Segmentation requires at least three polygon points", "分割多边形至少需要三个顶点"),
    ("YOLO Segmentation does not support keypoints", "分割格式不支持关键点"),
    ("YOLO OBB requires four-corner rotated box annotations", "旋转框格式需要四个角点"),
    ("expected 5 values", "应为 5 个数值（检测格式）"),
    ("expected 9 values for a YOLO OBB row", "旋转框每行应为 9 个数值"),
    ("invalid YOLO Segmentation row", "分割行格式无效"),
    ("invalid YOLO Pose keypoint row", "关键点行格式无效"),
    ("unknown class id ", "未知类别 ID "),
    ("normalized value out of range", "归一化数值超出范围"),
    ("normalized bbox out of range", "归一化框坐标超出范围"),
    ("normalized keypoint out of range", "归一化关键点坐标超出范围"),
    ("keypoint visibility must be 0, 1, or 2", "关键点可见性必须为 0、1 或 2"),
    ("bbox dimensions must be positive", "框的宽高必须为正数"),
    ("expected ", "应为 "),
    (" keypoints, got ", " 个关键点，实际 "),
    ("kpt_names has ", "kpt_names 有 "),
    (" names, but row has ", " 个名称，但该行有 "),
    # format_capabilities
    ("does not support: ", "不支持以下形状: "),
    ("YOLO Segmentation does not support multipart polygon instances", "分割格式不支持多部分多边形"),
    ("YOLO Pose requires at least one keypoint per annotation", "每个关键点标注至少需要一个点位"),
    ("YOLO Pose requires one consistent keypoint count across the dataset", "整个数据集的关键点数量必须一致"),
    ("YOLO Pose requires one consistent keypoint schema across the dataset", "整个数据集的关键点命名必须一致"),
    ("YOLO OBB requires exactly four corner points: ", "旋转框需要恰好四个角点: "),
    ("COCO requires one keypoint schema per category: ", "COCO 每个类别只能有一套关键点命名: "),
    # conversion / coco store
    ("inconsistent keypoint schema for YOLO Pose output", "YOLO 关键点输出格式不一致"),
    ("conversion save failed", "转换保存失败"),
    ("COCO keypoint schema conflicts with category: ", "COCO 关键点命名与类别冲突: "),
    ("Existing keypoints: ", "已有关键点: "),
    ("New keypoints: ", "当前关键点: "),
    ("Use the same keypoint names/count for this category, or a different category label.", "请为该类别使用相同的关键点名称/数量，或改用其他类别名称。"),
]


def translate_error(message: str) -> str:
    """Return the Chinese form of a known service error; pass through unknown."""
    text = str(message)
    for english, chinese in _PATTERNS:
        if english in text:
            text = text.replace(english, chinese)
    return text
