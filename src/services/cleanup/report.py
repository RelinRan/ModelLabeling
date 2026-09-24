"""Issue model shared by every format validator.

A rule never deletes or rewrites anything: it only describes what it found.
The severity tells the caller what may be done about it, which is the whole
point of separating the two -- "the XML declares a size that disagrees with
the JPEG" is a broken invariant, while "this box is 12px wide" is a policy
decision that belongs to the annotator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class Severity(str, Enum):
    """What the caller is allowed to do with a finding.

    STRUCTURAL
        An invariant between image, annotation and index is broken, so the
        finding has exactly one defensible repair. Safe to fix automatically.
    REVIEW
        A data-quality judgement that depends on the annotation policy (how
        small is too small, whether a box-less image is a negative sample).
        Never applied without the user asking for it.
    INFO
        Report only. Nothing to repair, but the annotator should know.
    """

    STRUCTURAL = "structural"
    REVIEW = "review"
    INFO = "info"


@dataclass(frozen=True)
class RuleSpec:
    severity: Severity
    tag: tuple[str, str]          # (zh, en)
    template: tuple[str, str]     # (zh, en) -- str.format(**data)


#: Every rule id the validators may emit. Keeping the severity and the wording
#: in one table means a new rule cannot be added with a missing translation, and
#: the dialog never needs to know how a finding is phrased.
RULES: dict[str, RuleSpec] = {
    # ---- layout, format independent -------------------------------------
    "image.unreadable": RuleSpec(
        Severity.STRUCTURAL, ("图片损坏", "Corrupt image"),
        ("无法解码：{error}", "cannot decode: {error}")),
    "image.stem_conflict": RuleSpec(
        Severity.STRUCTURAL, ("同名冲突", "Stem conflict"),
        ("与 {others} 共用同一个标注文件", "shares one annotation file with {others}")),
    # ---- Pascal VOC ------------------------------------------------------
    "voc.annotation.missing": RuleSpec(
        Severity.REVIEW, ("缺标注", "No annotation"),
        ("没有对应的 .xml", "no matching .xml")),
    "voc.annotation.unreadable": RuleSpec(
        Severity.STRUCTURAL, ("标注损坏", "Unreadable annotation"),
        ("XML 无法解析：{error}", "XML does not parse: {error}")),
    "voc.annotation.empty": RuleSpec(
        Severity.REVIEW, ("无目标", "No objects"),
        ("XML 存在但没有 <object>", "XML exists but has no <object>")),
    "voc.object.blank_name": RuleSpec(
        Severity.STRUCTURAL, ("标签为空", "Blank label"),
        ("第 {index} 个 <object> 的 <name> 为空", "<object> #{index} has an empty <name>")),
    "voc.object.unknown_label": RuleSpec(
        Severity.REVIEW, ("未知标签", "Unknown label"),
        ("{label} 不在项目标签中", "{label} is not one of the project labels")),
    "voc.box.missing": RuleSpec(
        Severity.STRUCTURAL, ("缺框", "Missing bndbox"),
        ("<object> #{index} 缺少 <bndbox>", "<object> #{index} has no <bndbox>")),
    "voc.box.bad_number": RuleSpec(
        Severity.STRUCTURAL, ("坐标非法", "Bad coordinate"),
        ("<object> #{index} 的 {field}={value} 不是数字",
         "<object> #{index} has {field}={value}, which is not a number")),
    "voc.box.not_integral": RuleSpec(
        Severity.REVIEW, ("坐标非整", "Non-integer coordinate"),
        ("<object> #{index} 的坐标含小数，VOC 规范要求整数",
         "<object> #{index} has fractional coordinates; VOC requires integers")),
    "voc.box.degenerate": RuleSpec(
        Severity.STRUCTURAL, ("退化框", "Degenerate box"),
        ("<object> #{index} 的宽高为 {width}x{height}",
         "<object> #{index} has size {width}x{height}")),
    "voc.box.out_of_bounds": RuleSpec(
        Severity.STRUCTURAL, ("越界框", "Out-of-bounds box"),
        ("<object> #{index} 的 [{xmin},{ymin},{xmax},{ymax}] 超出 {image_width}x{image_height}",
         "<object> #{index} [{xmin},{ymin},{xmax},{ymax}] is outside {image_width}x{image_height}")),
    "voc.box.small": RuleSpec(
        Severity.REVIEW, ("过小框", "Tiny box"),
        ("<object> #{index} 的 {width}x{height} 小于阈值 {threshold}",
         "<object> #{index} is {width}x{height}, below the {threshold} threshold")),
    "voc.box.duplicate": RuleSpec(
        Severity.REVIEW, ("重复框", "Duplicate box"),
        ("{count} 个完全相同的 {label} 框", "{count} identical {label} boxes")),
    "voc.box.geometry_conflict": RuleSpec(
        Severity.REVIEW, ("几何撞车", "Geometry conflict"),
        ("{labels} 共用同一位置", "{labels} share the same geometry")),
    "voc.box.near_duplicate": RuleSpec(
        Severity.REVIEW, ("近似重复框", "Near-duplicate box"),
        ("{label} 的 {first} 与 {second} 重叠 {iou}", "{first} and {second} overlap by {iou}")),
    "voc.size.missing": RuleSpec(
        Severity.STRUCTURAL, ("缺尺寸", "Missing size"),
        ("XML 没有 <size> 节点", "XML has no <size> element")),
    "voc.size.mismatch": RuleSpec(
        Severity.STRUCTURAL, ("尺寸不符", "Size mismatch"),
        ("声明 {declared_width}x{declared_height}，实际 {image_width}x{image_height}",
         "declares {declared_width}x{declared_height}, image is {image_width}x{image_height}")),
    "voc.depth.mismatch": RuleSpec(
        # One defensible repair -- write the real channel count -- so this is
        # a broken invariant, not a judgement call.
        Severity.STRUCTURAL, ("通道数不符", "Depth mismatch"),
        ("声明 depth={declared_depth}，实际 {actual_depth}",
         "declares depth={declared_depth}, image has {actual_depth}")),
    "voc.filename.mismatch": RuleSpec(
        Severity.STRUCTURAL, ("文件名不符", "Filename mismatch"),
        ("<filename>={declared}，实际 {actual}",
         "<filename>={declared}, file is {actual}")),
    "voc.orphan": RuleSpec(
        Severity.STRUCTURAL, ("孤立标注", "Orphan annotation"),
        ("没有对应的图片", "no matching image")),

    # ---- YOLO ------------------------------------------------------------
    "yolo.annotation.missing": RuleSpec(
        Severity.REVIEW, ("缺标注", "No annotation"),
        ("没有对应的 .txt", "no matching .txt")),
    "yolo.annotation.unreadable": RuleSpec(
        Severity.STRUCTURAL, ("标注损坏", "Unreadable annotation"),
        ("无法读取：{error}", "cannot read: {error}")),
    "yolo.annotation.empty": RuleSpec(
        Severity.REVIEW, ("无目标", "No objects"),
        ("文件存在但没有标注行", "file exists but has no annotation row")),
    "yolo.row.bad_arity": RuleSpec(
        Severity.STRUCTURAL, ("列数错误", "Bad row arity"),
        ("第 {line} 行有 {count} 列，应为 {expected}",
         "line {line} has {count} columns, expected {expected}")),
    "yolo.row.bad_number": RuleSpec(
        Severity.STRUCTURAL, ("数值非法", "Bad value"),
        ("第 {line} 行的 {value} 不是数字", "line {line} has {value}, which is not a number")),
    "yolo.row.class_unknown": RuleSpec(
        Severity.STRUCTURAL, ("类别越界", "Unknown class"),
        ("第 {line} 行的 class_id={class_id} 超出 {total} 类的类别表",
         "line {line} uses class_id={class_id}, outside the {total}-class table")),
    "yolo.row.out_of_range": RuleSpec(
        Severity.STRUCTURAL, ("归一化越界", "Out-of-range value"),
        ("第 {line} 行的坐标超出 [0,1]", "line {line} has a coordinate outside [0,1]")),
    "yolo.row.degenerate": RuleSpec(
        Severity.STRUCTURAL, ("退化框", "Degenerate box"),
        ("第 {line} 行的 w={width} h={height}", "line {line} has w={width} h={height}")),
    "yolo.row.small": RuleSpec(
        Severity.REVIEW, ("过小框", "Tiny box"),
        ("第 {line} 行的 {width}x{height} 小于阈值 {threshold}",
         "line {line} is {width}x{height}, below the {threshold} threshold")),
    "yolo.row.duplicate": RuleSpec(
        Severity.REVIEW, ("重复框", "Duplicate box"),
        ("{count} 个完全相同的 class {class_id} 标注行",
         "{count} identical class {class_id} rows")),
    "yolo.row.near_duplicate": RuleSpec(
        Severity.REVIEW, ("近似重复框", "Near-duplicate box"),
        ("第 {first} 行与第 {second} 行重叠 {iou}", "line {first} and line {second} overlap by {iou}")),
    "yolo.classes.missing": RuleSpec(
        Severity.STRUCTURAL, ("缺类别表", "No class table"),
        ("找不到 classes.txt，data.yaml 也没有 names", "no classes.txt and no data.yaml names")),
    "yolo.classes.preset_mismatch": RuleSpec(
        Severity.REVIEW, ("类别表不符", "Class table mismatch"),
        ("数据集={disk}，项目预设={project}", "dataset={disk}, project={project}")),
    "yolo.classes.unused": RuleSpec(
        Severity.INFO, ("未用类别", "Unused class"),
        ("{names} 没有被任何标注引用", "{names} is referenced by no annotation")),
    "yolo.cache.present": RuleSpec(
        Severity.REVIEW, ("陈旧缓存", "Stale cache"),
        ("{name} 不会被清理同步，训练前应删除", "{name} will not be updated; delete it before training")),
    "yolo.data_yaml.path_missing": RuleSpec(
        Severity.STRUCTURAL, ("路径失效", "Missing path"),
        ("{key}={value} 不存在", "{key}={value} does not exist")),
    "yolo.orphan": RuleSpec(
        Severity.STRUCTURAL, ("孤立标注", "Orphan annotation"),
        ("没有对应的图片", "no matching image")),
    "yolo.unrecognized": RuleSpec(
        Severity.REVIEW, ("非标注文件", "Unrecognized file"),
        ("位于标注目录但不是标注格式，清理不会删除", "in the label directory but not an annotation; cleanup leaves it")),

    # ---- COCO ------------------------------------------------------------
    "coco.json.missing": RuleSpec(
        Severity.STRUCTURAL, ("缺标注文件", "No annotation file"),
        ("目录里没有 COCO JSON", "no COCO JSON in the directory")),
    "coco.json.unreadable": RuleSpec(
        Severity.STRUCTURAL, ("标注损坏", "Unreadable annotation"),
        ("JSON 无法解析：{error}", "JSON does not parse: {error}")),
    "coco.image_record.missing_file": RuleSpec(
        Severity.STRUCTURAL, ("记录无图", "Record without image"),
        ("image id {id} 的 {file_name} 不在磁盘上",
         "image id {id} points at {file_name}, which is not on disk")),
    "coco.image_record.no_annotation": RuleSpec(
        Severity.REVIEW, ("无目标", "No objects"),
        ("image id {id} 没有标注", "image id {id} has no annotation")),
    "coco.image.on_disk_unregistered": RuleSpec(
        Severity.REVIEW, ("图片未登记", "Unregistered image"),
        ("磁盘上有图片但 JSON 没有记录", "on disk but absent from the JSON")),
    "coco.image.duplicate_id": RuleSpec(
        Severity.STRUCTURAL, ("ID 重复", "Duplicate id"),
        ("image id {id} 出现 {count} 次", "image id {id} appears {count} times")),
    "coco.annotation.duplicate_id": RuleSpec(
        Severity.STRUCTURAL, ("ID 重复", "Duplicate id"),
        ("annotation id {id} 出现 {count} 次", "annotation id {id} appears {count} times")),
    "coco.annotation.dangling_image": RuleSpec(
        Severity.STRUCTURAL, ("悬空引用", "Dangling reference"),
        ("annotation id {id} 指向不存在的 image_id {image_id}",
         "annotation id {id} points at missing image_id {image_id}")),
    "coco.annotation.unknown_category": RuleSpec(
        Severity.STRUCTURAL, ("未知类别", "Unknown category"),
        ("annotation id {id} 的 category_id={category_id} 未定义",
         "annotation id {id} uses undefined category_id={category_id}")),
    "coco.annotation.bad_bbox": RuleSpec(
        Severity.STRUCTURAL, ("框非法", "Invalid bbox"),
        ("annotation id {id} 的 bbox={bbox}", "annotation id {id} has bbox={bbox}")),
    "coco.annotation.area_mismatch": RuleSpec(
        Severity.STRUCTURAL, ("面积不符", "Area mismatch"),
        ("annotation id {id} 的 area={area}，bbox 面积为 {expected}",
         "annotation id {id} has area={area}, bbox area is {expected}")),
    "coco.annotation.out_of_bounds": RuleSpec(
        Severity.STRUCTURAL, ("越界框", "Out-of-bounds box"),
        ("annotation id {id} 超出 {image_width}x{image_height}",
         "annotation id {id} is outside {image_width}x{image_height}")),
    "coco.annotation.small": RuleSpec(
        Severity.REVIEW, ("过小框", "Tiny box"),
        ("annotation id {id} 的 {width}x{height} 小于阈值 {threshold}",
         "annotation id {id} is {width}x{height}, below the {threshold} threshold")),
    "coco.annotation.duplicate": RuleSpec(
        Severity.REVIEW, ("重复框", "Duplicate box"),
        ("annotation id {id} 与 id {other_id} 完全相同",
         "annotation id {id} is identical to id {other_id}")),
    "coco.annotation.near_duplicate": RuleSpec(
        Severity.REVIEW, ("近似重复框", "Near-duplicate box"),
        ("annotation id {id} 与 id {other_id} 重叠 {iou}",
         "annotation id {id} and id {other_id} overlap by {iou}")),
    "coco.annotation.geometry_conflict": RuleSpec(
        Severity.REVIEW, ("几何撞车", "Geometry conflict"),
        ("annotation id {id} 与 id {other_id} 位置相同但类别不同",
         "annotation id {id} and id {other_id} share one box under different categories")),
    "coco.category.unused": RuleSpec(
        Severity.INFO, ("未用类别", "Unused category"),
        ("category id {id} ({name}) 没有被引用", "category id {id} ({name}) is referenced by nothing")),
    "coco.db.json_divergence": RuleSpec(
        Severity.STRUCTURAL, ("DB 与 JSON 不一致", "DB/JSON divergence"),
        ("{detail}", "{detail}")),
}


@dataclass(frozen=True)
class Issue:
    """One finding. ``data`` keeps the facts so it can be rendered in any
    language, and so a future fixer can act without re-parsing the file."""

    rule: str
    target: Path
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> Severity:
        return RULES[self.rule].severity


def issue(rule: str, target: Path, **data: Any) -> Issue:
    """Build an issue, rejecting an unknown rule id at construction time."""
    if rule not in RULES:
        raise KeyError(f"unknown cleanup rule: {rule}")
    return Issue(rule, Path(target), data)


def format_issue(value: Issue, english: bool = False) -> str:
    """Render one issue as a tagged, single-line entry."""
    spec = RULES[value.rule]
    index = 1 if english else 0
    return f"[{spec.tag[index]}] {value.target.name}：{spec.template[index].format(**value.data)}"


@dataclass
class Plan:
    """The complete read-only verdict for one dataset."""

    root: Path
    format_name: str
    total_images: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, *values: Issue) -> None:
        self.issues.extend(values)

    def extend(self, values: Iterable[Issue]) -> None:
        self.issues.extend(values)

    def of_severity(self, severity: Severity) -> list[Issue]:
        return [value for value in self.issues if value.severity is severity]

    def by_rule(self) -> dict[str, list[Issue]]:
        grouped: dict[str, list[Issue]] = {}
        for value in self.issues:
            grouped.setdefault(value.rule, []).append(value)
        return grouped

    def counts(self) -> dict[str, int]:
        return {rule: len(values) for rule, values in sorted(self.by_rule().items())}

    def is_clean(self) -> bool:
        return not self.issues
