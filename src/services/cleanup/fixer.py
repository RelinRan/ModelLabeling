"""Repairs -- the only module in the package that changes a dataset.

Three properties are load-bearing here.

**Nothing is deleted without a copy.** Every removed file is written to a
backup directory *beside* the dataset first, and a delete whose copy failed is
abandoned. The backup never lives under the dataset root: anything left there
is dataset content as far as the next scan is concerned, so a backup inside it
would be read back as images and labels.

**A repair is never invented.** :data:`FIXES` maps a rule to the one action
that is defensible for it, and a rule missing from that table can only be
reported. ``voc.object.unknown_label`` is the clearest case: the box is
unusable *as labelled*, but the fix is "tell me the right label", not "delete
the annotation" -- which is what it would have to do. Guessing there would
destroy real annotation the moment someone scans a dataset whose labels are
not the current project's presets.

**The plan is not trusted.** :func:`fix_dataset` rescans between passes rather
than editing from the stale issues it started with, so a repair that changes
what the next rule sees (a file whose last box was removed is now a box-less
image) is caught. That also makes idempotence observable: the final scan is
the receipt, and a second :func:`fix_dataset` over a fixed dataset reports
nothing to do.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .report import RULES, Plan, Severity
from .runner import scan_dataset

# ---- actions ---------------------------------------------------------------
# One box or row, leaving the annotation file in place.
DROP_OBJECT = "drop_object"
# A box whose coordinates are outside the image: clamp it back in.
CLAMP_OBJECT = "clamp_object"
# A VOC coordinate carrying a fraction, which the format does not allow.
ROUND_OBJECT = "round_object"
# The whole annotation file, because no image needs it.
DROP_ANNOTATION = "drop_annotation"
# The image and its annotation: an image with no usable boxes left.
DROP_IMAGE = "drop_image"
# A COCO record, or one annotation inside the JSON.
DROP_RECORD = "drop_record"
# The declared width/height/depth/filename of a VOC file.
PATCH_HEADER = "patch_header"
# A COCO ``area``, recomputed from the bbox it must agree with.
RECOMPUTE_AREA = "recompute_area"
# An id that appears more than once: keep the first, drop the rest.
DROP_DUPLICATE_IDS = "drop_duplicate_ids"
# ``classes.txt`` built from the project presets.
WRITE_CLASSES = "write_classes"
# An Ultralytics label cache that a rewrite would leave stale.
DROP_CACHE = "drop_cache"

#: What the fixer may do about each rule. A rule absent from this table can
#: only be reported; see the module docstring for why that is not a gap.
FIXES: dict[str, str] = {
    "image.unreadable": DROP_IMAGE,

    "voc.annotation.missing": DROP_IMAGE,
    "voc.annotation.unreadable": DROP_IMAGE,
    "voc.annotation.empty": DROP_IMAGE,
    "voc.object.blank_name": DROP_OBJECT,
    "voc.box.missing": DROP_OBJECT,
    "voc.box.bad_number": DROP_OBJECT,
    "voc.box.not_integral": ROUND_OBJECT,
    "voc.box.degenerate": DROP_OBJECT,
    "voc.box.out_of_bounds": CLAMP_OBJECT,
    "voc.box.small": DROP_OBJECT,
    "voc.box.duplicate": DROP_OBJECT,
    "voc.box.near_duplicate": DROP_OBJECT,
    "voc.size.missing": PATCH_HEADER,
    "voc.size.mismatch": PATCH_HEADER,
    "voc.depth.mismatch": PATCH_HEADER,
    "voc.filename.mismatch": PATCH_HEADER,
    "voc.orphan": DROP_ANNOTATION,

    "yolo.annotation.missing": DROP_IMAGE,
    "yolo.annotation.unreadable": DROP_IMAGE,
    "yolo.annotation.empty": DROP_IMAGE,
    "yolo.row.bad_arity": DROP_OBJECT,
    "yolo.row.bad_number": DROP_OBJECT,
    "yolo.row.class_unknown": DROP_OBJECT,
    "yolo.row.out_of_range": DROP_OBJECT,
    "yolo.row.degenerate": DROP_OBJECT,
    "yolo.row.small": DROP_OBJECT,
    "yolo.row.duplicate": DROP_OBJECT,
    "yolo.row.near_duplicate": DROP_OBJECT,
    "yolo.classes.missing": WRITE_CLASSES,
    "yolo.cache.present": DROP_CACHE,
    "yolo.orphan": DROP_ANNOTATION,

    "coco.image_record.missing_file": DROP_IMAGE,
    "coco.image_record.no_annotation": DROP_IMAGE,
    "coco.image.on_disk_unregistered": DROP_IMAGE,
    "coco.image.duplicate_id": DROP_DUPLICATE_IDS,
    "coco.annotation.duplicate_id": DROP_DUPLICATE_IDS,
    "coco.annotation.dangling_image": DROP_RECORD,
    "coco.annotation.bad_bbox": DROP_RECORD,
    "coco.annotation.small": DROP_RECORD,
    "coco.annotation.duplicate": DROP_RECORD,
    "coco.annotation.near_duplicate": DROP_RECORD,
    "coco.annotation.area_mismatch": RECOMPUTE_AREA,
    "coco.annotation.out_of_bounds": CLAMP_OBJECT,
}


@dataclass(frozen=True)
class FixPolicy:
    """Which findings the fixer is allowed to act on.

    The rules are named, not severity-grouped, because the two sets do not
    line up: "delete every image with no boxes" and "delete every box below
    20px" are both REVIEW findings, and both are things a user may well want
    on -- while "the label is not in the project presets" is likewise REVIEW
    and must never be automatic.
    """

    rules: frozenset[str]

    @classmethod
    def structural(cls) -> "FixPolicy":
        """Repair broken invariants only: nothing that is a judgement call."""
        return cls(frozenset(
            rule for rule, severity in _severities().items()
            if severity is Severity.STRUCTURAL
            and rule in FIXES
            and rule not in REQUIRES_CONSENT
        ))

    @classmethod
    def quality(cls) -> "FixPolicy":
        """The annotation-policy rules: how small is too small, and whether a
        box-less or unreadable image is kept."""
        return cls(FixPolicy.quality_rules())

    @staticmethod
    def quality_rules() -> frozenset[str]:
        return frozenset(
            rule for rule, severity in _severities().items()
            if rule in FIXES
            and (severity is Severity.REVIEW or rule in REQUIRES_CONSENT)
        )

    @classmethod
    def of(cls, rules: Iterable[str]) -> "FixPolicy":
        return cls(frozenset(rules))

    @classmethod
    def everything(cls) -> "FixPolicy":
        return cls(frozenset(FIXES))

    def allows(self, rule: str) -> bool:
        return rule in self.rules and rule in FIXES

    def unfixable(self) -> set[str]:
        """Rules the caller asked for that no repair exists for.

        Surfaced rather than swallowed: a user who ticked a box and saw
        nothing happen deserves to know the tick could not do anything.
        """
        return {rule for rule in self.rules if rule not in FIXES}


def _severities() -> dict[str, Severity]:
    return {rule: spec.severity for rule, spec in RULES.items()}


def available_fixes() -> dict[str, str]:
    """Every rule a policy may name, with the action it maps to."""
    return dict(FIXES)


# ---- describing a repair ---------------------------------------------------


#: Rules that have a repair but must never be applied without the user
#: asking. ``image.unreadable`` is the case that forced this set: the file is
#: genuinely broken, so the finding is STRUCTURAL, but the only repair is to
#: delete the image -- and with it whatever annotation it had. The image may
#: be recoverable, the annotation certainly is not, so the decision belongs
#: to the user rather than to a default.
REQUIRES_CONSENT = frozenset({"image.unreadable"})


#: Rules that name a *group* of identical boxes rather than a single one.
#: Their repair keeps one member and drops the rest, so a blanket "drop
#: everything this finding points at" would delete the group outright --
#: which is how a duplicate-box repair once removed every box in the file.
#: They are handled explicitly, and skipped by the generic pass.
_GROUP_RULES = {
    "voc": {"voc.box.duplicate"},
    "yolo": {"yolo.row.duplicate"},
    "coco": {"coco.annotation.duplicate"},
}


@dataclass(frozen=True)
class Repair:
    """One intended change, before it is made."""

    rule: str
    action: str
    target: Path
    data: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """The action in the report's own words, for the log and manifest."""
        return f"{_ACTION_TEXT.get(self.action, self.action)}：{self.target.name}"


#: Rendered into the cleanup log. The rule tag is added by the caller, which
#: knows the language.
_ACTION_TEXT = {
    DROP_OBJECT: "移除标注框",
    CLAMP_OBJECT: "收拢越界框",
    ROUND_OBJECT: "坐标取整",
    DROP_ANNOTATION: "删除孤立标注",
    DROP_IMAGE: "删除图片及标注",
    DROP_RECORD: "删除标注记录",
    PATCH_HEADER: "修正声明信息",
    RECOMPUTE_AREA: "重算面积",
    DROP_DUPLICATE_IDS: "去重 ID",
    WRITE_CLASSES: "写入类别表",
    DROP_CACHE: "删除陈旧缓存",
}


def plan_repairs(plan: Plan, policy: FixPolicy) -> list[Repair]:
    """Turn a scan into the list of changes the policy allows.

    Sorted by target so a file touched by several rules is edited once.
    """
    repairs: list[Repair] = []
    for value in plan.issues:
        action = FIXES.get(value.rule)
        if action is None or not policy.allows(value.rule):
            continue
        repairs.append(Repair(value.rule, action, value.target, dict(value.data)))
    return repairs


# ---- backup ----------------------------------------------------------------


def _unlink(path: Path) -> None:
    """The one place a file is destroyed.

    A module-level function, looked up by name at call time, so a caller can
    substitute it: the test suite points it at a guard that refuses anything
    outside the test's own scratch directory. Deleting through this seam is
    what keeps the repair pass inside the same confinement the dialog's
    deletion always had.
    """
    path.unlink()


class Backup:
    """Copies every removed file aside before it is deleted.

    ``remove`` refuses to unlink unless the copy landed, so a full disk turns
    into a reported error rather than lost annotation.
    """

    def __init__(self, root: Path, directory: Path | None = None) -> None:
        self.root = Path(root)
        if directory is not None:
            self.directory = Path(directory)
        else:
            # ``mkdtemp`` rather than a timestamp: a name resolved to the
            # second is not unique, and two runs sharing one directory would
            # also share -- and so skip -- each other's copies, leaving a
            # manifest that lists files it never took.
            parent = Path(tempfile.gettempdir()) / "model-labeling-cleanup"
            parent.mkdir(parents=True, exist_ok=True)
            self.directory = Path(
                tempfile.mkdtemp(prefix=f"{self.root.name}-", dir=parent)
            )
        self.entries: list[dict[str, str]] = []
        self.errors: list[str] = []

    def save(self, path: Path, reason: str, kind: str = "removed") -> bool:
        relative = self._relative(path)
        destination = self.directory / relative
        if not destination.exists():
            # Only the first copy is taken. A file touched by two passes --
            # rewritten, then removed -- would otherwise be copied a second
            # time from a version this very run had already changed, and the
            # copy the user restores from would not be the file they had.
            # The event is still recorded below; only the bytes are frozen.
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            except OSError as exc:
                self.errors.append(f"备份失败 {relative}：{exc}")
                return False
        # ``kind`` is explicit rather than inferred from the reason: the log
        # has to tell a deleted file from a rewritten one, and matching on a
        # translated string would break the moment the wording changed.
        self.entries.append({"path": relative.as_posix(), "reason": reason, "kind": kind})
        return True

    def remove(self, path: Path, reason: str) -> bool:
        if not path.exists():
            return False
        if not self.save(path, reason):
            return False
        try:
            _unlink(path)
        except OSError as exc:
            self.errors.append(f"删除失败 {self._relative(path)}：{exc}")
            # The copy stays, but it is not a removal, and saying so is the
            # point: the log and the manifest are where a user checks what a
            # run did, and both would otherwise claim a file that is still
            # there was cleaned.
            self._restate(path, "kept")
            return False
        return True

    def _restate(self, path: Path, kind: str) -> None:
        """Re-label the entry ``save`` just added for this path."""
        relative = self._relative(path).as_posix()
        for entry in reversed(self.entries):
            if entry["path"] == relative and entry["kind"] == "removed":
                entry["kind"] = kind
                return

    def _relative(self, path: Path) -> Path:
        try:
            return path.relative_to(self.root)
        except ValueError:
            return Path(path.name)


# ---- applying --------------------------------------------------------------


@dataclass
class FixReport:
    """What a :func:`fix_dataset` run did, and what it left behind."""

    root: Path
    backup_dir: Path | None = None
    manifest_path: Path | None = None
    repairs: list[Repair] = field(default_factory=list)
    passes: int = 0
    errors: list[str] = field(default_factory=list)
    remaining: Plan | None = None
    #: Backup entries, split so a caller can log a deletion differently from
    #: a rewrite without inspecting the backup directory itself. ``kept`` is
    #: the copy of a delete that did not go through: the file is still there,
    #: and the copy is in the backup directory, so both halves need saying.
    removed: list[dict[str, str]] = field(default_factory=list)
    rewritten: list[dict[str, str]] = field(default_factory=list)
    kept: list[dict[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.repairs)

    @property
    def is_clean(self) -> bool:
        return self.remaining is not None and self.remaining.is_clean()

    def counts(self) -> dict[str, int]:
        counted: dict[str, int] = {}
        for repair in self.repairs:
            counted[repair.rule] = counted.get(repair.rule, 0) + 1
        return dict(sorted(counted.items()))


def fix_dataset(
    root: Path,
    *,
    presets=None,
    policy: FixPolicy | None = None,
    detected=None,
    min_box_size: int | None = None,
    near_duplicate_iou: float | None = None,
    max_passes: int = 3,
    backup_dir: Path | None = None,
    write_manifest: bool = True,
    progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> FixReport:
    """Repair a dataset until the scan comes back clean, or nothing is left.

    Each pass rescans, so a repair is never made from a stale reading: the
    cascade the user's rules imply -- the last box in a file is below the
    threshold, so the box goes, so the image has no boxes, so the image goes --
    plays out over successive passes instead of being guessed in one.

    ``dry_run`` runs every pass and reports the plan without touching a byte.
    """
    from src.services.dataset_detector import DatasetDetector

    root = Path(root)
    keywords: dict[str, Any] = {}
    if min_box_size is not None:
        keywords["min_box_size"] = min_box_size
    if near_duplicate_iou is not None:
        keywords["near_duplicate_iou"] = near_duplicate_iou

    report = FixReport(root=root)
    backup = Backup(root, backup_dir)
    policy = policy or FixPolicy.structural()

    for attempt in range(1, max_passes + 1):
        report.passes = attempt
        detected = detected or DatasetDetector.detect(root)
        plan = scan_dataset(root, presets=presets, detected=detected, **keywords)
        repairs = plan_repairs(plan, policy)

        if not repairs:
            report.remaining = plan
            break

        report.repairs.extend(repairs)
        if dry_run:
            # Report the first pass and stop: a later pass would be reporting
            # the effect of edits that were never made.
            report.remaining = plan
            break

        if progress is not None:
            progress(f"第 {attempt} 轮：{len(repairs)} 处")
        before = _inventory(root)
        report.errors.extend(apply_repairs(detected, repairs, backup, presets))
        if _inventory(root) == before:
            # The pass left the dataset exactly as it found it, so the next
            # scan would find the same repairs and the pass after that the
            # same again. That happens when what remains is something this
            # machine will not allow -- a locked file, a read-only
            # directory -- so stop and report it instead of repeating it.
            report.remaining = plan
            break
    else:
        # Ran out of passes with work still pending: report what is left
        # rather than claiming success.
        report.remaining = scan_dataset(root, presets=presets, detected=detected, **keywords)

    if not dry_run and report.repairs:
        if backup.entries:
            report.backup_dir = backup.directory
        report.removed = [e for e in backup.entries if e.get("kind") == "removed"]
        report.rewritten = [e for e in backup.entries if e.get("kind") == "rewritten"]
        report.kept = [e for e in backup.entries if e.get("kind") == "kept"]
        report.errors.extend(backup.errors)
        if write_manifest:
            report.manifest_path = _write_manifest(report, backup, policy)
    return report


def _inventory(root: Path) -> frozenset[tuple[str, int, int]]:
    """Name, size and mtime of every file under ``root``.

    The pass loop uses this to notice a pass that changed nothing. Size alone
    will not do: correcting a ``<depth>4</depth>`` to ``3`` is a rewrite of
    exactly the same length, so the timestamp has to be part of the answer.
    """
    entries = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return frozenset(entries)


def apply_repairs(detected, repairs: list[Repair], backup: Backup,
                  presets=None) -> list[str]:
    """Make the changes, one target file at a time."""
    grouped: dict[Path, list[Repair]] = {}
    for repair in repairs:
        grouped.setdefault(repair.target, []).append(repair)

    dispatch = {
        "voc": _apply_voc,
        "yolo": _apply_yolo,
        "coco": _apply_coco,
    }.get(detected.format_name)
    if dispatch is None:
        return [f"未知的数据集格式：{detected.format_name}"]
    return dispatch(detected, grouped, backup, presets)


# ---- Pascal VOC ------------------------------------------------------------


def _drop_image_pair(detected, image_path: Path, suffix: str, backup: Backup,
                     errors: list[str]) -> None:
    """Delete an image and the annotation that belongs to it.

    The order is the whole point. An annotation is only an orphan once its
    image is really gone, so an image that could not be deleted keeps its
    label: otherwise a delete that failed on a locked file quietly becomes a
    lost annotation, and the dataset ends up with neither.
    """
    annotation = _annotation_for(detected, image_path, suffix)
    backup.remove(image_path, "无可用标注的图片")
    if image_path.exists():
        errors.append(f"{image_path.name}：图片未能删除，配对标注保留")
        return
    backup.remove(annotation, "无可用标注的图片")


def _apply_voc(detected, grouped, backup: Backup, presets=None) -> list[str]:
    errors: list[str] = []
    for target, repairs in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        actions = {repair.action for repair in repairs}
        try:
            if actions <= {DROP_ANNOTATION}:
                # An .xml no image needs: the repair's target is the file.
                backup.remove(target, "孤立标注")
                continue
            if DROP_IMAGE in actions:
                _drop_image_pair(detected, target, ".xml", backup, errors)
                continue
            errors.extend(_edit_voc(detected, target, repairs, backup))
        except OSError as exc:
            errors.append(f"{target.name}：{exc}")
    return errors


def _edit_voc(detected, image_path: Path, repairs, backup: Backup) -> list[str]:
    xml_path = _annotation_for(detected, image_path, ".xml")
    if not xml_path.is_file():
        return [f"{xml_path.name}：文件不存在"]
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError) as exc:
        return [f"{xml_path.name}：{exc}"]
    root = tree.getroot()
    objects = root.findall("object")

    # Clamping and rounding are applied further down, where the image
    # dimensions they need have been read.
    drop: set[int] = set()
    for repair in repairs:
        # A duplicate names its whole group, and the group is precisely what
        # must not all go: handled below, where one member is kept.
        if repair.rule in _GROUP_RULES["voc"]:
            continue
        if repair.action == DROP_OBJECT:
            drop.update(_indices(repair))
    # A duplicate keeps the first; a near-duplicate keeps the first of the pair.
    for repair in repairs:
        if repair.rule == "voc.box.duplicate":
            drop.update(_indices(repair)[1:])
        elif repair.rule == "voc.box.near_duplicate":
            second = repair.data.get("second_index")
            if isinstance(second, int):
                drop.add(second)

    dimensions = _image_dimensions(image_path)
    for repair in repairs:
        if repair.action == CLAMP_OBJECT and dimensions is not None:
            _clamp_objects(objects, _indices(repair), dimensions, drop)
        elif repair.action == ROUND_OBJECT:
            _round_objects(objects, _indices(repair), drop)

    for index in sorted(drop, reverse=True):
        if 1 <= index <= len(objects):
            root.remove(objects[index - 1])

    for repair in repairs:
        if repair.action == PATCH_HEADER:
            _patch_voc_header(root, image_path, dimensions)

    _atomic_write(xml_path, ET.tostring(root, encoding="utf-8", xml_declaration=True),
                  backup)
    return []


def _patch_voc_header(root, image_path: Path, dimensions) -> None:
    """Write the header fields the file's own image disagrees with.

    Each field has exactly one true value, which is why these are the fixes
    that can be made without asking: the image is the authority.
    """
    if dimensions is None:
        return
    width, height, depth = dimensions
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    for name, value in (("width", width), ("height", height), ("depth", depth)):
        node = size.find(name)
        if node is None:
            node = ET.SubElement(size, name)
        node.text = str(value)
    filename = root.find("filename")
    if filename is not None and (filename.text or "").strip():
        filename.text = image_path.name
    elif filename is None:
        node = ET.Element("filename")
        node.text = image_path.name
        root.insert(0, node)


def _clamp_objects(objects, indices: Iterable[int], dimensions, drop: set[int]) -> None:
    """Pull an out-of-bounds box back inside the image.

    Clamping is a guess about intent, so it is deliberately the mildest repair
    available; a box that has nothing left inside the image afterwards is
    dropped instead of being written as a zero-area rectangle.
    """
    width, height, _depth = dimensions
    for index in indices:
        if not 1 <= index <= len(objects):
            continue
        box = objects[index - 1].find("bndbox")
        if box is None:
            continue
        values = {}
        for name in ("xmin", "ymin", "xmax", "ymax"):
            try:
                values[name] = float(box.findtext(name, "0"))
            except (TypeError, ValueError):
                values[name] = 0.0
        values["xmin"] = max(0.0, min(values["xmin"], width))
        values["xmax"] = max(0.0, min(values["xmax"], width))
        values["ymin"] = max(0.0, min(values["ymin"], height))
        values["ymax"] = max(0.0, min(values["ymax"], height))
        if values["xmax"] <= values["xmin"] or values["ymax"] <= values["ymin"]:
            drop.add(index)
            continue
        for name, value in values.items():
            node = box.find(name)
            if node is None:
                node = ET.SubElement(box, name)
            node.text = str(round(value))


def _round_objects(objects, indices: Iterable[int], drop: set[int]) -> None:
    for index in indices:
        if not 1 <= index <= len(objects):
            continue
        box = objects[index - 1].find("bndbox")
        if box is None:
            continue
        values = {}
        for name in ("xmin", "ymin", "xmax", "ymax"):
            try:
                values[name] = round(float(box.findtext(name, "0")))
            except (TypeError, ValueError):
                values[name] = 0
        if values["xmax"] <= values["xmin"] or values["ymax"] <= values["ymin"]:
            drop.add(index)
            continue
        for name, value in values.items():
            node = box.find(name)
            if node is not None:
                node.text = str(value)


# ---- YOLO ------------------------------------------------------------------


def _apply_yolo(detected, grouped, backup: Backup, presets=None) -> list[str]:
    errors: list[str] = []
    for target, repairs in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        actions = {repair.action for repair in repairs}
        try:
            if DROP_CACHE in actions:
                backup.remove(target, "陈旧缓存")
                continue
            if WRITE_CLASSES in actions:
                continue  # one file for the whole dataset, written below
            if actions <= {DROP_ANNOTATION}:
                backup.remove(target, "孤立标注")
                continue
            if DROP_IMAGE in actions:
                _drop_image_pair(detected, target, ".txt", backup, errors)
                continue
            errors.extend(_edit_yolo(detected, target, repairs, backup))
        except OSError as exc:
            errors.append(f"{target.name}：{exc}")

    writes_classes = any(
        repair.action == WRITE_CLASSES
        for repairs in grouped.values() for repair in repairs
    )
    if writes_classes:
        errors.extend(_write_classes(detected, presets))
    return errors


def _write_classes(detected, presets) -> list[str]:
    """Write the class table the rows are already keyed to.

    Only reachable when presets supplied the table and the dataset has none,
    so this records what is already true rather than deciding anything: the
    row's integer class id already means ``presets[id]`` to every reader that
    is given the project, including this app.
    """
    names = [preset.name for preset in presets or []]
    if not names:
        return ["没有可写入的类别表：项目预设为空"]
    path = detected.annotation_dir / "classes.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    except OSError as exc:
        return [f"classes.txt：{exc}"]
    return []


def _edit_yolo(detected, image_path: Path, repairs, backup: Backup) -> list[str]:
    txt_path = _annotation_for(detected, image_path, ".txt")
    if not txt_path.is_file():
        return [f"{txt_path.name}：文件不存在"]
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [f"{txt_path.name}：{exc}"]

    lines = text.splitlines()
    drop: set[int] = set()
    for repair in repairs:
        if repair.rule in _GROUP_RULES["yolo"]:
            continue
        if repair.action == DROP_OBJECT:
            drop.update(_lines(repair))
    for repair in repairs:
        if repair.rule == "yolo.row.duplicate":
            drop.update(_lines(repair)[1:])
        elif repair.rule == "yolo.row.near_duplicate":
            second = repair.data.get("second")
            if isinstance(second, int):
                drop.add(second)

    kept = [
        raw for number, raw in enumerate(lines, start=1)
        if number not in drop
    ]
    # Every line gone means the file has no boxes left, which is the image
    # rule's business, not this one: leave the empty file and let the next
    # pass see the image as unusable.
    _atomic_write(txt_path, "".join(f"{line}\n" for line in kept).encode("utf-8"), backup)
    return []


# ---- COCO ------------------------------------------------------------------


def _apply_coco(detected, grouped, backup: Backup, presets=None) -> list[str]:
    errors: list[str] = []
    for target, repairs in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        # The file goes now; the record goes with the JSON rewrite below, so
        # both halves of one repair land in the same manifest. A delete that
        # failed takes its record out of that rewrite: the record describes a
        # file that is still there, and dropping it would strand the file.
        names: Mapping[int, str] = {}
        if target.suffix.lower() == ".json" and any(
            repair.action == DROP_IMAGE for repair in repairs
        ):
            names = _coco_file_names(target)

        survivors: list[Repair] = []
        for repair in repairs:
            if repair.action != DROP_IMAGE:
                survivors.append(repair)
                continue
            path = _coco_image_file(detected, target, repair, names)
            if path is not None:
                backup.remove(path, "无可用标注的图片")
                if path.exists():
                    errors.append(f"{path.name}：图片未能删除，JSON 记录保留")
                    continue
            survivors.append(repair)

        if target.suffix.lower() != ".json":
            # ``coco.image.on_disk_unregistered`` names the image itself, and
            # that is the whole repair: the JSON has no record to drop.
            continue
        try:
            errors.extend(_edit_coco(detected, target, survivors, backup))
        except OSError as exc:
            errors.append(f"{target.name}：{exc}")
    return errors


def _coco_image_file(detected, target: Path, repair,
                     names: Mapping[int, str]) -> Path | None:
    """The file one delete repair names, when that file is on disk.

    Two shapes arrive here. A repair on the JSON carries only a record id, so
    the file name has to be read back out of the document. A repair on the
    image itself -- an image the JSON never registered -- already names the
    file, and there is no record to look up.
    """
    if target.suffix.lower() != ".json":
        return target if target.is_file() else None

    name = repair.data.get("file_name")
    if not isinstance(name, str) or not name:
        identifier = repair.data.get("id")
        name = names.get(identifier) if isinstance(identifier, int) else None
    if not isinstance(name, str) or not name:
        return None
    path = detected.image_dir / name
    return path if path.is_file() else None


def _coco_file_names(json_path: Path) -> dict[int, str]:
    """``image id -> file_name``, for turning a record into a file."""
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(document, dict):
        return {}
    names: dict[int, str] = {}
    for record in document.get("images", []):
        if not isinstance(record, dict):
            continue
        identifier = _as_int(record.get("id"))
        name = record.get("file_name")
        if identifier is not None and isinstance(name, str) and name:
            names[identifier] = name
    return names


def _edit_coco(detected, json_path: Path, repairs, backup: Backup) -> list[str]:
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{json_path.name}：{exc}"]
    if not isinstance(document, dict):
        return [f"{json_path.name}：根节点不是对象"]

    images = [item for item in document.get("images", []) if isinstance(item, dict)]
    annotations = [item for item in document.get("annotations", []) if isinstance(item, dict)]

    drop_annotations: set[int] = set()
    drop_images: set[int] = set()
    clamp: set[int] = set()
    recompute: set[int] = set()
    dedupe_images = False
    dedupe_annotations = False

    for repair in repairs:
        action = repair.action
        if repair.rule in _GROUP_RULES["coco"]:
            continue  # names a whole group; only some of it goes, below
        if action == DROP_RECORD:
            drop_annotations.update(_ids(repair))
        elif action == DROP_IMAGE:
            identifier = repair.data.get("id")
            if isinstance(identifier, int):
                drop_images.add(identifier)
        elif action == CLAMP_OBJECT:
            clamp.update(_ids(repair))
        elif action == RECOMPUTE_AREA:
            recompute.update(_ids(repair))
        elif action == DROP_DUPLICATE_IDS:
            if repair.rule == "coco.image.duplicate_id":
                dedupe_images = True
            else:
                dedupe_annotations = True

    for repair in repairs:
        if repair.rule == "coco.annotation.duplicate":
            # Keep the first of the group, drop the rest.
            drop_annotations.update(_ids(repair)[1:])

    dimensions = _record_dimensions(images)

    if dedupe_images:
        images, duplicates = _dedupe_by_id(images)
        drop_images.update(duplicates)
    if dedupe_annotations:
        annotations, duplicates = _dedupe_by_id(annotations)
        drop_annotations.update(duplicates)

    keep_images: list[dict] = []
    for record in images:
        identifier = _as_int(record.get("id"))
        if identifier in drop_images:
            continue
        keep_images.append(record)
    kept_ids = {_as_int(record.get("id")) for record in keep_images}

    keep_annotations: list[dict] = []
    for row in annotations:
        identifier = _as_int(row.get("id"))
        if identifier in drop_annotations:
            continue
        if _as_int(row.get("image_id")) not in kept_ids:
            # Its image is gone, so the annotation has nothing to point at.
            continue
        if identifier in clamp:
            _clamp_coco(row, dimensions)
        if identifier in recompute or identifier in clamp:
            _recompute_area(row)
        keep_annotations.append(row)

    document["images"] = keep_images
    document["annotations"] = keep_annotations
    _atomic_write(
        json_path,
        json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"),
        backup,
    )
    return []


def _record_dimensions(images) -> dict[int, tuple[int, int]]:
    dimensions: dict[int, tuple[int, int]] = {}
    for record in images:
        identifier = _as_int(record.get("id"))
        width = _as_int(record.get("width"))
        height = _as_int(record.get("height"))
        if identifier is not None and width and height:
            dimensions[identifier] = (width, height)
    return dimensions


def _clamp_coco(row: dict, dimensions: dict[int, tuple[int, int]]) -> None:
    bbox = row.get("bbox")
    image_id = _as_int(row.get("image_id"))
    if not isinstance(bbox, list) or len(bbox) != 4 or image_id not in dimensions:
        return
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return
    image_width, image_height = dimensions[image_id]
    x = max(0.0, min(x, image_width))
    y = max(0.0, min(y, image_height))
    width = max(0.0, min(width, image_width - x))
    height = max(0.0, min(height, image_height - y))
    row["bbox"] = [x, y, width, height]


def _recompute_area(row: dict) -> None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    try:
        _x, _y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return
    row["area"] = width * height


def _dedupe_by_id(rows) -> tuple[list[dict], set[int]]:
    seen: set = set()
    kept: list[dict] = []
    dropped: set[int] = set()
    for row in rows:
        identifier = _as_int(row.get("id"))
        if identifier is None:
            kept.append(row)
            continue
        if identifier in seen:
            dropped.add(identifier)
            continue
        seen.add(identifier)
        kept.append(row)
    return kept, dropped


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---- shared helpers --------------------------------------------------------


def _annotation_for(detected, image_path: Path, suffix: str) -> Path:
    """The annotation file an image mirrors onto.

    Mirrors the layout instead of searching for it, so a nested image folder
    (``images/train/a.jpg``) lands on its own nested label folder.
    """
    try:
        relative = image_path.relative_to(detected.image_dir)
    except ValueError:
        relative = Path(image_path.name)
    return detected.annotation_dir / relative.with_suffix(suffix)


def _image_dimensions(path: Path) -> tuple[int, int, int] | None:
    from PIL import Image

    from src.utils.pixels import channel_count

    try:
        with Image.open(path) as image:
            return image.size[0], image.size[1], channel_count(image)
    except Exception:
        return None


def _indices(repair: Repair) -> list[int]:
    value = repair.data.get("index")
    if isinstance(value, int):
        return [value]
    values = repair.data.get("indices")
    if isinstance(values, (list, tuple)):
        return [item for item in values if isinstance(item, int)]
    return []


def _ids(repair: Repair) -> list[int]:
    value = repair.data.get("id")
    values = repair.data.get("ids")
    if isinstance(values, (list, tuple)):
        return [item for item in values if isinstance(item, int)]
    return [value] if isinstance(value, int) else []


def _lines(repair: Repair) -> list[int]:
    value = repair.data.get("line")
    if isinstance(value, int):
        return [value]
    values = repair.data.get("lines")
    if isinstance(values, (list, tuple)):
        return [item for item in values if isinstance(item, int)]
    return []


def _atomic_write(path: Path, payload: bytes, backup: Backup) -> None:
    """Replace a file in one step, keeping a copy of what it was.

    The old content is backed up here rather than by the caller: a rewrite is
    as destructive as a delete, and the manifest should carry the file either
    way.
    """
    if path.is_file():
        backup.save(path, "改写前", kind="rewritten")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_manifest(report: FixReport, backup: Backup, policy: FixPolicy) -> Path:
    """Record everything the run did, next to the backup it made."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = report.backup_dir or (backup.directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cleanup-{stamp}.json"
    document = {
        "root": str(report.root),
        "finished": stamp,
        "passes": report.passes,
        "policy": sorted(policy.rules),
        "repairs": [
            {
                "rule": repair.rule,
                "action": repair.action,
                "target": repair.target.name,
                "data": {key: str(value) for key, value in repair.data.items()},
            }
            for repair in report.repairs
        ],
        # Every file in the backup directory is accounted for here: what was
        # removed, what was rewritten, and what a failed delete left a copy of.
        "removed": report.removed,
        "rewritten": report.rewritten,
        "kept": report.kept,
        "errors": report.errors,
        "remaining": report.remaining.counts() if report.remaining else None,
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return path
