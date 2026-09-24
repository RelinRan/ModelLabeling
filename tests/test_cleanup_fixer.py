# -*- coding: utf-8 -*-
"""The repair pass.

What these tests are really protecting:

* a repair is never invented for a finding that has no defensible one --
  deleting a box because its label is not in the current presets would destroy
  real annotation, and is the failure mode that would do the most damage;
* everything removed is copied aside first, and a copy that fails stops the
  delete;
* the pass is idempotent, which is the operational meaning of "clean".
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

from src.models.annotation import LabelPreset
from src.services.cleanup import FixPolicy, fix_dataset, scan_dataset
from src.services.cleanup.fixer import FIXES
from src.services.cleanup.report import RULES, Severity


def presets_for(*names: str) -> list[LabelPreset]:
    return [LabelPreset(name, index, "#00e5ff") for index, name in enumerate(names)]


def defect_stem(slot: int = 0) -> str:
    """The image a single injected defect lands on.

    The fixture packs defects into frames from index 0 in a fixed order, so
    one defect asked for on its own is always the first frame.
    """
    return f"VIDEO_000000000000_0_{152 + slot:06d}"


def tree_fingerprint(root) -> dict:
    from pathlib import Path

    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in Path(root).rglob("*") if path.is_file()
    }


def voc_boxes(root, stem: str) -> list[tuple[int, int, int, int]]:
    """The boxes a VOC file holds, as written."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(root / "Annotations" / f"{stem}.xml")
    return [
        tuple(int(obj.find("bndbox").findtext(field))
              for field in ("xmin", "ymin", "xmax", "ymax"))
        for obj in tree.getroot().findall("object")
    ]


def voc_header(root, stem: str) -> dict:
    import xml.etree.ElementTree as ET

    size = ET.parse(root / "Annotations" / f"{stem}.xml").getroot().find("size")
    return {name: int(size.findtext(name)) for name in ("width", "height", "depth")}


def yolo_rows(root, stem: str) -> list[str]:
    path = root / "labels" / "train" / f"{stem}.txt"
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coco_document(root) -> dict:
    return json.loads((root / "annotations.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the rule table
# --------------------------------------------------------------------------


def test_every_fix_names_a_rule_that_exists():
    assert set(FIXES) <= set(RULES)


def test_structural_policy_never_includes_a_judgement_call():
    """A REVIEW finding depends on annotation policy -- how small is too
    small, whether a box-less image is a negative sample -- so it must never
    ride along with the repairs that are simply correct."""
    policy = FixPolicy.structural()
    for rule in policy.rules:
        assert RULES[rule].severity is Severity.STRUCTURAL, rule


def test_a_corrupt_image_is_not_deleted_without_being_asked(synthetic_dataset):
    """The carve-out that keeps "structural" from meaning "ruthless".

    A truncated image is genuinely broken, so it is a STRUCTURAL finding --
    but the only repair is to delete it, and with it the annotation it
    carried. The image may be recoverable; the annotation will not be.
    """
    assert "image.unreadable" not in FixPolicy.structural().rules
    assert "image.unreadable" in FixPolicy.quality_rules()

    root = synthetic_dataset("corrupt", fmt="voc", frames=6, defects=("unreadable_image",))
    image = next(path for path in (root / "JPEGImages").glob("*.jpg")
                 if path.stat().st_size < 5000)

    fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.structural())
    assert image.is_file()

    fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.quality())
    assert not image.is_file()


def test_a_requested_rule_with_no_repair_is_reported_not_swallowed():
    policy = FixPolicy.of(["voc.box.small", "voc.object.unknown_label", "nonsense"])
    assert policy.allows("voc.box.small")
    assert not policy.allows("voc.object.unknown_label")
    assert policy.unfixable() == {"voc.object.unknown_label", "nonsense"}


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------


def test_dry_run_touches_nothing(synthetic_dataset):
    root = synthetic_dataset("dry", fmt="voc", frames=12,
                             defects=("small_box", "wrong_depth", "orphan_annotation"))
    before = tree_fingerprint(root)
    report = fix_dataset(root, presets=presets_for("obj"),
                         policy=FixPolicy.everything(), dry_run=True)

    assert report.repairs, "the dry run should still describe the repairs"
    assert tree_fingerprint(root) == before
    assert report.backup_dir is None
    assert report.manifest_path is None


def test_every_removed_file_is_copied_aside_first(synthetic_dataset):
    root = synthetic_dataset("backup", fmt="voc", frames=12,
                             defects=("orphan_annotation",))
    report = fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.everything())

    assert report.backup_dir is not None
    assert (report.backup_dir / "Annotations" / "ORPHAN_000001.xml").is_file()
    assert not (root / "Annotations" / "ORPHAN_000001.xml").exists()
    # Beside the dataset, never inside it: anything under the root would be
    # read back as dataset content by the next scan.
    assert root.resolve() not in report.backup_dir.resolve().parents


def test_a_failed_backup_stops_the_delete(synthetic_dataset, monkeypatch):
    """A full disk must become a reported error, not lost annotation."""
    import shutil

    root = synthetic_dataset("no-backup", fmt="voc", frames=12,
                             defects=("orphan_annotation",))
    orphan = root / "Annotations" / "ORPHAN_000001.xml"

    def refuse(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(shutil, "copy2", refuse)
    report = fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.everything())

    assert orphan.is_file(), "the file must survive a failed backup"
    assert any("备份失败" in error for error in report.errors), report.errors


def test_the_backup_holds_the_file_as_the_run_found_it(synthetic_dataset):
    """The copy is the recovery path, so it has to be the original.

    A file the small-box rule empties is then a box-less image, and the next
    pass removes it. Two passes touch it, and the second copy would be of
    what the first pass had already written -- an annotation with no box in
    it, which recovers nothing.
    """
    root = synthetic_dataset("two-touch", fmt="voc", frames=6,
                             defects=("small_box",))
    xml_path = root / "Annotations" / f"{defect_stem()}.xml"
    original = xml_path.read_text(encoding="utf-8")

    report = fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.everything())

    assert not xml_path.exists(), "the emptied image and its annotation should go"
    assert any(entry["path"] == xml_path.relative_to(root).as_posix()
               for entry in report.removed), report.removed
    copy = report.backup_dir / xml_path.relative_to(root)
    assert copy.read_text(encoding="utf-8") == original


def test_every_copy_in_the_backup_is_accounted_for(synthetic_dataset, locked_file):
    """A delete that could not go through leaves a copy behind, and the
    manifest has to say so: it is the index to the backup directory."""
    root = synthetic_dataset("locked", fmt="voc", frames=12,
                             defects=("no_object",))
    # The fixture fills the tail of the frame list with box-less images.
    frame = defect_stem(slot=11)
    image = root / "JPEGImages" / f"{frame}.jpg"
    relative = image.relative_to(root).as_posix()

    with locked_file(image):
        report = fix_dataset(root, presets=presets_for("obj"),
                             policy=FixPolicy.everything())

    assert image.is_file()
    assert any("删除失败" in error for error in report.errors), report.errors
    assert relative not in {entry["path"] for entry in report.removed}
    assert relative in {entry["path"] for entry in report.kept}
    copy = report.backup_dir / image.relative_to(root)
    assert copy.is_file(), "the copy the kept entry refers to must be there"

    document = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert relative not in {entry["path"] for entry in document["removed"]}
    assert relative in {entry["path"] for entry in document["kept"]}


def test_the_manifest_records_what_was_done(synthetic_dataset):
    root = synthetic_dataset("manifest", fmt="voc", frames=12,
                             defects=("wrong_depth", "orphan_annotation"))
    report = fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.everything())

    document = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert document["root"] == str(root)
    assert document["policy"]
    assert any(item["rule"] == "voc.orphan" for item in document["repairs"])
    assert any(item["path"] == "Annotations/ORPHAN_000001.xml"
               for item in document["removed"])


# --------------------------------------------------------------------------
# Pascal VOC
# --------------------------------------------------------------------------


def test_a_small_box_goes_and_then_its_image_follows(synthetic_dataset):
    """The cascade the rules imply, played out over passes rather than
    guessed in one: drop the last box, and the image is now box-less."""
    root = synthetic_dataset("cascade", fmt="voc", frames=12, defects=("small_box",))
    stem = defect_stem()

    # Pass one alone leaves an empty annotation behind ...
    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["voc.box.small"]), max_passes=1)
    assert (root / "Annotations" / f"{stem}.xml").is_file()
    assert voc_boxes(root, stem) == []

    # ... which is what the second rule was waiting for.
    report = fix_dataset(root, presets=presets_for("obj"),
                         policy=FixPolicy.of(["voc.box.small", "voc.annotation.empty"]))
    assert not (root / "Annotations" / f"{stem}.xml").exists()
    assert not (root / "JPEGImages" / f"{stem}.jpg").exists()
    assert report.passes >= 2


def test_a_duplicate_keeps_one_box(synthetic_dataset):
    root = synthetic_dataset("dupe", fmt="voc", frames=12, defects=("duplicate_box",))
    stem = defect_stem()
    assert len(voc_boxes(root, stem)) == 2

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["voc.box.duplicate"]))
    assert voc_boxes(root, stem) == [(120, 120, 300, 340)]


def test_a_near_duplicate_keeps_the_first(synthetic_dataset):
    root = synthetic_dataset("near", fmt="voc", frames=12, defects=("near_duplicate_box",))
    stem = defect_stem()

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["voc.box.near_duplicate"]))
    assert voc_boxes(root, stem) == [(120, 120, 300, 340)]


def test_an_out_of_bounds_box_is_clamped_not_dropped(synthetic_dataset):
    """Clamping is a guess, so it is the mildest repair available -- and it is
    still better than writing a box a convertor cannot encode."""
    root = synthetic_dataset("bounds", fmt="voc", frames=12,
                             defects=("out_of_bounds",), width=640, height=480)
    stem = defect_stem()

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["voc.box.out_of_bounds"]))
    assert voc_boxes(root, stem) == [(0, 0, 640, 480)]


def test_the_declared_header_is_corrected_to_the_image(synthetic_dataset):
    """The writer used to declare <depth>len(".jpg")</depth> -- 4 -- so this
    is the repair that undoes the app's own bug, on files it wrote itself."""
    root = synthetic_dataset("header", fmt="voc", frames=6,
                             defects=("wrong_depth", "size_mismatch"), width=640, height=480)
    stem = defect_stem()
    assert voc_header(root, stem) == {"width": 320, "height": 240, "depth": 4}

    fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.structural())
    assert voc_header(root, stem) == {"width": 640, "height": 480, "depth": 3}


def test_an_unknown_label_never_costs_a_box(synthetic_dataset):
    """The guard that matters most.

    ``known_labels`` comes from the current project's presets, so a dataset
    annotated under different labels reports every single box as unknown.
    Dropping them would delete the whole annotation set of a perfectly good
    dataset -- so there is no automatic repair for this rule at all.
    """
    root = synthetic_dataset("labels", fmt="voc", frames=4, classes=("4R",))
    before = {stem: voc_boxes(root, stem) for stem in (
        defect_stem(i) for i in range(4))}
    assert all(boxes for boxes in before.values())

    report = fix_dataset(root, presets=presets_for("something-else"),
                         policy=FixPolicy.everything())

    for stem, boxes in before.items():
        assert voc_boxes(root, stem) == boxes
    # Even asking for everything gets nothing here: the rule has no repair.
    assert report.repairs == []
    assert report.backup_dir is None


def test_a_geometry_conflict_is_left_for_a_human(synthetic_dataset):
    """Same box, two labels: dropping either would be a coin flip."""
    # Two classes, or the fixture builds two identical "obj" boxes -- which is
    # a duplicate, a different finding with a repair of its own.
    root = synthetic_dataset("conflict", fmt="voc", frames=12,
                             classes=("obj", "4R"), defects=("label_conflict",))
    stem = defect_stem()
    before = voc_boxes(root, stem)
    assert len(before) == 2

    fix_dataset(root, presets=presets_for("obj", "4R"), policy=FixPolicy.everything())
    assert voc_boxes(root, stem) == before
    plan = scan_dataset(root, presets=presets_for("obj", "4R"))
    assert "voc.box.geometry_conflict" in {value.rule for value in plan.issues}


# --------------------------------------------------------------------------
# YOLO
# --------------------------------------------------------------------------


def test_a_small_yolo_row_is_removed(synthetic_dataset):
    root = synthetic_dataset("yolo-small", fmt="yolo", frames=12, defects=("small_box",))
    stem = defect_stem()
    assert len(yolo_rows(root, stem)) == 1

    fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.of(["yolo.row.small"]))
    assert yolo_rows(root, stem) == []


def test_the_class_table_is_written_from_the_presets(synthetic_dataset):
    """The app writes rows keyed to the preset ids and never writes a table,
    so recording the table adds information for other tools without deciding
    anything: the ids already mean exactly this."""
    root = synthetic_dataset("yolo-classes", fmt="yolo", frames=4)
    table = root / "labels" / "classes.txt"
    assert not table.exists()

    fix_dataset(root, presets=presets_for("obj", "4R"),
                policy=FixPolicy.of(["yolo.classes.missing"]))

    # Beside the label folders, not inside one: the table describes every
    # split, and this is where the reader already looks for it.
    assert table.read_text(encoding="utf-8") == "obj\n4R\n"


def test_a_stale_cache_is_deleted(synthetic_dataset):
    root = synthetic_dataset("yolo-cache", fmt="yolo", frames=4)
    cache = root / "labels" / "train" / "labels.cache"
    cache.write_bytes(b"stale")

    fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.of(["yolo.cache.present"]))
    assert not cache.exists()


def test_a_file_beside_the_labels_is_never_deleted(synthetic_dataset):
    """The old orphan pass destroyed a notes.txt kept next to the labels."""
    root = synthetic_dataset("yolo-notes", fmt="yolo", frames=4)
    notes = root / "labels" / "train" / "notes.txt"
    notes.write_text("frames 40-60 skipped, subject left the frame\n", encoding="utf-8")

    report = fix_dataset(root, presets=presets_for("obj"), policy=FixPolicy.everything())
    assert notes.is_file()
    assert not any("notes.txt" in entry["path"] for entry in
                   _manifest_removed(report))


def _manifest_removed(report) -> list:
    if report.manifest_path is None:
        return []
    return json.loads(report.manifest_path.read_text(encoding="utf-8"))["removed"]


# --------------------------------------------------------------------------
# COCO
# --------------------------------------------------------------------------


def test_a_dangling_annotation_is_dropped(synthetic_dataset):
    root = synthetic_dataset("coco-dangling", fmt="coco", frames=4)
    path = root / "annotations.json"
    document = coco_document(root)
    document["annotations"][0]["image_id"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["coco.annotation.dangling_image"]))

    assert len(coco_document(root)["annotations"]) == len(document["annotations"]) - 1


def test_an_area_is_recomputed_rather_than_the_box_removed(synthetic_dataset):
    root = synthetic_dataset("coco-area", fmt="coco", frames=4)
    path = root / "annotations.json"
    document = coco_document(root)
    document["annotations"][0]["area"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["coco.annotation.area_mismatch"]))

    row = coco_document(root)["annotations"][0]
    _x, _y, width, height = row["bbox"]
    assert row["area"] == width * height
    assert len(coco_document(root)["annotations"]) == len(document["annotations"])


def test_a_record_with_no_annotation_takes_its_file_with_it(synthetic_dataset):
    root = synthetic_dataset("coco-empty", fmt="coco", frames=12, defects=("no_object",))
    stem = defect_stem(8)  # no_object takes the tail of the frame range
    assert (root / "images" / f"{stem}.jpg").is_file()

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["coco.image_record.no_annotation"]))

    assert not (root / "images" / f"{stem}.jpg").exists()
    assert all(row["file_name"] != f"{stem}.jpg" for row in coco_document(root)["images"])


def test_an_out_of_bounds_coco_box_is_clamped_and_its_area_follows(synthetic_dataset):
    root = synthetic_dataset("coco-bounds", fmt="coco", frames=12,
                             defects=("out_of_bounds",), width=640, height=480)
    stem = defect_stem()
    identifier = next(row["id"] for row in coco_document(root)["images"]
                      if row["file_name"] == f"{stem}.jpg")

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["coco.annotation.out_of_bounds"]))

    row = next(item for item in coco_document(root)["annotations"]
               if item["image_id"] == identifier)
    assert row["bbox"] == [0.0, 0.0, 640.0, 480.0]
    assert row["area"] == 640 * 480


def test_duplicate_image_ids_keep_the_first_record(synthetic_dataset):
    root = synthetic_dataset("coco-ids", fmt="coco", frames=4)
    path = root / "annotations.json"
    document = coco_document(root)
    document["images"][1]["id"] = document["images"][0]["id"]
    path.write_text(json.dumps(document), encoding="utf-8")

    fix_dataset(root, presets=presets_for("obj"),
                policy=FixPolicy.of(["coco.image.duplicate_id"]))

    ids = [row["id"] for row in coco_document(root)["images"]]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# the property that defines "clean"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,defects,policy", [
    ("voc", ("small_box", "duplicate_box", "wrong_depth", "size_mismatch",
             "orphan_annotation", "no_object"), "everything"),
    ("yolo", ("small_box", "duplicate_box", "orphan_annotation", "no_object"), "everything"),
    ("coco", ("small_box", "duplicate_box", "no_object"), "everything"),
])
def test_fixing_twice_changes_nothing_the_second_time(
        synthetic_dataset, fmt, defects, policy):
    """Idempotence is the operational meaning of "the dataset is clean": if a
    second pass still finds work, the first pass did not finish the job."""
    root = synthetic_dataset(f"idem-{fmt}", fmt=fmt, frames=14, defects=defects)
    presets = presets_for("obj")

    first = fix_dataset(root, presets=presets, policy=FixPolicy.everything())
    assert first.repairs, "the first pass should have had work to do"
    after_first = tree_fingerprint(root)

    second = fix_dataset(root, presets=presets, policy=FixPolicy.everything())

    assert second.repairs == [], [repair.rule for repair in second.repairs]
    assert tree_fingerprint(root) == after_first
    # Nothing removed and nothing rewritten: a second run must not even make
    # a backup directory.
    assert second.manifest_path is None


def test_the_real_dataset_repair_is_idempotent(voc_test):
    """The whole point, on real data.

    A staged slice of the designated dataset has one structural defect --
    every file declares the wrong ``<depth>`` -- and no box is supposed to
    move. Runs on a copy: the shared dataset is never written.
    """
    root = voc_test.stage("depth-repair", limit=60)
    presets = presets_for("4L")

    boxes_before = {path.name: path.read_text(encoding="utf-8").count("<object>")
                    for path in (root / "Annotations").glob("*.xml")}
    report = fix_dataset(root, presets=presets, policy=FixPolicy.structural())

    assert set(report.counts()) == {"voc.depth.mismatch"}, report.counts()
    assert report.errors == []
    # No image was collateral damage, and no box changed.
    assert len(list((root / "JPEGImages").glob("*.jpg"))) == len(boxes_before)
    boxes_after = {path.name: path.read_text(encoding="utf-8").count("<object>")
                   for path in (root / "Annotations").glob("*.xml")}
    assert boxes_after == boxes_before

    after = {value.rule for value in scan_dataset(root, presets=presets).issues}
    assert "voc.depth.mismatch" not in after
    # The labels are not this project's presets, so the finding is expected --
    # and it must not have cost a single box, which the counts above prove.
    assert "voc.object.unknown_label" in after

    second = fix_dataset(root, presets=presets, policy=FixPolicy.structural())
    assert second.repairs == []


def test_a_defective_voc_dataset_scans_clean_afterwards(synthetic_dataset):
    """End to end: every injectable VOC defect that has a repair, fixed, then
    scanned and found free of exactly those findings."""
    fixable = (
        "no_object", "small_box", "duplicate_box", "near_duplicate_box",
        "out_of_bounds", "size_mismatch", "wrong_depth", "orphan_annotation",
    )
    root = synthetic_dataset("e2e-voc", fmt="voc", frames=20, defects=fixable)
    presets = presets_for("obj")
    before = {value.rule for value in scan_dataset(root, presets=presets).issues}
    # A box-less image here has no XML at all rather than an empty one; both
    # mean "nothing to train on" and both take the image with them.
    assert before >= {
        "voc.annotation.missing", "voc.box.small", "voc.box.duplicate",
        "voc.box.near_duplicate", "voc.box.out_of_bounds", "voc.size.mismatch",
        "voc.depth.mismatch", "voc.orphan",
    }, sorted(before)

    report = fix_dataset(root, presets=presets, policy=FixPolicy.everything())
    assert report.errors == [], report.errors

    after = {value.rule for value in scan_dataset(root, presets=presets).issues}
    assert not (after & set(FIXES)), sorted(after)
