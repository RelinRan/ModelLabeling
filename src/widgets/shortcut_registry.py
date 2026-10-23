from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortcutSpec:
    """One source of truth for menu labels, help text and Qt bindings."""

    key: str
    zh_name: str
    en_name: str
    display: str
    qt_sequences: tuple[str, ...]

    def name(self, english: bool) -> str:
        return self.en_name if english else self.zh_name

    def menu_text(self, english: bool) -> str:
        return f"{self.name(english)}  {self.display}"


SHORTCUTS: dict[str, ShortcutSpec] = {
    "new": ShortcutSpec("new", "新建", "New", "Ctrl+N", ("Ctrl+N",)),
    "open": ShortcutSpec("open", "打开", "Open", "Ctrl+O", ("Ctrl+O",)),
    "history": ShortcutSpec("history", "历史", "History", "Ctrl+H", ("Ctrl+H",)),
    "save": ShortcutSpec("save", "保存", "Save", "Ctrl+S", ("Ctrl+S",)),
    "quit": ShortcutSpec("quit", "退出", "Exit", "Ctrl+Q", ("Ctrl+Q",)),
    "label_groups": ShortcutSpec("label_groups", "标签分组", "Label Groups", "Ctrl+L+G", ("Ctrl+L, G", "Ctrl+L, Ctrl+G")),
    "keypoint_groups": ShortcutSpec("keypoint_groups", "点位类型", "Keypoint Types", "Ctrl+K+G", ("Ctrl+K, G", "Ctrl+K, Ctrl+G")),
    "settings": ShortcutSpec("settings", "应用设置", "Application Settings", "Ctrl+A+S", ("Ctrl+A, S", "Ctrl+A, Ctrl+S")),
    "file_filter": ShortcutSpec("file_filter", "文件筛选", "File Filter", "Ctrl+I+F", ("Ctrl+I, F", "Ctrl+I, Ctrl+F")),
    "annotation_assist": ShortcutSpec("annotation_assist", "标注辅助", "Annotation Assist", "Ctrl+C+A", ("Ctrl+C, A", "Ctrl+C, Ctrl+A")),
    "fit": ShortcutSpec("fit", "适应画布", "Fit Canvas", "Ctrl+0", ("Ctrl+0",)),
    "zoom_in": ShortcutSpec("zoom_in", "图片放大", "Image Zoom In", "Ctrl++", ("Ctrl++",)),
    "zoom_out": ShortcutSpec("zoom_out", "图片缩小", "Image Zoom Out", "Ctrl+-", ("Ctrl+-",)),
    "previous": ShortcutSpec("previous", "上张图片", "Previous Image", "A/↑", ("A", "Up")),
    "next": ShortcutSpec("next", "下张图片", "Next Image", "D/↓", ("D", "Down")),
    "statistics": ShortcutSpec("statistics", "数据统计", "Statistics", "Ctrl+T+S", ("Ctrl+T, S",)),
    "conversion": ShortcutSpec("conversion", "数据转换", "Dataset Conversion", "Ctrl+D+C", ("Ctrl+D, C",)),
    "auto_label": ShortcutSpec("auto_label", "自动标注", "Auto Label", "Ctrl+A+L", ("Ctrl+A, L", "Ctrl+A, Ctrl+L")),
    "video_frames": ShortcutSpec("video_frames", "视频提帧", "Extract Video Frames", "Ctrl+V+F", ("Ctrl+V, F", "Ctrl+V, Ctrl+F")),
    "dataset_synthesis": ShortcutSpec("dataset_synthesis", "数据合成", "Synthesize Dataset", "Ctrl+D+S", ("Ctrl+D, S", "Ctrl+D, Ctrl+S")),
    "dataset_compare": ShortcutSpec("dataset_compare", "数据对比", "Compare Dataset", "Ctrl+D+P", ("Ctrl+D, P", "Ctrl+D, Ctrl+P")),
    "cleanup": ShortcutSpec("cleanup", "清理数据", "Clean Dataset", "Ctrl+C+D", ("Ctrl+C, D", "Ctrl+C, Ctrl+D")),
}


def shortcut(key: str) -> ShortcutSpec:
    return SHORTCUTS[key]


COMMAND_HANDLERS = {
    "new": "open_dataset_init", "open": "open_directory", "history": "open_history",
    "save": "save_current", "quit": "close", "label_groups": "open_label_groups",
    "keypoint_groups": "open_keypoint_groups", "settings": "open_settings",
    "file_filter": "open_image_filter", "annotation_assist": "open_crosshair",
    "statistics": "open_statistics", "conversion": "open_conversion",
    "auto_label": "auto_label_all", "video_frames": "open_video_frames",
    "dataset_synthesis": "open_dataset_synthesis", "dataset_compare": "open_dataset_compare",
    "cleanup": "open_cleanup",
}


def chord_commands() -> dict[tuple[str, str], str]:
    """Map two-stroke letter chords to their window handler names."""
    commands = {}
    for command, handler in COMMAND_HANDLERS.items():
        for sequence in shortcut(command).qt_sequences:
            strokes = sequence.split(", ")
            if len(strokes) == 2:
                commands[tuple(stroke.removeprefix("Ctrl+") for stroke in strokes)] = handler
    return commands


def shortcut_rows(english: bool) -> list[tuple[str, str]]:
    """Rows for the shortcut help dialog, in the same order as the menus."""
    order = (
        "new", "open", "history", "save", "quit", "fit", "zoom_in", "zoom_out",
        "previous", "next", "label_groups", "keypoint_groups", "settings",
        "file_filter", "annotation_assist", "statistics", "conversion",
        "auto_label", "video_frames", "dataset_synthesis", "dataset_compare", "cleanup",
    )
    rows = [(shortcut(key).display, shortcut(key).name(english)) for key in order]
    rows.extend([
        ("Mouse wheel" if english else "鼠标滚轮", "Zoom toward cursor" if english else "以光标为中心缩放"),
        ("W", "Toggle drawing" if english else "切换绘制状态"),
        ("1-9", "Select bound label" if english else "选择绑定标签"),
        ("Ctrl+1-9", "Bind selected label" if english else "绑定当前标签"),
        ("Shift + drag" if english else "Shift + 拖动", "Constrain square" if english else "绘制正方形"),
        ("Enter / double click" if english else "Enter / 双击", "Finish polygon or keypoints" if english else "完成多边形或关键点"),
        ("Backspace", "Remove last point while drawing" if english else "绘制时撤销上一个点"),
        ("Delete / Backspace", "Delete selected annotation" if english else "删除选中标注"),
        ("Esc", "Cancel current shape / exit drawing" if english else "取消当前图形 / 退出绘制"),
        ("Ctrl+Z / Ctrl+Y", "Undo / redo" if english else "撤销 / 重做"),
    ])
    return rows
