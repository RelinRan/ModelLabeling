from __future__ import annotations


LANGUAGES = {"zh_CN": "中文", "en_US": "English"}

_TEXT = {
    "zh_CN": {
        "settings": "应用设置",
        "language": "应用语言",
        "label_settings": "标签设置",
        "operations": "标注操作",
        "statistics": "数据统计",
        "open": "打开",
        "save": "保存",
        "previous": "上一张",
        "next": "下一张",
        "zoom_in": "放大",
        "zoom_out": "缩小",
        "fit": "适应",
        "conversion": "转换",
        "auto_label": "自动标注",
        "add": "新增",
        "edit": "编辑",
        "delete": "删除",
        "color": "颜色",
        "open_dataset": "请打开图片目录或数据集目录",
        "no_statistics": "请先打开图片目录后再查看统计",
        "no_image": "当前没有打开图片",
        "progress": "标注进度",
        "current_file": "当前文件",
        "current_labels": "当前图片标签",
        "none": "暂无",
        "close": "关闭",
        "default_group": "默认标签",
    },
    "en_US": {
        "settings": "Annotation Settings",
        "language": "App language",
        "label_settings": "Label Settings",
        "operations": "Annotation Operations",
        "statistics": "Statistics",
        "open": "Open",
        "save": "Save",
        "previous": "Previous",
        "next": "Next",
        "zoom_in": "Zoom In",
        "zoom_out": "Zoom Out",
        "fit": "Fit",
        "conversion": "Convert",
        "auto_label": "Auto Label",
        "add": "Add",
        "edit": "Edit",
        "delete": "Delete",
        "color": "Color",
        "open_dataset": "Open an image directory or dataset directory",
        "no_statistics": "Open an image directory before viewing statistics",
        "no_image": "No image is open",
        "progress": "Labeling progress",
        "current_file": "Current file",
        "current_labels": "Labels in current image",
        "none": "None",
        "close": "Close",
        "default_group": "Default Labels",
    },
}


def text(key: str, language: str = "zh_CN") -> str:
    return _TEXT.get(language, _TEXT["zh_CN"]).get(key, key)
