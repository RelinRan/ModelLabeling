from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QPlainTextEdit, QVBoxLayout


def _table_layout(rows: list[tuple[str, str]], spacing: int = 9) -> QGridLayout:
    layout = QGridLayout()
    layout.setContentsMargins(22, 18, 22, 18)
    layout.setHorizontalSpacing(34)
    layout.setVerticalSpacing(spacing)
    for row, (left, right) in enumerate(rows):
        first = QLabel(left.replace(" ", "") if left in {"Ctrl + A + S", "Ctrl + +"} else left)
        first.setStyleSheet("font-family: Consolas; font-weight: 600;")
        first.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        second = QLabel(right)
        second.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        second.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(first, row, 0)
        layout.addWidget(second, row, 1)
    layout.setColumnStretch(1, 1)
    return layout


class _LegacyShortcutsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("快捷按键")
        self.setMinimumWidth(460)
        layout = _table_layout([
            ("A / \u2191", "\u4e0a\u5f20\u56fe\u7247"), ("D / \u2193", "\u4e0b\u5f20\u56fe\u7247"),
            ("Ctrl+O", "打开数据集"), ("Ctrl+S", "保存标注"), ("Ctrl+Q", "退出应用"),
            ("A", "上张图片"), ("D", "下张图片"), ("Ctrl + +", "图片放大"),
            ("Ctrl+-", "图片缩小"), ("Ctrl+0", "适应画布"), ("W", "启用绘制"),
            ("Ctrl+L+G", "标签分组"), ("Ctrl + A + S", "应用设置"),
            ("Ctrl+I+F", "图片筛选"), ("Ctrl+C+A", "十字辅助"),
            ("Ctrl+D+S", "数据统计"), ("Ctrl+D+C", "数据转换"), ("Ctrl+A+L", "自动标注"),
            ("Delete / Backspace", "删除选中标注"), ("Esc", "取消当前绘制"),
        ])
        self.setLayout(layout)


class _LegacyAboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("关于软件")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        title = QLabel("ModelLabeling")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        table = _table_layout([
            ("产品", "桌面端图像标注工作台"),
            ("功能", "支持 COCO、YOLO、Pascal VOC 数据集标注与互相转换"),
            ("标注", "支持矩形、正方形、多边形绘制、拖拽调整与快捷操作"),
            ("能力", "支持 ONNX 自动标注、进度统计、标签分组和批量转换"),
            ("版本", "v1.0.0"),
            ("作者", "RelinRan"),
            ("GitHub", "https://github.com/RelinRan"),
            ("Email", "relinran@foxmail.com"),
        ])
        layout.addLayout(table)


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("Shortcuts" if english else "快捷按键")
        self.setFixedWidth(380)
        rows = [
            ("A / \u2191", "Previous image" if english else "上张图片"),
            ("D / \u2193", "Next image" if english else "下张图片"),
            ("Ctrl+H", "History" if english else "历史"),
            ("Ctrl+O", "Open" if english else "打开"),
            ("Ctrl+S", "Save" if english else "保存"),
            ("Ctrl+Q", "Exit" if english else "退出"),
            ("Ctrl+0", "Fit canvas" if english else "适应画布"),
            ("Ctrl++", "Zoom in" if english else "图片放大"),
            ("Ctrl+-", "Zoom out" if english else "图片缩小"),
            ("W", "Enable drawing" if english else "启用绘制"),
            ("Ctrl+L+G", "Label groups" if english else "标签分组"),
            ("Ctrl+A+S", "Application settings" if english else "应用设置"),
            ("Ctrl+I+F", "File filter" if english else "文件筛选"),
            ("Ctrl+C+A", "Crosshair" if english else "十字辅助"),
            ("Ctrl+D+S", "Statistics" if english else "数据统计"),
            ("Ctrl+D+C", "Dataset conversion" if english else "数据转换"),
            ("Ctrl+A+L", "Auto labeling" if english else "自动标注"),
            ("Delete / Backspace", "Delete annotation" if english else "删除选中标注"),
            ("Esc", "Cancel drawing" if english else "取消当前绘制"),
        ]
        self.setLayout(_table_layout(rows, spacing=7))


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("About Software" if english else "关于软件")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        title = QLabel("ModelLabeling")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        rows = (
            [
                ("Product", "Desktop image annotation workbench"),
                ("Features", "COCO, YOLO, and Pascal VOC annotation and conversion"),
                ("Annotation", "Rectangle, square, polygon, resize, move, and keyboard shortcuts"),
                ("Capabilities", "ONNX auto labeling, progress statistics, label groups, and batch conversion"),
                ("Version", "v1.0.0"),
                ("Author", "RelinRan"),
                ("GitHub", "https://github.com/RelinRan"),
                ("Email", "relinran@foxmail.com"),
            ]
            if english
            else [
                ("产品", "桌面端图像标注工作台"),
                ("功能", "支持 COCO、YOLO、Pascal VOC 数据集标注与互相转换"),
                ("标注", "支持矩形、正方形、多边形绘制、调整、移动和快捷操作"),
                ("能力", "支持 ONNX 自动标注、进度统计、标签分组和批量转换"),
                ("版本", "v1.0.0"),
                ("作者", "RelinRan"),
                ("GitHub", "https://github.com/RelinRan"),
                ("邮箱", "relinran@foxmail.com"),
            ]
        )
        layout.addLayout(_table_layout(rows))


class FormatGuideDialog(QDialog):
    """Reference for the supported dataset layouts and annotation files."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("Format Guide" if english else "格式说明")
        self.resize(820, 680)

        content = self._english_content() if english else self._chinese_content()
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setStyleSheet(
            "QPlainTextEdit { background: #25272A; color: #D7DAE0; "
            "border: 1px solid #464A50; border-radius: 5px; "
            "padding: 12px; font-family: Consolas, 'Microsoft YaHei UI'; font-size: 12px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(editor)

    @staticmethod
    def _chinese_content() -> str:
        return """ModelLabeling 支持以下数据集格式：YOLO、Pascal VOC、COCO

一、YOLO

推荐目录结构：
dataset/
├─ images/
│  ├─ train/                 图片文件（也支持直接放在 images/ 下）
│  └─ val/
├─ labels/
│  ├─ train/                 与图片同名的 .txt 标注文件
│  └─ val/
├─ classes.txt               每行一个类别名称，行号就是 class_id
└─ data.yaml                 可选，支持 names 字段

单张图片 image_001.jpg 对应 image_001.txt。
每行一个目标，格式为：
class_id  center_x  center_y  width  height

坐标均为相对图片尺寸的归一化值，范围通常为 0 到 1。
示例：
0 0.5125 0.4800 0.2500 0.3600

二、Pascal VOC

推荐目录结构：
dataset/
├─ JPEGImages/               图片文件
├─ Annotations/              与图片同名的 .xml 文件
└─ ImageSets/                可选，数据集划分文件
   └─ Main/
      ├─ train.txt
      └─ val.txt

单张图片 JPEGImages/image_001.jpg 对应 Annotations/image_001.xml。
XML 中常用结构：
<annotation>
  <filename>image_001.jpg</filename>
  <size><width>...</width><height>...</height></size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>...</xmin><ymin>...</ymin>
      <xmax>...</xmax><ymax>...</ymax>
    </bndbox>
  </object>
</annotation>

三、COCO

推荐目录结构：
dataset/
├─ images/                   图片文件
└─ annotations/
   ├─ instances.json         COCO 标注文件
   └─ annotations.json       也支持此命名

COCO 标注文件为 JSON，主要包含：
images       图片信息：id、file_name、width、height
annotations  目标信息：image_id、category_id、bbox
categories   类别信息：id、name

bbox 格式为：
[x, y, width, height]

其中 x、y 是左上角像素坐标，width、height 是框的像素宽高。
图片的 file_name 必须能与 images/ 下的实际文件对应。

目录选择提示
1. 选择数据集根目录，应用会自动识别图片目录、标注目录和格式。
2. YOLO 通常需要 classes.txt 或 data.yaml 解析类别名称。
3. Pascal VOC 要保证图片名和 XML 文件名（不含扩展名）一致。
4. COCO 要保证 JSON 中的 file_name 与实际图片文件对应。
"""

    @staticmethod
    def _english_content() -> str:
        return """ModelLabeling supports YOLO, Pascal VOC, and COCO datasets.

1. YOLO

Recommended layout:
dataset/
|-- images/
|   |-- train/               Image files (images/ may also contain images directly)
|   `-- val/
|-- labels/
|   |-- train/               .txt annotation files with matching image names
|   `-- val/
|-- classes.txt              One class name per line; line number is class_id
`-- data.yaml                Optional; the names field is supported

image_001.jpg matches image_001.txt.
One object per line:
class_id  center_x  center_y  width  height

Coordinates are normalized relative to image width and height, usually 0..1.
Example:
0 0.5125 0.4800 0.2500 0.3600

2. Pascal VOC

Recommended layout:
dataset/
|-- JPEGImages/              Image files
|-- Annotations/             .xml files with matching image names
`-- ImageSets/               Optional dataset split files
    `-- Main/
        |-- train.txt
        `-- val.txt

JPEGImages/image_001.jpg matches Annotations/image_001.xml.
The XML commonly contains filename, size, and object/bndbox elements:
<object>
  <name>person</name>
  <bndbox>
    <xmin>...</xmin><ymin>...</ymin>
    <xmax>...</xmax><ymax>...</ymax>
  </bndbox>
</object>

3. COCO

Recommended layout:
dataset/
|-- images/                  Image files
`-- annotations/
    |-- instances.json       COCO annotation file
    `-- annotations.json     This filename is also supported

The JSON mainly contains:
images       id, file_name, width, height
annotations  image_id, category_id, bbox
categories   id, name

bbox format:
[x, y, width, height]

x and y are the top-left pixel coordinates; width and height are pixel sizes.
file_name must resolve to an actual image under the selected dataset.

Directory selection tips
1. Select the dataset root; the application detects image/annotation folders.
2. YOLO usually needs classes.txt or data.yaml for class names.
3. Pascal VOC image and XML filenames must match apart from the extension.
4. COCO file_name values must match the actual image files.
"""
