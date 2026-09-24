from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QStackedWidget, QVBoxLayout, QWidget


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
            ("Ctrl+L+G", "标签分组"), ("Ctrl+K+G", "点位类型"), ("Ctrl + A + S", "参数设置"),
            ("Ctrl+I+F", "图片筛选"), ("Ctrl+C+A", "标注辅助"),
            ("Ctrl+T+S", "数据统计"), ("Ctrl+D+C", "数据转换"), ("Ctrl+A+L", "自动标注"),
            ("Ctrl+V+F", "视频提帧"), ("Ctrl+D+S", "数据合成"),
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
            ("功能", "六种数据集合任务：YOLO 检测/分割/关键点/旋转框、Pascal VOC、COCO"),
            ("标注", "支持矩形、正方形、多边形绘制、拖拽调整与快捷操作"),
            ("能力", "工作空间式新建向导、大数据集合秒开、ONNX 自动标注、批量转换与统计"),
            ("版本", "v1.0.1"),
            ("作者", "RelinRan"),
            ("GitHub", "https://github.com/RelinRan"),
            ("Email", "relinran@foxmail.com"),
        ])
        layout.addLayout(table)


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("Shortcuts" if english else "\u5feb\u6377\u6309\u952e")
        self.setFixedWidth(420)
        en_rows = [
            ("Ctrl+N", "New dataset"), ("Ctrl+O", "Open dataset"), ("Ctrl+H", "History"),
            ("Ctrl+S", "Save"), ("Ctrl+Q", "Exit"),
            ("A / Up", "Previous image"), ("D / Down", "Next image"),
            ("Mouse wheel", "Zoom toward cursor"), ("Ctrl+0", "Fit canvas"),
            ("Ctrl++ / Ctrl+-", "Zoom in / out"), ("W", "Toggle drawing"),
            ("1-9", "Select bound label"), ("Ctrl+1-9", "Bind selected label"),
            ("Shift + drag", "Constrain square"),
            ("Enter / double click", "Finish polygon or keypoints"),
            ("Backspace", "Remove last point while drawing"),
            ("Esc", "Cancel current shape / exit drawing"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
            ("Delete", "Delete selected annotation"),
            ("Ctrl+L+G", "Label groups"), ("Ctrl+K+G", "Keypoint types"),
            ("Ctrl+A+S", "Settings"), ("Ctrl+I+F", "File filter"), ("Ctrl+C+A", "Annotation assist"),
            ("Ctrl+T+S", "Statistics"), ("Ctrl+D+C", "Dataset conversion"),
            ("Ctrl+V+F", "Extract video frames"), ("Ctrl+D+S", "Synthesize dataset"),
            ("Ctrl+D+P", "Compare dataset"),
            ("Ctrl+A+L", "Auto labeling"),
        ]
        zh_rows = [
            ("Ctrl+N", "\u65b0\u5efa\u6570\u636e\u96c6"), ("Ctrl+O", "\u6253\u5f00\u6570\u636e\u96c6"), ("Ctrl+H", "\u5386\u53f2"),
            ("Ctrl+S", "\u4fdd\u5b58"), ("Ctrl+Q", "\u9000\u51fa"),
            ("A / \u2191", "\u4e0a\u4e00\u5f20\u56fe\u7247"), ("D / \u2193", "\u4e0b\u4e00\u5f20\u56fe\u7247"),
            ("\u9f20\u6807\u6eda\u8f6e", "\u4ee5\u5149\u6807\u4e3a\u4e2d\u5fc3\u7f29\u653e"), ("Ctrl+0", "\u9002\u914d\u753b\u5e03"),
            ("Ctrl++ / Ctrl+-", "\u653e\u5927 / \u7f29\u5c0f"), ("W", "\u5207\u6362\u7ed8\u5236\u72b6\u6001"),
            ("1-9", "\u9009\u62e9\u7ed1\u5b9a\u6807\u7b7e"), ("Ctrl+1-9", "\u7ed1\u5b9a\u5f53\u524d\u6807\u7b7e"),
            ("Shift + \u62d6\u52a8", "\u7ed8\u5236\u6b63\u65b9\u5f62"),
            ("Enter / \u53cc\u51fb", "\u5b8c\u6210\u591a\u8fb9\u5f62\u6216\u5173\u952e\u70b9"),
            ("Backspace", "\u7ed8\u5236\u65f6\u64a4\u9500\u4e0a\u4e00\u4e2a\u70b9"),
            ("Esc", "\u53d6\u6d88\u5f53\u524d\u56fe\u5f62 / \u9000\u51fa\u7ed8\u5236"),
            ("Ctrl+Z / Ctrl+Y", "\u64a4\u9500 / \u91cd\u505a"),
            ("Delete", "\u5220\u9664\u9009\u4e2d\u6807\u6ce8"),
            ("Ctrl+L+G", "\u6807\u7b7e\u5206\u7ec4"), ("Ctrl+K+G", "\u5173\u952e\u70b9\u7c7b\u578b"),
            ("Ctrl+A+S", "\u53c2\u6570\u8bbe\u7f6e"), ("Ctrl+I+F", "\u6587\u4ef6\u7b5b\u9009"), ("Ctrl+C+A", "\u6807\u6ce8\u8f85\u52a9"),
            ("Ctrl+T+S", "\u6570\u636e\u7edf\u8ba1"), ("Ctrl+D+C", "\u6570\u636e\u8f6c\u6362"),
            ("Ctrl+V+F", "\u89c6\u9891\u63d0\u5e27"), ("Ctrl+D+S", "\u6570\u636e\u5408\u6210"),
            ("Ctrl+D+P", "\u6570\u636e\u5bf9\u6bd4"), ("Ctrl+A+L", "\u81ea\u52a8\u6807\u6ce8"),
        ]
        self.setLayout(_table_layout(en_rows if english else zh_rows, spacing=7))


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("About Software" if english else "\u5173\u4e8e\u8f6f\u4ef6")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        title = QLabel("ModelLabeling")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        rows = ([
            ("Product", "Desktop image annotation workbench"),
            ("Formats", "YOLO detection / segmentation / pose / OBB, Pascal VOC, COCO"),
            ("Annotation", "Rectangle, square, polygon, rotated box, keypoints; undo/redo and continuous drawing"),
            ("Tools", "Dataset wizard, large-dataset indexing, ONNX auto labeling, conversion, statistics, video frame extraction and dataset synthesis"),
            ("Version", "v1.0.1"), ("Author", "RelinRan"),
            ("GitHub", "https://github.com/RelinRan"), ("Email", "relinran@foxmail.com"),
        ] if english else [
            ("\u4ea7\u54c1", "\u684c\u9762\u7aef\u56fe\u50cf\u6807\u6ce8\u5de5\u4f5c\u53f0"),
            ("\u6570\u636e\u683c\u5f0f", "YOLO \u68c0\u6d4b / \u5206\u5272 / \u5173\u952e\u70b9 / \u65cb\u8f6c\u6846\u3001Pascal VOC\u3001COCO"),
            ("\u6807\u6ce8", "\u77e9\u5f62\u3001\u6b63\u65b9\u5f62\u3001\u591a\u8fb9\u5f62\u3001\u65cb\u8f6c\u6846\u3001\u5173\u952e\u70b9\uff1b\u652f\u6301\u64a4\u9500\u91cd\u505a\u548c\u8fde\u7eed\u7ed8\u5236"),
            ("\u529f\u80fd", "\u6570\u636e\u96c6\u521b\u5efa\u5411\u5bfc\u3001\u5927\u6570\u636e\u96c6\u7d22\u5f15\u3001ONNX \u81ea\u52a8\u6807\u6ce8\u3001\u683c\u5f0f\u8f6c\u6362\u3001\u6570\u636e\u7edf\u8ba1\u3001\u89c6\u9891\u63d0\u5e27\u4e0e\u6570\u636e\u5408\u6210"),
            ("\u7248\u672c", "v1.0.1"), ("\u4f5c\u8005", "RelinRan"),
            ("GitHub", "https://github.com/RelinRan"), ("\u90ae\u7bb1", "relinran@foxmail.com"),
        ])
        layout.addLayout(_table_layout(rows))

class UsageGuideDialog(QDialog):
    """Full user manual: quick start, annotation operations, and the dataset
    format reference (formerly the standalone Format Guide dialog)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("User Guide" if english else "\u4f7f\u7528\u8bf4\u660e")
        self.resize(700, 420)

        sections = self._sections_english() if english else self._sections_chinese()
        self.categories = QListWidget()
        self.categories.setObjectName("settingsCategories")
        self.categories.setFixedWidth(120)
        self.categories.setFrameShape(QListWidget.Shape.NoFrame)
        self.categories.setLineWidth(0)
        self.categories.setMidLineWidth(0)
        self.categories.addItems([title for title, _ in sections])

        self.pages = QStackedWidget()
        for _, content in sections:
            self.pages.addWidget(self._content_page(content))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(0)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(20)
        content.addWidget(self.categories)
        content.addWidget(self.pages, 1)
        layout.addLayout(content)
        self.categories.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.categories.setCurrentRow(0)

    @staticmethod
    def _content_page(content: str) -> QWidget:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setStyleSheet(
            "QPlainTextEdit { background: #25272A; color: #D7DAE0; "
            "border: 1px solid #464A50; border-radius: 5px; "
            "padding: 14px; font-family: Consolas, 'Microsoft YaHei UI'; font-size: 12px; }"
        )
        return editor

    @staticmethod
    def _sections_chinese() -> list[tuple[str, str]]:
        return [
            ("\u5feb\u901f\u5f00\u59cb", """ModelLabeling \u7528\u4e8e\u56fe\u50cf\u6570\u636e\u96c6\u6d4f\u89c8\u4e0e\u6807\u6ce8\u3002

1. \u6253\u5f00\u73b0\u6709\u6570\u636e\u96c6 (Ctrl+O)\uff0c\u6216\u4f7f\u7528\u65b0\u5efa\u6570\u636e\u96c6 (Ctrl+N) \u521b\u5efa\u6807\u51c6\u76ee\u5f55\u3002
2. \u5728\u753b\u5e03\u5de6\u4e0a\u89d2\u9009\u62e9\u77e9\u5f62\u3001\u6b63\u65b9\u5f62\u3001\u591a\u8fb9\u5f62\u3001\u65cb\u8f6c\u6846\u6216\u5173\u952e\u70b9\u6807\u6ce8\u3002
3. \u4f7f\u7528 A/D \u6216\u65b9\u5411\u952e\u5207\u6362\u56fe\u7247\uff1b\u6807\u6ce8\u81ea\u52a8\u4fdd\u5b58\u3002
4. \u4f7f\u7528 Ctrl+Z / Ctrl+Y \u64a4\u9500 / \u91cd\u505a\u3002

\u652f\u6301 YOLO \u68c0\u6d4b\u3001\u5206\u5272\u3001Pose\u3001OBB\uff0cPascal VOC \u548c COCO \u6570\u636e\u96c6\u3002"""),
            ("\u6807\u6ce8\u4e0e\u5feb\u6377\u952e", """\u7ed8\u5236\u65b9\u5f0f
  \u77e9\u5f62 / \u6b63\u65b9\u5f62\uff1a\u6309\u4f4f\u5de6\u952e\u62d6\u52a8
  \u591a\u8fb9\u5f62\uff1a\u9010\u70b9\u70b9\u51fb\uff0c\u53cc\u51fb\u6216 Enter \u7ed3\u675f
  \u65cb\u8f6c\u6846\uff1a\u5148\u62d6\u51fa\u6846\uff0c\u518d\u62d6\u52a8\u65cb\u8f6c\u624b\u67c4
  \u5173\u952e\u70b9\uff1a\u6309\u5f53\u524d\u70b9\u4f4d\u7c7b\u578b\u9010\u4e2a\u70b9\u51fb
  Esc \u53d6\u6d88\u5f53\u524d\u56fe\u5f62\uff1bDelete \u5220\u9664\u9009\u4e2d\u6807\u6ce8\u3002

\u5e38\u7528\u5feb\u6377\u952e
  Ctrl+O \u6253\u5f00  |  Ctrl+S \u4fdd\u5b58  |  Ctrl+H \u5386\u53f2
  Ctrl+L+G \u6807\u7b7e\u5206\u7ec4  |  Ctrl+K+G \u5173\u952e\u70b9\u7c7b\u578b
  Ctrl+T+S \u6570\u636e\u7edf\u8ba1  |  Ctrl+D+C \u6570\u636e\u8f6c\u6362
  Ctrl+A+L \u81ea\u52a8\u6807\u6ce8  |  Ctrl+A+S \u53c2\u6570\u8bbe\u7f6e
"""),
            ("\u89c6\u9891\u5de5\u5177\u4e0e\u6570\u636e\u96c6", """\u89c6\u9891\u63d0\u5e27 (Ctrl+V+F)
  \u9009\u62e9\u89c6\u9891\u6839\u76ee\u5f55\uff08\u4f1a\u9012\u5f52\u626b\u63cf\u5b50\u76ee\u5f55\uff09\u3001\u5e27\u56fe\u8f93\u51fa\u76ee\u5f55\u548c\u62bd\u5e27\u9891\u7387\uff08\u9ed8\u8ba4 5 FPS\uff09\u3002
  \u5904\u7406\u8fdb\u5ea6\u5728\u4e3b\u9875\u5e95\u90e8\u72b6\u6001\u680f\u53f3\u4fa7\u663e\u793a\uff0c\u5b8c\u6210\u540e\u4f1a\u663e\u793a\u7ed3\u679c\u63d0\u793a\u3002

\u6570\u636e\u5408\u6210 (Ctrl+D+S)
  \u9009\u62e9\u5e27\u56fe\u6570\u636e\u76ee\u5f55\u3001\u5408\u6210\u8f93\u51fa\u76ee\u5f55\u53ca COCO / YOLO / Pascal VOC \u683c\u5f0f\u3002
  \u5efa\u8bae\u8f93\u51fa\u76ee\u5f55\u4e3a\u7a7a\uff0c\u907f\u514d\u8986\u76d6\u73b0\u6709\u6570\u636e\u3002
  \u5408\u6210\u8fdb\u5ea6\u4f1a\u663e\u793a\u5728\u4e3b\u9875\u5e95\u90e8\u72b6\u6001\u680f\u53f3\u4fa7\uff0c\u5b8c\u6210\u540e\u663e\u793a\u6210\u529f\u3001\u5931\u8d25\u548c\u8df3\u8fc7\u6570\u91cf\u3002

\u5176\u4ed6\u5de5\u5177
  Ctrl+I+F \u6587\u4ef6\u7b5b\u9009  |  Ctrl+C+A \u6807\u6ce8\u8f85\u52a9
  Ctrl+T+S \u6570\u636e\u7edf\u8ba1  |  Ctrl+A+L \u81ea\u52a8\u6807\u6ce8
""")
        ]
    @staticmethod
    def _sections_english() -> list[tuple[str, str]]:
        return [
            ("Quick Start", """Quick start (three steps)

Step 1  Open or create a dataset
  · Open (Ctrl+O): pick the dataset root; the format is detected
  · New (Ctrl+N): turn an existing image folder into a standard
    dataset inside a workspace and open it

Step 2  Pick an annotation method
  · Use the combo at the canvas top-left: Rectangle / Square /
    Polygon / Rotated Box / Keypoints
  · Only methods supported by the current dataset are listed

Step 3  Draw and save
  · Draw with the conventional gestures (see the next section)
  · Auto save is on by default (~0.3s after each shape)
  · Progress is shown live in the bottom status bar

Tips
  · The first open of a large dataset builds an index; later
    opens are much faster
  · Use A/D or arrows to move between images
  · Ctrl+Z immediately undo an accidental shape
"""),
            ("Annotation", """Annotation methods and operations

General
  Selecting a method keeps it armed for continuous drawing
  Esc            first press drops the current shape, second exits
  Right click    exits drawing; on an annotation opens the editor
  Ctrl+Z / Ctrl+Y    undo / redo (draw, delete, move, relabel)
  Delete / Backspace delete the selected annotation
  Double click   opens the annotation editor

Rectangle
  Drag with the left button (any direction)
  Edit: drag inside to move, corner handles to resize
  Hold Shift while dragging to constrain a square

Square
  Drag; width and height stay equal

Polygon
  Left click each vertex; double click or Enter closes; right
  click closes too; Backspace removes the last vertex
  Edit: drag vertex handles while selected

Rotated Box (YOLO OBB)
  Drag like a rectangle (starts axis aligned)
  Select, then drag the green handle to rotate around the
  center; Shift snaps to 15-degree steps
  Edit: drag inside to move

Keypoints (YOLO Pose / COCO)
  Left click each point; auto-finishes at the schema count,
  double click or Enter finishes early
  The count box beside the method combo changes the schema
  (17 = official COCO person; others auto-name kpt_1..kpt_N
  and sync to kpt_shape in data.yaml)
  Right click a point to cycle visibility: 2 -> 0 -> 1
  Edit: drag points; the editor changes coordinates/visibility

Assist
  W          toggle the last used method
  Mouse wheel  zoom toward the cursor
  1-9        select the label bound to that key; Ctrl+1-9 binds
             the selected label to that key
  Zoom persists across images; Ctrl+0 fits and resets
  Ctrl+C+A   crosshair settings

  The last dataset and image reopen on launch; this can be
  changed to an empty start in Settings -> General.
"""),
            ("Dataset Formats", FormatGuideDialog._english_content()),
            ("New Dataset", """New Dataset wizard (Ctrl+N)

Datasets live in a workspace like IDE projects: the image
source is only material; the real dataset is created under
workspace/<name>/.

Steps
  1. File menu -> New (Ctrl+N)
  2. Choose a workspace (remembered between runs; it can hold
     many datasets)
  3. Enter a dataset name (must be unique in the workspace)
  4. Optional: image source folder
     · A plain folder: images are copied into images/
     · An existing dataset: images AND annotations imported
     · Empty: creates the structure; add images later
  5. Choose the target format:
     YOLO Detection / Segmentation / Pose / OBB, VOC, COCO
  6. Optional class names; keypoint count for pose
  7. Click "Create"

What happens
  · workspace/<name>/ is created with the standard structure
  · Images are copied in (nested folders flattened, name
    collisions prefixed); the source is never modified
  · A duplicate name is rejected instead of overwritten
  · The workspace location is remembered
"""),
            ("More", """More features

Label groups (Ctrl+L+G)
  Reusable label template library kept across datasets

Keypoint types (Ctrl+K+G)
  Predefine keypoint counts and names; pick a type in the canvas top-left before drawing keypoints

File filter (Ctrl+I+F)
  Filter the image list by name, status, or label

Statistics (Ctrl+T+S)
  Label counts, per-class distribution, progress

Conversion (Ctrl+D+C)
  Batch convert between YOLO / VOC / COCO

Extract video frames (Ctrl+V+F)
  Recursively scan a video folder and save sampled frames at the selected FPS. Progress appears at the bottom-right of the main status bar; a completion summary is shown when done.

Synthesize dataset (Ctrl+D+S)
  Build a COCO, YOLO, or Pascal VOC dataset from a frame folder; output must be empty. Progress appears at the bottom-right of the main status bar, followed by a readable success/failure/skip summary.

Auto labeling (Ctrl+A+L)
  Requires an ONNX model in Application Settings; official
  YOLO detection/Pose models are supported; can be stopped

Application settings (Ctrl+A+S)
  Annotation method, line/text size, save mode, language

Full shortcut list: Help -> Shortcuts
"""),
        ]


class FormatGuideDialog(QDialog):


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
按任务不同，每行一个目标，格式为：

检测 (Detection)
class_id  center_x  center_y  width  height
示例：0 0.5125 0.4800 0.2500 0.3600

分割 (Segmentation)
class_id  x1 y1  x2 y2  x3 y3 ...        （多边形顶点）
示例：0 0.10 0.10 0.50 0.10 0.30 0.40

关键点 (Pose)
class_id  cx cy w h  px1 py1 v1  px2 py2 v2 ...
v 为可见性：0 未标注、1 遮挡、2 可见
点位数量由 data.yaml 的 kpt_shape 声明，如 [17, 3]

旋转框 (OBB)
class_id  x1 y1  x2 y2  x3 y3  x4 y4     （4 个角点）

以上坐标均为相对图片尺寸的归一化值（0 到 1）。
任务类型由 data.yaml 的 task 字段或行格式自动识别。

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
One object per line; the row format depends on the task:

Detection
class_id  center_x  center_y  width  height
Example: 0 0.5125 0.4800 0.2500 0.3600

Segmentation
class_id  x1 y1  x2 y2  x3 y3 ...        (polygon vertices)
Example: 0 0.10 0.10 0.50 0.10 0.30 0.40

Pose
class_id  cx cy w h  px1 py1 v1  px2 py2 v2 ...
v is visibility: 0 unlabeled, 1 occluded, 2 visible
The point count comes from kpt_shape in data.yaml, e.g. [17, 3]

OBB (rotated boxes)
class_id  x1 y1  x2 y2  x3 y3  x4 y4     (four corner points)

All coordinates are normalized (0..1). The task is detected from
the data.yaml task field or the row shape.

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
