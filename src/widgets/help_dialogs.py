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
            ("功能", "六种数据集合任务：YOLO 检测/分割/关键点/旋转框、Pascal VOC、COCO"),
            ("标注", "支持矩形、正方形、多边形绘制、拖拽调整与快捷操作"),
            ("能力", "工作空间式新建向导、大数据集合秒开、ONNX 自动标注、批量转换与统计"),
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
        self.setFixedWidth(400)
        if english:
            rows = [
                ("Ctrl+N", "New dataset"), ("Ctrl+O", "Open"), ("Ctrl+H", "History"),
                ("Ctrl+S", "Save"), ("Ctrl+Q", "Exit"),
                ("A / Up", "Previous image"), ("D / Down", "Next image"),
                ("Mouse wheel", "Zoom toward cursor"), ("Ctrl+0", "Fit canvas"),
                ("Ctrl++ / Ctrl+-", "Zoom in / out"), ("Zoom persists", "Kept across images; Ctrl+0 resets"),
                ("W", "Toggle drawing"), ("1-9", "Pick bound label"), ("Ctrl+1-9", "Bind selected label to key"),
                ("Shift + drag", "Constrain square"),
                ("Enter / double click", "Finish polygon or keypoints"),
                ("Backspace", "Remove last point while drawing"),
                ("Esc", "Drop current shape, then exit drawing"),
                ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                ("Delete / Backspace", "Delete selected annotation"),
                ("Ctrl+L+G", "Label groups"), ("Ctrl+K+G", "Keypoint types"), ("Ctrl+A+S", "Settings"),
                ("Ctrl+I+F", "File filter"), ("Ctrl+C+A", "Annotation assist"),
                ("Ctrl+D+S", "Statistics"), ("Ctrl+D+C", "Dataset conversion"),
                ("Ctrl+A+L", "Auto labeling"),
            ]
        else:
            rows = [
                ("Ctrl+N", "新建数据集"), ("Ctrl+O", "打开"), ("Ctrl+H", "历史"),
                ("Ctrl+S", "保存"), ("Ctrl+Q", "退出"),
                ("A / ↑", "上张图片"), ("D / ↓", "下张图片"),
                ("鼠标滚轮", "缩放画布（以光标为锚点）"), ("Ctrl+0", "适应画布"),
                ("Ctrl++ / Ctrl+-", "放大 / 缩小"), ("缩放保持", "换图不重置；Ctrl+0 重置"),
                ("W", "启用 / 退出绘制"), ("1-9", "选择绑定标签"), ("Ctrl+1-9", "绑定当前标签到该键"),
                ("Shift + 拖拽", "画正方形"),
                ("Enter / 双击", "完成多边形 / 关键点"),
                ("Backspace", "绘制中撤销上一点"),
                ("Esc", "取消当前形状，再按退出绘制"),
                ("Ctrl+Z / Ctrl+Y", "撤销 / 重做"),
                ("Delete / Backspace", "删除选中标注"),
                ("Ctrl+L+G", "标签分组"), ("Ctrl+K+G", "点位类型"), ("Ctrl+A+S", "参数设置"),
                ("Ctrl+I+F", "文件筛选"), ("Ctrl+C+A", "标注辅助"),
                ("Ctrl+D+S", "数据统计"), ("Ctrl+D+C", "数据转换"),
                ("Ctrl+A+L", "自动标注"),
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
                ("Features", "Six dataset tasks: YOLO detect/segment/pose/OBB, Pascal VOC, COCO"),
                ("Annotation", "Rectangle, square, polygon, rotated box, keypoints; undo/redo and continuous drawing"),
                ("Capabilities", "Workspace dataset wizard, SQLite index for large sets, ONNX auto labeling, conversion, statistics"),
                ("Version", "v1.0.0"),
                ("Author", "RelinRan"),
                ("GitHub", "https://github.com/RelinRan"),
                ("Email", "relinran@foxmail.com"),
            ]
            if english
            else [
                ("产品", "桌面端图像标注工作台"),
                ("功能", "六种数据集合任务：YOLO 检测/分割/关键点/旋转框、Pascal VOC、COCO"),
                ("标注", "矩形、正方形、多边形、旋转框、关键点五种方式；撤销重做与连续标注"),
                ("能力", "工作空间式新建向导、大数据集合秒开、ONNX 自动标注、批量转换与统计"),
                ("版本", "v1.0.0"),
                ("作者", "RelinRan"),
                ("GitHub", "https://github.com/RelinRan"),
                ("邮箱", "relinran@foxmail.com"),
            ]
        )
        layout.addLayout(_table_layout(rows))


class UsageGuideDialog(QDialog):
    """Full user manual: quick start, annotation operations, and the dataset
    format reference (formerly the standalone Format Guide dialog)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        english = getattr(getattr(parent, "settings", None), "language", "zh_CN") == "en_US"
        self.setWindowTitle("User Guide" if english else "使用说明")
        self.resize(940, 700)

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
            ("快速上手", """快速上手（三步开始标注）

第一步  打开或新建数据集
  · 打开 (Ctrl+O)：选择数据集根目录，格式自动识别
  · 新建 (Ctrl+N)：已有图片文件夹时使用，选择目标格式后
    自动创建标准目录结构（不移动、不修改任何图片），
    并直接打开开始标注

第二步  选择标注方式
  · 画布左上角下拉框选择：矩形 / 正方形 / 多边形 / 旋转框 / 关键点位
  · 只显示当前数据集支持的方式
  · 选中后在右侧标签面板点选标签（可选，默认标签可用）

第三步  绘制与保存
  · 按各方式的常规操作绘制（详见"标注方式与操作"）
  · 默认自动保存：绘制完成约 0.3 秒后写入标注文件
  · 底部状态栏实时显示 加载/统计/保存 进度

小提示
  · 第一次打开大数据集会建立索引，之后再打开会快很多
  · 图片切换用 A/D 或 ↑/↓
  · 误画了立即 Ctrl+Z 撤销
"""),
            ("标注方式与操作", """标注方式与操作（主流常规操作）

一、通用操作
  选择方式后保持连续标注，画完立即可画下一个
  Esc           第一次取消当前半成品，第二次退出绘制
  右键(空白处)  退出绘制；点击已有标注弹出编辑对话框
  Ctrl+Z / Ctrl+Y   撤销 / 重做（绘制、删除、移动、改标签均可撤销）
  Delete / Backspace 删除选中标注
  双击标注      打开编辑对话框（改标签、关键点坐标与可见性）

二、矩形
  按住左键拖拽画出（任意方向）
  编辑：框内拖动移动，四角手柄缩放
  按住 Shift 拖拽 = 临时画正方形

三、正方形
  拖拽时自动锁定宽高相等，其余同矩形

四、多边形
  逐点左键单击；双击或 Enter 闭合；右键闭合
  Backspace 撤销上一个点；Esc 取消当前形状
  编辑：选中后拖动顶点手柄

五、旋转框 (YOLO OBB)
  像矩形一样拖拽画出（初始水平）
  选中后出现绿色旋转手柄，拖动绕中心旋转
  按住 Shift 旋转按 15° 吸附
  编辑：框内拖动移动

六、关键点位 (YOLO Pose / COCO)
  逐点左键单击，点满自动完成；双击或 Enter 提前完成
  画布上"点位数"框可修改数量（17=COCO 人体官方点位，
  其他数量自动命名 kpt_1..kpt_N，修改会写入 data.yaml）
  右键点位循环切换可见性：可见(2) → 未标注(0) → 遮挡(1)
  编辑：拖动点位；编辑对话框可改坐标和可见性

七、辅助
  W          快速启用/退出上次使用的标注方式
  鼠标滚轮   缩放画布（以光标位置为锚点）
  1-9        选择绑定该键的标签；Ctrl+1-9 把当前选中标签绑定到该键
  缩放保持   换图时保留缩放级别；Ctrl+0 适应画布并重置
  Ctrl+C+A   标注辅助线设置

  启动时自动恢复上次的数据集与最后浏览的图片，
  可在 参数设置 → 通用设置 中改为空画布启动。
"""),
            ("数据集格式说明", FormatGuideDialog._chinese_content()),
            ("新建数据集向导", """新建数据集向导 (Ctrl+N)

像编程管理项目一样管理数据集：数据集统一放在工作空间里，
图片来源只是素材，真正的数据集创建在工作空间的子目录中。

操作步骤
  1. 文件菜单 → 新建 (Ctrl+N)
  2. 选择工作空间（记忆上次选择，多个数据集可共处一个工作空间）
  3. 输入数据集名称（同工作空间内不能重名）
  4. 可选：选择数据来源（图片目录、数据集目录）
       · 普通图片文件夹 → 图片复制进新数据集的 images/
       · 已是标准数据集 → 图片和标注文件一并导入
       · 不填 → 创建空数据集，稍后自行放入图片
  5. 选择标注格式：
       YOLO 检测 / 分割 / 关键点 / 旋转框，Pascal VOC，COCO
  6. 可选：标签类别（逗号分隔）；关键点格式可设点位数
  7. 点击"创建"

会发生什么
  · 创建 工作空间/名称/ 数据集目录，含标准结构
    （images/ + labels/ + data.yaml，或 Annotations/、annotations/）
  · 图片复制进来（嵌套子目录会拍平，重名自动加前缀）
  · 源文件夹内容不受任何影响
  · 同名数据集会被拒绝，不会覆盖
  · 创建完成自动打开，工作空间位置已记住
"""),
            ("更多功能", """更多功能

标签分组 (Ctrl+L+G)
  维护可复用的标签模板库，跨数据集保留

点位类型 (Ctrl+K+G)
  预定义关键点的点数与名称；画制关键点前在画布左上角选择类型即可采用其点位名称

文件筛选 (Ctrl+I+F)
  按文件名、标注状态（已标/未标）、标签过滤图片列表

数据统计 (Ctrl+D+S)
  查看总标注数、各类别分布、标注进度

数据转换 (Ctrl+D+C)
  YOLO / VOC / COCO 互相批量转换

自动标注 (Ctrl+A+L)
  需在参数设置中选择 YOLO 模型；
  支持官方 YOLO 检测/Pose 模型，可中途停止

参数设置 (Ctrl+A+S)
  标注方式、线宽字号线宽、保存模式、语言等

快捷键完整列表见 帮助 → 快捷按键
"""),
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

Statistics (Ctrl+D+S)
  Label counts, per-class distribution, progress

Conversion (Ctrl+D+C)
  Batch convert between YOLO / VOC / COCO

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
