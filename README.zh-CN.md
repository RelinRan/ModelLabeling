# ModelLabeling

**[English](README.md)** | 简体中文

Windows 优先的 Python/PySide6 桌面图像标注工作台。面向 YOLO / Pascal VOC / COCO 数据集的快速标注、管理与转换，覆盖 Ultralytics 官方全部四种标注文件任务（检测、分割、关键点、旋转框），内置大数据集索引、新建数据集向导与 ONNX 自动标注。

- 版本：v1.0.0
- 作者：RelinRan · [GitHub](https://github.com/RelinRan) · relinran@foxmail.com

---

## 目录

1. [功能总览](#功能总览)
2. [支持的数据集格式](#支持的数据集格式)
3. [标注方式与操作](#标注方式与操作)
4. [快速开始](#快速开始)
5. [使用指南](#使用指南)
6. [快捷键](#快捷键)
7. [架构设计](#架构设计)
8. [性能与大数据集](#性能与大数据集)
9. [测试](#测试)
10. [常见问题](#常见问题)

---

## 功能总览

| 类别 | 能力 |
| --- | --- |
| 数据集任务 | YOLO 检测 / YOLO 分割 / YOLO 关键点(Pose) / YOLO 旋转框(OBB) / Pascal VOC / COCO |
| 标注方式 | 矩形、正方形、多边形、旋转框、关键点位（按当前数据集能力自动过滤） |
| 点位类型 | 预定义点数、点位名称与标注标签 (Ctrl+K+G)；内置「姿态」(17 点 COCO 人体，标签 pose) 与「汽车」(标签 car)；画布上一键套用整套 schema |
| 数据集初始化 | 新建向导：工作空间 + 名称 + 参数，图片素材复制进来（可导入已有数据集标注） |
| 兼容性 | 完全无标注的数据集（空 labels/、无 Annotations/、纯图片文件夹）可直接打开 |
| 自动标注 | ONNX 推理，支持官方 YOLO 检测/Pose 模型，后台可停止 |
| 数据转换 | YOLO / VOC / COCO 批量互转，保留目录层级与 data.yaml 类名 |
| 编辑体验 | 连续绘制、撤销/重做、拖动/缩放/旋转、关键点可见性（COCO 0/1/2 规范） |
| 大数据集 | SQLite 路径索引 + keyset 分页，万级图片列表流畅 |
| 界面 | IDEA 风格深色 UI，中英双语，所有菜单/对话框/状态栏随语言即时刷新 |

---

## 支持的数据集格式

### YOLO（Ultralytics 官方四种任务）

推荐目录结构：

```
dataset/
├─ images/                图片（支持 train/ 子目录）
├─ labels/                与图片同名的 .txt（可为空）
├─ classes.txt            每行一个类别名，行号即 class_id
└─ data.yaml              可选：task、names、kpt_shape 等
```

每行一个目标，按任务不同：

| 任务 | 行格式 | 示例 |
| --- | --- | --- |
| 检测 | `class cx cy w h` | `0 0.5125 0.4800 0.2500 0.3600` |
| 分割 | `class x1 y1 x2 y2 …`（多边形顶点） | `0 0.1 0.1 0.5 0.1 0.3 0.4` |
| 关键点 | `class cx cy w h px py v …` | `0 0.5 0.5 0.4 0.4 0.4 0.4 2 …` |
| 旋转框 | `class x1 y1 x2 y2 x3 y3 x4 y4`（4 角点） | `0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5` |

- 坐标均为 0–1 归一化值；关键点可见性 `v`：0 未标注、1 遮挡、2 可见
- 点位数量由 `data.yaml` 的 `kpt_shape: [N, 3]` 声明（画布上可改，自动写回）
- 任务识别：优先 `data.yaml` 的 `task` 字段，否则按行格式推断

### Pascal VOC

```
dataset/
├─ JPEGImages/            图片
└─ Annotations/           与图片同名的 .xml
```

XML 含 `filename`、`size`、`object/bndbox`（像素坐标）。`Annotations/` 可为空，保存时自动创建。

### COCO

```
dataset/
├─ images/
└─ annotations/
   └─ annotations.json    （或 instances.json）
```

JSON 含 `images`（id/file_name/width/height）、`annotations`（image_id/category_id/bbox，bbox 为 `[x, y, w, h]` 像素）、`categories`。支持 segmentation 与 keypoints。空 annotations 列表的合法文档同样可打开。

---

## 标注方式与操作

选择方式后**保持连续标注**（画完立即可画下一个），符合 CVAT/LabelMe 主流惯例。

| 方式 | 绘制 | 编辑 |
| --- | --- | --- |
| 矩形 | 按住左键拖拽（任意方向）；按住 **Shift** 临时画正方形 | 框内拖动移动，四角手柄缩放 |
| 正方形 | 拖拽时自动锁定等宽高 | 同矩形 |
| 多边形 | 逐点左键单击；**双击 / Enter / 右键** 闭合；Backspace 撤销上一点 | 拖动顶点手柄 |
| 旋转框 (OBB) | 像矩形一样拖拽画出 | 选中后拖动绿色手柄绕中心旋转，**Shift 吸附 15°**；框内拖动移动 |
| 关键点位 (Pose) | 逐点单击，点满类型点数自动完成；**Tab** 跳过当前点（记为未标注）；双击/Enter 提前完成 | 拖动点位标记（框自动跟随）；**右键点位循环可见性 2→0→1**；框体不可拖动——外接框由点位推导 |

画布左上角的下拉框只列出当前数据集支持的方式（如 YOLO 检测只有矩形/正方形）。关键点模式下同一行依次是**点位类型选择器**和实时**进度胶囊**（`点位名 (i/N)`）；点位数量由所选类型决定，画布上不再单独改数。

通用操作：

- **Ctrl+Z / Ctrl+Y** 撤销/重做（绘制、删除、移动、改标签、编辑对话框均可撤销）
- **Esc** 第一次取消当前半成品，第二次退出绘制；**右键空白处**退出绘制
- **双击标注 / 右键标注** 打开编辑对话框
- 自动保存：绘制完成约 0.3 秒后写入标注文件；保存时严格校验格式合法性，拒绝任何非标准数据
- 误点（微小拖拽）不退出绘制状态

---

## 快速开始

### 环境要求

- Python 3.9+（64 位）
- Windows 10/11（Linux/macOS 理论可运行，未系统测试）

### 安装与运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

可选依赖：`onnxruntime` 用于自动标注推理（未安装时其余功能不受影响）。

### 三步开始标注

1. **打开** (Ctrl+O) 选择数据集根目录——格式、目录、任务自动识别；或对裸图片文件夹用**新建** (Ctrl+N)
2. 画布左上角选择**标注方式**
3. 绘制——自动保存开启，底部状态栏实时显示加载/统计/保存进度

---

## 使用指南

应用内 **帮助 → 使用说明** 含完整手册（快速上手 / 标注操作 / 数据集格式 / 新建向导 / 更多功能），以下为要点。

### 新建数据集向导 (Ctrl+N)

像编程管理项目一样管理数据集：数据集统一放在**工作空间**里，图片来源只是素材，真正的数据集创建在 `工作空间/名称/` 中：

1. 选择**工作空间**（记忆上次选择，可容纳多个数据集）
2. 输入**数据集名称**（同工作空间内唯一，非法字符实时提示）
3. 可选选**数据来源**（图片目录或数据集目录）：普通文件夹 → 图片复制进 `images/`（嵌套目录拍平）；已是标准数据集 → 图片和**标注一并导入**；不填 → 空数据集
4. 选择目标格式（六种任务）+ 可选标签类别、关键点数
5. 点“创建”——完成后自动打开，源文件夹不受任何影响

### 图片切换与浏览

A/D 或 ↑/↓ 切换（首尾安全）；**滚轮缩放**（以光标为锚点）、Ctrl+± 缩放、Ctrl+0 适应画布；缩放级别跨图保持（标注局部细节时换图不重置）；数字键 **1-9** 快速切换标签；列表按文件名/状态/标签筛选 (Ctrl+I+F)。

### 启动恢复

默认**启动时自动恢复上次打开的数据集**（并回到最后浏览的图片）；可在 参数设置 → 通用设置 → 启动 改为空画布启动。

### 标签管理

标签分组 (Ctrl+L+G) 维护跨数据集的模板库（应用级 SQLite 持久化）；画新标签首次保存时自动注册进 classes.txt / COCO categories。

### 点位类型 (Ctrl+K+G)

把关键点 schema 一次定义好，标注时直接选用：

- 每个类型包含**名称**、**点数**、逐点**点位名**（可增删改，按标注顺序编号）和一个**标注标签**
- 内置类型：姿态（COCO 官方 17 人体点，标签 `pose`，受保护）与 汽车（轮毂/车窗/车灯，标签 `car`）
- 画布左上角选择类型即刻生效；完成的标注自动采用该类型的标签与颜色
- 类型的标签会自动注册进「默认标签」分组；被类型引用期间，标签管理中禁止删除该标签
- 关键点标注的编辑窗口中标签锁定为类型的标签（下拉箭头隐藏），点位坐标与可见性仍可编辑

### 数据统计与转换

- 统计 (Ctrl+D+S)：总标注数、类别分布、标注进度（后台线程计算，完成后可查）
- 转换 (Ctrl+D+C)：YOLO/VOC/COCO 批量互转

### 自动标注 (Ctrl+A+L)

在参数设置中选择 YOLO 模型后可用；支持官方 YOLO 检测与 Pose 模型（从模型元数据解码类别与关键点）；后台运行、可停止、进度见状态栏。

## 快捷键

| 按键 | 功能 | 按键 | 功能 |
| --- | --- | --- | --- |
| Ctrl+N | 新建数据集 | Ctrl+O | 打开 |
| Ctrl+S | 保存 | Ctrl+Q | 退出 |
| A / ↑ | 上一张 | D / ↓ | 下一张 |
| W | 启用/退出绘制 | Esc | 取消当前形状/退出绘制 |
| Ctrl+Z / Y | 撤销 / 重做 | Delete | 删除选中标注 |
| Shift+拖拽 | 画正方形 | Enter / 双击 | 完成多边形/关键点 |
| Backspace | 绘制中撤销上一点 | 右键 | 退出绘制 / 编辑标注 |
| Ctrl+0 | 适应画布 | 滚轮 / Ctrl+± | 缩放（光标为锚点） |
| 1-9 | 快速选标签 | 缩放跨图保持 | 换图不重置缩放，Ctrl+0 重置 |
| Ctrl+L+G | 标签分组 | Ctrl+K+G | 点位类型 |
| Ctrl+A+S | 参数设置 | Ctrl+I+F | 文件筛选 |
| Ctrl+C+A | 标注辅助 | Ctrl+D+S | 数据统计 |
| Ctrl+D+C | 数据转换 | Ctrl+A+L | 自动标注 |
| Ctrl+H | 历史 | Tab | 跳过当前关键点 |

---

## 架构设计

```
app.py                     入口
src/
├─ models/                 数据模型
│  ├─ annotation.py        Annotation / ShapeType / Keypoint / LabelPreset
│  └─ project.py           ProjectSettings / ProjectState / ImageRecord
├─ services/               服务层（无 UI 依赖）
│  ├─ dataset_detector.py  目录结构与任务识别（含纯图片回退、无标注支持）
│  ├─ dataset_initializer.py  新建向导的原地初始化
│  ├─ dataset_index.py     数据集级 SQLite 路径索引（分页/筛选/定位）
│  ├─ dataset_session.py   当前数据集的权威上下文
│  ├─ annotation_service.py格式适配与加载/保存（YOLO×4 / VOC / COCO）
│  ├─ coco_store.py        COCO 的 SQLite 工作副本（单图事务编辑）
│  ├─ label_group_store.py 应用级标签模板库（SQLite）
│  ├─ keypoint_group_store.py 应用级点位类型库（SQLite）
│  ├─ format_capabilities.py 任务能力矩阵与保存前校验
│  ├─ format_adapters.py   格式分发层
│  ├─ yolo_metadata.py     data.yaml 读写（kpt_shape 等）
│  ├─ conversion_service.py数据集互转
│  ├─ onnx_service.py      ONNX 推理
│  ├─ workers.py           全部后台 Worker（扫描/计数/统计/单图/自动标注/保存）
│  └─ operation_coordinator.py 后台操作互斥
└─ widgets/                界面层
   ├─ main_window.py       主窗口与全部流程编排
   ├─ canvas_view.py       画布（五种标注方式、旋转手柄、撤销栈）
   └─ …                    各对话框与面板
tests/                     单元测试（68 个）
```

### 关键设计

- **四级 SQLite，各管一层生命周期**：数据集级索引（`%LOCALAPPDATA%\ModelLabeling\index\`，按根路径哈希命名）、COCO 目录级工作副本（`.model_labeling.sqlite3`）、应用级标签库（`label_groups.sqlite3`）、应用级点位类型库（`keypoint_groups.sqlite3`），互不耦合
- **打开流程**：探测格式 → DatasetSession → 后台扫描建索引（增量，靠 size+mtime 跳过未变文件）→ **首批结果即可开始标注**（非阻塞加载）→ 统计后台计算
- **按需加载**：列表只读索引元数据；点选图片后单图标注后台加载；COCO 单图编辑走单事务 upsert
- **严格标准输出**：保存前 `format_capabilities.validate_annotations` 校验（形状合法性、关键点数量/schema 一致性、OBB 四角点等），违规拒绝而非降级写私有格式
- **线程模型**：扫描/计数/统计/单图加载/保存各自独立 Worker；统计跑在 Python 线程上以规避 PySide6 QThread 在繁忙 GUI 下的堆损坏问题；`OperationCoordinator` 保证同一时刻只有一个重操作

---

## 性能与大数据集

- 数千张图片首次打开秒级（建索引），再次打开直接走缓存
- 列表使用 keyset 分页（避免 OFFSET 退化），滚动加载
- 万级图片目录不阻塞 UI：加载期间即可开始标注

---

## 测试

```powershell
python -m pytest tests -q
```

68 个单元测试覆盖：格式往返（六种任务）、数据集识别（含无标注/纯图片/OBB 推断）、能力校验、画布交互（五种方式绘制/旋转/撤销/键盘）、初始化向导、设置对话框、ONNX 结果映射、操作互斥等。

另有端到端脚本 `test_annotation_flow.py`（三格式打开 × 五方式标注 × 落盘校验 + 无标注/OBB 场景），可直接运行。

---

## 常见问题

**打开文件夹提示"unsupported dataset format"？**
确认文件夹内有图片；或用新建向导 (Ctrl+N) 初始化为标准结构。

**YOLO 数据集没有 classes.txt 会怎样？**
可以打开和标注；首次保存时自动创建，并用当前标签补全类别。

**修改关键点数量后旧标注怎么办？**
点位数量由所选点位类型决定 (Ctrl+K+G)；编辑类型点数时保留已改过的自定义点名，超出截尾、不足补 kpt_N。保存校验会拒绝混合点位数量的数据集——删除或重画旧标注后再换类型。YOLO Pose 的点数会写入 `data.yaml` 的 `kpt_shape`。

**保存关键点时提示"关键点命名与类别冲突"？**
COCO 每个类别只允许一套关键点命名。报错会列出两套命名；为该标签统一命名，或让类型改用其他标签。

**四边形分割数据集被识别成 OBB？**
9 列行存在天然歧义（4 顶点多边形 = OBB 角点）。在 参数设置 → 任务格式 里手动切换，或给 `data.yaml` 写 `task: segment`。

**COCO 大 JSON 编辑慢？**
本工具把 COCO 规范化为 SQLite 工作副本（`annotations/.model_labeling.sqlite3`），单图编辑为单事务；退出或切换数据集时自动导出回 JSON。
