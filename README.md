# ModelLabeling

Windows-first Python/PySide6 image annotation tool with YOLO/VOC support.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open an image directory from the toolbar. The application creates a `labels` directory beside the images and stores the project settings in `model_labeling.json`.

The current build supports standard rectangle, square, polygon, and keypoint annotation modes. Selected annotations can be moved and resized on the canvas; polygon vertices and pose keypoints can be dragged, and pose annotations expose coordinate and visibility editing in the annotation dialog. Dataset tasks are explicit: COCO, YOLO Detection, YOLO Segmentation, YOLO Pose, and Pascal VOC. Unsupported geometry is rejected rather than written into a private extension format. Ultralytics YOLO Pose ONNX models are detected from model metadata and their official keypoint rows are decoded. COCO segmentation/keypoint data and official YOLO task formats are supported when `onnxruntime` is installed for model inference.
