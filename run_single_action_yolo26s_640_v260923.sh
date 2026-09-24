#!/usr/bin/env bash
# YOLO26s 640: YOLO dataset training -> ONNX(nms=True) -> RK3588/RK3576 FP16 RKNN
set -Eeuo pipefail
YOLO_ROOT="/home/ruilin/PycharmProjects/YOLO"
DATASET_PARENT="$YOLO_ROOT/dataset"
ARCHIVE="$DATASET_PARENT/yolo_single_action_v260923.zip"
DATASET_NAME="yolo_single_action_v260923"
DATASET_DIR="$DATASET_PARENT/$DATASET_NAME"
MODELS_DIR="$YOLO_ROOT/models"
RKNN_DIR="$YOLO_ROOT/rknn"
RUNS_DIR="$YOLO_ROOT/runs/detect"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="single_action_yolo26s_640_$STAMP"
LOG="$YOLO_ROOT/${RUN_NAME}_pipeline.log"
IMAGE_SIZE=640
DEVICE="1"
exec > >(tee -a "$LOG") 2>&1
source /home/ruilin/miniconda3/etc/profile.d/conda.sh
mkdir -p "$DATASET_PARENT" "$MODELS_DIR" "$RKNN_DIR/rk3588" "$RKNN_DIR/rk3576" "$RUNS_DIR"
echo "[0/5] pipeline=$RUN_NAME"
echo "archive=$ARCHIVE"
echo "dataset=$DATASET_DIR"
if [[ ! -f "$ARCHIVE" ]]; then echo "ERROR: archive not found: $ARCHIVE" >&2; exit 1; fi
if [[ -f "$DATASET_DIR/data.yaml" ]]; then
  echo "Dataset already extracted; reusing: $DATASET_DIR"
else
  echo "Extracting dataset..."
  unzip -q "$ARCHIVE" -d "$DATASET_PARENT"
fi
if [[ ! -f "$DATASET_DIR/data.yaml" ]]; then echo "ERROR: data.yaml not found after extraction: $DATASET_DIR/data.yaml" >&2; exit 1; fi
sed -i "s#^path:.*#path: $DATASET_DIR#" "$DATASET_DIR/data.yaml"
TRAIN_PARAMS="$(conda run --no-capture-output -n yolo python "$YOLO_ROOT/compute_train_params.py" --data "$DATASET_DIR/data.yaml" --imgsz "$IMAGE_SIZE")"
read -r EPOCHS PATIENCE AUTO_BATCH <<< "$TRAIN_PARAMS"
# GPU 1 is shared with another workload; cap runtime batch to avoid OOM.
BATCH=4
echo "Auto params: epochs=$EPOCHS patience=$PATIENCE auto_batch=$AUTO_BATCH runtime_batch=$BATCH imgsz=$IMAGE_SIZE device=$DEVICE"
if [[ ! -f "$YOLO_ROOT/yolo26s.pt" ]]; then echo "ERROR: base model missing: $YOLO_ROOT/yolo26s.pt" >&2; exit 1; fi
echo "[1/5] Training YOLO26s..."
conda run --no-capture-output -n yolo yolo detect train \
  model="$YOLO_ROOT/yolo26s.pt" data="$DATASET_DIR/data.yaml" \
  imgsz="$IMAGE_SIZE" epochs="$EPOCHS" batch="$BATCH" patience="$PATIENCE" \
  workers=16 device="$DEVICE" cache=False cos_lr=True close_mosaic=10 \
  mosaic=0.3 fliplr=0.0 project="$RUNS_DIR" name="$RUN_NAME" exist_ok=True
BEST_MODEL="$RUNS_DIR/$RUN_NAME/weights/best.pt"
if [[ ! -f "$BEST_MODEL" ]]; then echo "ERROR: best.pt missing: $BEST_MODEL" >&2; exit 1; fi
cp -f "$BEST_MODEL" "$MODELS_DIR/${RUN_NAME}_best.pt"
echo "[2/5] Exporting ONNX (nms=True)..."
conda run --no-capture-output -n yolo yolo export \
  model="$BEST_MODEL" format=onnx imgsz="$IMAGE_SIZE" \
  opset=12 simplify=True dynamic=False nms=True
ONNX_SOURCE="${BEST_MODEL%.pt}.onnx"
ONNX_OUTPUT="$MODELS_DIR/${RUN_NAME}_best_nms.onnx"
if [[ ! -f "$ONNX_SOURCE" ]]; then echo "ERROR: ONNX source missing: $ONNX_SOURCE" >&2; exit 1; fi
cp -f "$ONNX_SOURCE" "$ONNX_OUTPUT"
convert_rknn() {
  local target="$1"
  local output="$RKNN_DIR/$target/${RUN_NAME}_${target}_fp16.rknn"
  echo "Converting $target -> $output"
  conda run --no-capture-output -n rknn2.3.2 python "$YOLO_ROOT/convert_onnx_to_rknn_fp16.py" \
    --input "$ONNX_OUTPUT" --output "$output" --target "$target"
  [[ -f "$output" ]] || { echo "ERROR: RKNN output missing: $output" >&2; exit 1; }
}
echo "[3/5] Exporting RK3588 FP16..."
convert_rknn rk3588
echo "[4/5] Exporting RK3576 FP16..."
convert_rknn rk3576
echo "[5/5] Completed"
ls -lh "$MODELS_DIR/${RUN_NAME}_best.pt" "$ONNX_OUTPUT" \
  "$RKNN_DIR/rk3588/${RUN_NAME}_rk3588_fp16.rknn" \
  "$RKNN_DIR/rk3576/${RUN_NAME}_rk3576_fp16.rknn"
echo "RUN_NAME=$RUN_NAME"
echo "ONNX_OUTPUT=$ONNX_OUTPUT"
echo "RK3588_OUTPUT=$RKNN_DIR/rk3588/${RUN_NAME}_rk3588_fp16.rknn"
echo "RK3576_OUTPUT=$RKNN_DIR/rk3576/${RUN_NAME}_rk3576_fp16.rknn"
