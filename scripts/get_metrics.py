from ultralytics import YOLO

print("Loading model...")
model = YOLO('yolov8n.onnx')  # Use optimized ONNX if available, else YOLO handles it
print("Evaluating on data.yaml...")
try:
    metrics = model.val(data='data.yaml', split='val')
    print("--- Metrics ---")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Inference Latency: {metrics.speed['inference']:.2f} ms")
except Exception as e:
    print(f"Error evaluating: {e}")
