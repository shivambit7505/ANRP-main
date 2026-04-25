import os
from ultralytics import YOLO

def export_model_to_onnx(model_path):
    if not os.path.exists(model_path):
        print(f"Warning: Model {model_path} not found. Skipping ONNX export.")
        return None
        
    print(f"Exporting {model_path} to ONNX format...")
    model = YOLO(model_path)
    
    # Exporting to ONNX format. simplify=True optimizes the graph architecture for ONNX runtime
    onnx_path = model.export(format='onnx', opset=12, simplify=True)
    print(f"Successfully exported to {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    base_model_path = os.path.join(project_root, 'yolov8n.pt')
    plate_model_path = os.path.join(project_root, 'runs', 'license_plate_detector', 'weights', 'best.pt')
    
    print("--- Starting Runtime Optimization (ONNX Export) ---")
    export_model_to_onnx(base_model_path)
    export_model_to_onnx(plate_model_path)
    
    print("ONNX export complete. The pipeline will now automatically detect and use these optimized models.")
