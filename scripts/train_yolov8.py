import os
import json
import glob
from PIL import Image
from ultralytics import YOLO

def convert_json_to_yolo(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        print(f"Warning: Ensure {images_dir} and {labels_dir} exist.")
        return
        
    print("Checking for required JSON to YOLO label conversion...")
    converted_count = 0
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(labels_dir, filename)
        txt_path = os.path.join(labels_dir, filename.replace('.json', '.txt'))
        
        if os.path.exists(txt_path):
            continue
            
        base_name = filename.replace('.json', '')
        image_files = glob.glob(os.path.join(images_dir, f"{base_name}.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        image_path = image_files[0]
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            continue
            
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
                
        yolo_lines = []
        for obj in data:
            class_id = 0
            x_min = obj['x']
            y_min = obj['y']
            width = obj['width']
            height = obj['height']
            
            x_center = x_min + (width / 2.0)
            y_center = y_min + (height / 2.0)
            
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
            
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        converted_count += 1
        
    if converted_count > 0:
        print(f"Successfully converted {converted_count} JSON labels to YOLO .txt format.")
    else:
        print("No new JSON labels needed conversion.")

def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    dataset_dir = os.path.join(project_root, 'Dataset')
    data_yaml_path = os.path.join(project_root, 'data.yaml')
    
    # Dynamically update data.yaml to avoid path resolution errors
    yaml_lines = [
        f"path: {dataset_dir.replace(chr(92), '/')}",
        "train: images",
        "val: images",
        "",
        "nc: 1",
        "names:",
        "  0: 'license_plate'"
    ]
    with open(data_yaml_path, 'w') as f:
        f.write('\n'.join(yaml_lines))
    runs_dir = os.path.join(project_root, 'runs')

    convert_json_to_yolo(dataset_dir)

    model = YOLO('yolov8n.pt')
    
    print(f"Starting Highly Optimized YOLOv8 CPU training using: {data_yaml_path}")
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        patience=5,
        imgsz=320,
        rect=True,
        batch=16,
        cache=False,
        freeze=10,
        workers=4,
        project=runs_dir,
        name='license_plate_detector',
        optimizer='AdamW',
        lr0=0.001,
        mosaic=0.0,
        mixup=0.0,
        single_cls=True,
        device='cpu',
        verbose=False
    )
    print("Training complete. Model saved in runs/license_plate_detector/weights/best.pt")
    
    print("Calculating Mean Average Precision and Latency metrics...")
    metrics = model.val()
    
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    latency = metrics.speed['inference']
    
    print("-" * 40)
    print("=== Performance Metrics ===")
    print(f"Mean Average Precision (mAP@50): {map50:.4f}")
    print(f"Mean Average Precision (mAP@50-95): {map50_95:.4f}")
    print(f"Mean Inference Latency: {latency:.2f} ms / image")
    print("-" * 40)

if __name__ == '__main__':
    train()
