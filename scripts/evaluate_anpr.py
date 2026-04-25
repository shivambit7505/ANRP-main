import os
import sys
import json
import cv2

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anpr_engine import ANPREngine

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def calculate_cer(predicted, ground_truth):
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    distance = levenshtein_distance(predicted, ground_truth)
    return distance / len(ground_truth)

def evaluate(dataset_json):
    """
    Evaluate ANPR performance on a dataset.
    Dataset JSON format: {"image_path": "ground_truth_text"}
    """
    if not os.path.exists(dataset_json):
        print(f"Dataset JSON not found: {dataset_json}")
        return

    with open(dataset_json, 'r') as f:
        dataset = json.load(f)

    print("Initializing ANPR Engine...")
    engine = ANPREngine()
    
    total_cer = 0.0
    exact_matches = 0
    total_samples = len(dataset)
    
    if total_samples == 0:
        print("Empty dataset.")
        return

    print(f"Evaluating {total_samples} samples...")

    for img_path, gt_text in dataset.items():
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_path}")
            total_samples -= 1
            continue

        predicted_text, conf = engine.extract_text(image)
        
        cer = calculate_cer(predicted_text, gt_text)
        total_cer += cer
        
        is_match = predicted_text == gt_text
        if is_match:
            exact_matches += 1
            
        print(f"Img: {os.path.basename(img_path)} | GT: {gt_text} | Pred: {predicted_text} | CER: {cer:.2f}")

    avg_cer = total_cer / total_samples if total_samples > 0 else 0
    word_accuracy = (exact_matches / total_samples) * 100 if total_samples > 0 else 0

    print("-" * 40)
    print("=== Evaluation Results ===")
    print(f"Total Samples Valid: {total_samples}")
    print(f"Character Error Rate (CER): {avg_cer:.4f}")
    print(f"Word Accuracy (Exact Match): {word_accuracy:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ANPR Engine")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSON dataset (image_path: ground_truth)")
    args = parser.parse_args()
    
    evaluate(args.dataset)
