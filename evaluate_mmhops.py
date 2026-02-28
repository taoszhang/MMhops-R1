#!/usr/bin/env python3
"""
MMhops Evaluation Script

This script evaluates predictions on the MMhops benchmark dataset.
The dataset is available at: https://huggingface.co/datasets/taoszhang/MMhops

Usage:
    python evaluate_mmhops.py --prediction-file predictions.jsonl

Prediction File Format (JSONL):
    Each line should be a JSON object with the following fields:
    - id (required): Sample ID matching the dataset (e.g., "MMhops_test_00001")
    - prediction (required): Model's predicted answer as a string

Example prediction file:
    {"id": "MMhops_test_00001", "prediction": "174th"}
    {"id": "MMhops_test_00002", "prediction": "12"}
    ...

Evaluation Metrics:
    - String: Exact match after normalization (lowercase, remove articles/punctuation)
    - Numerical: Match if prediction falls within the answer range (±10% tolerance)
    - Time: Exact match after normalization

Output:
    - Overall accuracy
    - Accuracy breakdown by split (Bridge/Compare)
    - Accuracy breakdown by question type (String/Numerical/Time)
"""

import argparse
import json
import re
import string
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple, Union

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not found. Install with: pip install datasets")
    exit(1)

try:
    from word2number import w2n
except ImportError:
    print("Error: 'word2number' package not found. Install with: pip install word2number")
    exit(1)


# ==============================================================================
# Text Normalization Functions
# ==============================================================================

def normalize_answer(text: str) -> str:
    """Normalize text by removing articles, punctuation, whitespace, and lowercasing."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))


# ==============================================================================
# Numerical Processing Functions
# ==============================================================================

def replace_number_words(text: str) -> str:
    """Convert English number words to digits (e.g., 'twelve' -> '12')."""
    pattern = re.compile(
        r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
        r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
        r'eighty|ninety|hundred|thousand|million|billion|trillion|'
        r'point|and|[-\s])+\b', re.IGNORECASE
    )

    def convert(match):
        try:
            return str(w2n.word_to_num(match.group()))
        except ValueError:
            return match.group()

    return pattern.sub(convert, text)


def find_all(s: str, c: str) -> Generator[int, None, None]:
    """Find all occurrences of a character in a string."""
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)


def clean_str_range(text: str) -> str:
    """Clean range expressions (e.g., '9-10' -> '9 - 10')."""
    idx_list = list(find_all(text, '-'))
    idx_replace = [idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()]
    new_str = ''.join(' - ' if idx in idx_replace else s for idx, s in enumerate(text))
    return new_str


def process_numerical_answer(string_number: str) -> Union[float, List[float]]:
    """Parse numerical answer string into a number or range [min, max]."""
    try:
        string_number = replace_number_words(str(string_number))
    except Exception:
        string_number = str(string_number)
    
    string_number = clean_str_range(string_number)
    numerical_numbers_tmp = re.findall(
        r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string_number
    )
    numerical_numbers_tmp = [n.replace(',', '').strip('.') for n in numerical_numbers_tmp]
    
    numerical_numbers = []
    for n in numerical_numbers_tmp:
        if n.count('.') > 1:
            n = n.split('.')[0]
        numerical_numbers.append(float(n))

    if len(numerical_numbers) > 2:
        numerical_numbers = numerical_numbers[:2]

    if len(numerical_numbers) == 2:
        first_val, second_val = numerical_numbers
        return [first_val, second_val] if first_val <= second_val else first_val
    elif len(numerical_numbers) == 1:
        return numerical_numbers[0]
    else:
        return [0, 0]


def safe_division(x: float, y: float) -> float:
    """Safe division that returns 0 when dividing by 0."""
    return x / y if y != 0 else 0


def range_intersection_over_union(x_list: List[float], y_list: List[float]) -> float:
    """Calculate IoU of two numerical ranges."""
    min_1, max_1 = min(x_list), max(x_list)
    min_2, max_2 = min(y_list), max(y_list)

    overlap = max(0.0, min(max_1, max_2) - max(min_1, min_2))
    length_x = (max_1 - min_1) + 1e-12
    length_y = (max_2 - min_2) + 1e-12
    iou = safe_division(overlap, length_x + length_y - overlap)
    return iou


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction exactly matches normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_string(prediction: str, ground_truths: List[str]) -> int:
    """Evaluate string-type answer (exact match after normalization)."""
    return max(int(exact_match_score(prediction, gt)) for gt in ground_truths)


def evaluate_time(prediction: str, ground_truths: List[str]) -> int:
    """Evaluate time-type answer (exact match after normalization)."""
    return max(int(exact_match_score(prediction, gt)) for gt in ground_truths)


def evaluate_numerical(prediction: str, ground_truths: List[str]) -> int:
    """
    Evaluate numerical-type answer.
    
    Ground truths define a valid range. Prediction is correct if:
    1. It falls within the range, OR
    2. If prediction is also a range, IoU >= 0.5
    """
    pred_value = process_numerical_answer(prediction)
    
    try:
        gt_values = [float(gt) for gt in ground_truths]
        min_value = min(gt_values)
        max_value = max(gt_values)
    except (ValueError, TypeError):
        return 0

    if isinstance(pred_value, list):
        if min_value <= pred_value[0] <= max_value and min_value <= pred_value[1] <= max_value:
            return 1
        iou = range_intersection_over_union(pred_value, [min_value, max_value])
        return 1 if iou >= 0.5 - 1e-12 else 0
    else:
        return 1 if min_value <= pred_value <= max_value else 0


def evaluate_sample(prediction: str, answer_eval: List[str], problem_type: str) -> int:
    """Evaluate a single sample based on its problem type."""
    problem_type = problem_type.lower()
    
    if problem_type == "string":
        return evaluate_string(prediction, answer_eval)
    elif problem_type == "time":
        return evaluate_time(prediction, answer_eval)
    elif problem_type == "numerical":
        return evaluate_numerical(prediction, answer_eval)
    else:
        print(f"Warning: Unknown problem type '{problem_type}', treating as string")
        return evaluate_string(prediction, answer_eval)


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_predictions(filepath: str) -> Dict[str, str]:
    """
    Load predictions from a JSONL file.
    
    Expected format per line:
        {"id": "MMhops_test_00001", "prediction": "answer"}
    
    Returns:
        Dict mapping sample ID to prediction string
    """
    predictions = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            
            if 'id' not in item:
                print(f"Warning: Line {line_num} missing 'id' field, skipping")
                continue
            if 'prediction' not in item:
                print(f"Warning: Line {line_num} missing 'prediction' field, skipping")
                continue
            
            predictions[item['id']] = str(item['prediction'])
    
    return predictions


def load_dataset_from_hf(dataset_name: str = "taoszhang/MMhops", 
                         split: str = "test",
                         cache_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load MMhops dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset repository name
        split: Dataset split to load ("test", "train", "validation")
        cache_dir: Optional cache directory for downloaded data
    
    Returns:
        List of dataset samples
    """
    print(f"Loading dataset '{dataset_name}' (split: {split})...")
    
    kwargs = {"path": dataset_name, "split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    
    try:
        dataset = load_dataset(**kwargs)
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        print("Tip: Make sure you have internet access and the dataset exists.")
        exit(1)
    
    samples = list(dataset)
    print(f"Loaded {len(samples)} samples")
    return samples


# ==============================================================================
# Main Evaluation Logic
# ==============================================================================

def run_evaluation(predictions: Dict[str, str], 
                   dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run evaluation and compute metrics.
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    results = {
        "total": {"correct": 0, "total": 0},
        "by_split": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_split_type": defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})),
        "missing_predictions": [],
    }
    
    for sample in dataset:
        sample_id = sample["id"]
        split = sample["split"]
        problem_type = sample["problem_type"]
        answer_eval = sample["answer_eval"]
        
        if sample_id not in predictions:
            results["missing_predictions"].append(sample_id)
            continue
        
        prediction = predictions[sample_id]
        score = evaluate_sample(prediction, answer_eval, problem_type)
        
        results["total"]["correct"] += score
        results["total"]["total"] += 1
        
        results["by_split"][split]["correct"] += score
        results["by_split"][split]["total"] += 1
        
        results["by_type"][problem_type]["correct"] += score
        results["by_type"][problem_type]["total"] += 1
        
        results["by_split_type"][split][problem_type]["correct"] += score
        results["by_split_type"][split][problem_type]["total"] += 1
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results in a formatted table."""
    def calc_acc(d):
        return safe_division(d["correct"], d["total"]) * 100
    
    print("\n" + "=" * 60)
    print("MMhops Evaluation Results")
    print("=" * 60)
    
    # Overall accuracy
    total = results["total"]
    print(f"\nOverall Accuracy: {calc_acc(total):.2f}% ({total['correct']}/{total['total']})")
    
    # By question type
    print("\nBy Question Type:")
    for qtype in ["String", "Numerical", "Time"]:
        if qtype in results["by_type"]:
            d = results["by_type"][qtype]
            print(f"  {qtype:12s}: {calc_acc(d):6.2f}% ({d['correct']:4d}/{d['total']:4d})")
    
    # By split
    print("\nBy Split:")
    for split in ["Bridge", "Compare"]:
        if split in results["by_split"]:
            d = results["by_split"][split]
            print(f"  {split:12s}: {calc_acc(d):6.2f}% ({d['correct']:4d}/{d['total']:4d})")
    
    # Detailed breakdown
    for split in ["Bridge", "Compare"]:
        if split in results["by_split_type"]:
            print(f"\n{split} Breakdown:")
            for qtype in ["String", "Numerical", "Time"]:
                if qtype in results["by_split_type"][split]:
                    d = results["by_split_type"][split][qtype]
                    print(f"  {qtype:12s}: {calc_acc(d):6.2f}% ({d['correct']:4d}/{d['total']:4d})")
    
    # Missing predictions warning
    if results["missing_predictions"]:
        n_missing = len(results["missing_predictions"])
        print(f"\nWarning: {n_missing} samples have no predictions")
        if n_missing <= 5:
            print(f"  Missing IDs: {results['missing_predictions']}")
        else:
            print(f"  First 5 missing IDs: {results['missing_predictions'][:5]}")
    
    print("\n" + "=" * 60)


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    def calc_acc(d):
        return round(safe_division(d["correct"], d["total"]) * 100, 2)
    
    output = {
        "overall_accuracy": calc_acc(results["total"]),
        "total_samples": results["total"]["total"],
        "correct_samples": results["total"]["correct"],
        "by_question_type": {
            qtype: {"accuracy": calc_acc(d), "correct": d["correct"], "total": d["total"]}
            for qtype, d in results["by_type"].items()
        },
        "by_split": {
            split: {"accuracy": calc_acc(d), "correct": d["correct"], "total": d["total"]}
            for split, d in results["by_split"].items()
        },
        "detailed_breakdown": {
            split: {
                qtype: {"accuracy": calc_acc(d), "correct": d["correct"], "total": d["total"]}
                for qtype, d in type_results.items()
            }
            for split, type_results in results["by_split_type"].items()
        },
        "missing_predictions_count": len(results["missing_predictions"]),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions on the MMhops benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prediction File Format (JSONL):
  Each line should be a JSON object with:
  - id: Sample ID (e.g., "MMhops_test_00001")
  - prediction: Model's predicted answer

Example:
  {"id": "MMhops_test_00001", "prediction": "174th"}
  {"id": "MMhops_test_00002", "prediction": "12"}
        """
    )
    
    parser.add_argument(
        "--prediction-file", "-p",
        type=str,
        required=True,
        help="Path to prediction file (JSONL format)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="taoszhang/MMhops",
        help="HuggingFace dataset name (default: taoszhang/MMhops)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "validation"],
        help="Dataset split to evaluate on (default: test)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloaded dataset"
    )
    
    parser.add_argument(
        "--output-json", "-o",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from: {args.prediction_file}")
    predictions = load_predictions(args.prediction_file)
    print(f"Loaded {len(predictions)} predictions")
    
    # Load dataset
    dataset = load_dataset_from_hf(args.dataset, args.split, args.cache_dir)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = run_evaluation(predictions, dataset)
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output_json:
        save_results_json(results, args.output_json)


if __name__ == "__main__":
    main()
