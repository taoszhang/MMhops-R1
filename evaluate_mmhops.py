#!/usr/bin/env python3
"""Official evaluator for the MMhops benchmark.

The evaluator accepts one final answer per sample in JSONL format::

    {"id": "MMhops_test_00001", "prediction": "174th"}

Every reference sample is included in the denominator. Missing predictions are
therefore incorrect, while duplicate or unknown prediction IDs are rejected.
The evaluator deliberately does not extract answers from model trajectories or
score retrieval actions and output formatting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from word2number import w2n


EVALUATOR_NAME = "MMhops official evaluator"
EVALUATOR_VERSION = "1.0.0"
DATASET_NAME = "taoszhang/MMhops"
DATASET_REVISION = "20bca45129d28f56bfefcffe7df0c080757fe9a7"
EXPECTED_SPLIT_SIZES = {"train": 21777, "validation": 3112, "test": 6223}

CANONICAL_SPLITS = {"bridge": "Bridge", "compare": "Compare"}
CANONICAL_QUESTION_TYPES = {
    "string": "String",
    "numerical": "Numerical",
    "time": "Time",
}
SPLIT_ORDER = ("Bridge", "Compare")
QUESTION_TYPE_ORDER = ("String", "Numerical", "Time")

NumericAnswer = Union[float, List[float]]
MetricCounter = Dict[str, int]


class EvaluationError(ValueError):
    """Raised when predictions or references violate the evaluation contract."""


@dataclass(frozen=True)
class ReferenceSample:
    """Validated fields required to score one MMhops sample."""

    sample_id: str
    split: str
    question_type: str
    answer_eval: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset identity recorded alongside evaluation results."""

    name: str
    split: str
    revision: str
    fingerprint: str


# -----------------------------------------------------------------------------
# Paper evaluation kernel
# -----------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize String and Time answers exactly as in the paper evaluator."""

    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Return whether two answers match after paper-compatible normalization."""

    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_exact_match(prediction: str, ground_truths: Sequence[str]) -> int:
    """Score a String or Time answer against all released aliases."""

    return max(int(exact_match_score(prediction, answer)) for answer in ground_truths)


_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|trillion|"
    r"point|and|[-\s])+\b",
    re.IGNORECASE,
)
_NUMERICAL_VALUE_PATTERN = re.compile(
    r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
)


def replace_number_words(text: str) -> str:
    """Convert English number-word spans using the paper evaluator's rule."""

    def convert(match: re.Match[str]) -> str:
        try:
            return str(w2n.word_to_num(match.group()))
        except ValueError:
            return match.group()

    return _NUMBER_WORD_PATTERN.sub(convert, text)


def find_all(text: str, character: str) -> Generator[int, None, None]:
    """Yield every occurrence of ``character`` in ``text``."""

    index = text.find(character)
    while index != -1:
        yield index
        index = text.find(character, index + 1)


def clean_str_range(text: str) -> str:
    """Separate a hyphen following a digit, as in ``9-10``."""

    replace_at = {index for index in find_all(text, "-") if index >= 1 and text[index - 1].isdigit()}
    return "".join(" - " if index in replace_at else character for index, character in enumerate(text))


def process_numerical_answer(string_number: str) -> NumericAnswer:
    """Parse a numerical prediction with the evaluator used for paper results.

    The historical rule converts English number words, extracts at most the
    first two numerical values, and treats two ascending values as a range.
    This behavior is intentionally retained for metric compatibility.
    """

    try:
        string_number = replace_number_words(string_number)
    except Exception:
        string_number = str(string_number)

    values_as_text = _NUMERICAL_VALUE_PATTERN.findall(clean_str_range(string_number))
    values: List[float] = []
    for value_as_text in values_as_text[:2]:
        value_as_text = value_as_text.replace(",", "").strip(".")
        if value_as_text.count(".") > 1:
            value_as_text = value_as_text.split(".")[0]
        values.append(float(value_as_text))

    if len(values) == 2:
        first_value, second_value = values
        return [first_value, second_value] if first_value <= second_value else first_value
    if len(values) == 1:
        return values[0]
    return [0.0, 0.0]


def safe_division(numerator: float, denominator: float) -> float:
    """Divide safely, returning zero for an empty denominator."""

    return numerator / denominator if denominator else 0.0


def range_intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    """Calculate continuous IoU between two numerical ranges."""

    first_min, first_max = min(first), max(first)
    second_min, second_max = min(second), max(second)
    overlap = max(0.0, min(first_max, second_max) - max(first_min, second_min))
    first_length = (first_max - first_min) + 1e-12
    second_length = (second_max - second_min) + 1e-12
    return safe_division(overlap, first_length + second_length - overlap)


def evaluate_numerical(prediction: str, ground_truths: Sequence[str]) -> int:
    """Score a scalar or range against the released ``answer_eval`` interval."""

    prediction_value = process_numerical_answer(prediction)
    reference_values = [float(value) for value in ground_truths]
    reference_min = min(reference_values)
    reference_max = max(reference_values)

    if isinstance(prediction_value, list):
        if (
            reference_min <= prediction_value[0] <= reference_max
            and reference_min <= prediction_value[1] <= reference_max
        ):
            return 1
        iou = range_intersection_over_union(prediction_value, [reference_min, reference_max])
        return int(iou >= 0.5 - 1e-12)
    return int(reference_min <= prediction_value <= reference_max)


def evaluate_sample(prediction: str, ground_truths: Sequence[str], question_type: str) -> int:
    """Score one final answer using the metric reported in the paper."""

    # A missing or explicitly empty final answer is never a valid submission.
    # This also avoids the historical numeric fallback ``[0, 0]`` accidentally
    # rewarding an empty prediction for a reference interval containing zero.
    if not prediction.strip():
        return 0
    if question_type in {"String", "Time"}:
        return evaluate_exact_match(prediction, ground_truths)
    if question_type == "Numerical":
        return evaluate_numerical(prediction, ground_truths)
    raise EvaluationError(f"Unsupported question type: {question_type!r}")


# -----------------------------------------------------------------------------
# Input and reference validation
# -----------------------------------------------------------------------------


def load_predictions(filepath: Union[str, Path]) -> Dict[str, str]:
    """Load prediction JSONL and reject malformed or duplicate records."""

    path = Path(filepath)
    if not path.is_file():
        raise EvaluationError(f"Prediction file does not exist: {path}")

    predictions: Dict[str, str] = {}
    first_line_by_id: Dict[str, int] = {}
    try:
        file = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Unable to read prediction file {path}: {exc}") from exc

    with file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise EvaluationError(f"Line {line_number} must be a JSON object.")
            if "id" not in item or "prediction" not in item:
                raise EvaluationError(f"Line {line_number} must contain both 'id' and 'prediction'.")

            sample_id = item["id"]
            prediction = item["prediction"]
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise EvaluationError(f"Line {line_number} has an invalid 'id'; expected a non-empty string.")
            if not isinstance(prediction, str):
                raise EvaluationError(f"Line {line_number} has a non-string 'prediction'.")
            if sample_id in predictions:
                first_line = first_line_by_id[sample_id]
                raise EvaluationError(
                    f"Duplicate prediction ID {sample_id!r} on lines {first_line} and {line_number}."
                )

            predictions[sample_id] = prediction
            first_line_by_id[sample_id] = line_number

    return predictions


def _canonical_field(value: Any, mapping: Mapping[str, str], field_name: str, sample_id: str) -> str:
    if not isinstance(value, str):
        raise EvaluationError(f"Reference {sample_id!r} has a non-string {field_name!r} field.")
    canonical = mapping.get(value.lower())
    if canonical is None:
        supported = ", ".join(mapping.values())
        raise EvaluationError(f"Reference {sample_id!r} has unsupported {field_name}={value!r}; expected {supported}.")
    return canonical


def validate_references(dataset: Iterable[Mapping[str, Any]]) -> List[ReferenceSample]:
    """Validate the released dataset schema and return immutable references."""

    references: List[ReferenceSample] = []
    seen_ids = set()
    required_fields = {"id", "split", "problem_type", "answer_eval"}

    for row_number, sample in enumerate(dataset, 1):
        if not isinstance(sample, Mapping):
            raise EvaluationError(f"Reference row {row_number} is not an object.")
        missing_fields = sorted(required_fields - set(sample))
        if missing_fields:
            raise EvaluationError(f"Reference row {row_number} is missing fields: {', '.join(missing_fields)}.")

        sample_id = sample["id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise EvaluationError(f"Reference row {row_number} has an invalid ID.")
        if sample_id in seen_ids:
            raise EvaluationError(f"Duplicate reference ID: {sample_id!r}.")
        seen_ids.add(sample_id)

        split = _canonical_field(sample["split"], CANONICAL_SPLITS, "split", sample_id)
        question_type = _canonical_field(
            sample["problem_type"], CANONICAL_QUESTION_TYPES, "problem_type", sample_id
        )
        answer_eval = sample["answer_eval"]
        if isinstance(answer_eval, (str, bytes)) or not isinstance(answer_eval, Sequence) or not answer_eval:
            raise EvaluationError(f"Reference {sample_id!r} has an invalid or empty 'answer_eval'.")
        if any(not isinstance(answer, str) or not answer.strip() for answer in answer_eval):
            raise EvaluationError(f"Reference {sample_id!r} contains an invalid answer alias.")

        aliases = tuple(answer_eval)
        if question_type == "Numerical":
            if len(aliases) != 2:
                raise EvaluationError(
                    f"Numerical reference {sample_id!r} must contain exactly two range bounds; got {len(aliases)}."
                )
            for bound in aliases:
                try:
                    parsed_bound = float(bound)
                except ValueError as exc:
                    raise EvaluationError(
                        f"Numerical reference {sample_id!r} contains a non-numeric bound: {bound!r}."
                    ) from exc
                if not math.isfinite(parsed_bound):
                    raise EvaluationError(
                        f"Numerical reference {sample_id!r} contains a non-finite bound: {bound!r}."
                    )

        references.append(
            ReferenceSample(
                sample_id=sample_id,
                split=split,
                question_type=question_type,
                answer_eval=aliases,
            )
        )

    if not references:
        raise EvaluationError("The reference dataset is empty.")
    return references


def load_official_dataset(
    split: str,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Mapping[str, Any]], DatasetMetadata]:
    """Load one split of the official Hugging Face dataset."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise EvaluationError(
            "Missing dependency 'datasets'. Install it with: pip install -r requirements.txt"
        ) from exc

    kwargs: Dict[str, Any] = {
        "path": DATASET_NAME,
        "split": split,
        "revision": DATASET_REVISION,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        dataset = load_dataset(**kwargs)
    except Exception as exc:
        raise EvaluationError(f"Unable to load {DATASET_NAME!r} split {split!r}: {exc}") from exc

    expected_size = EXPECTED_SPLIT_SIZES[split]
    if len(dataset) != expected_size:
        raise EvaluationError(
            f"Official {split!r} split has {len(dataset)} samples; expected {expected_size}. "
            "Check the dataset revision and local cache."
        )

    reference_columns = ["id", "split", "problem_type", "answer_eval"]
    missing_columns = sorted(set(reference_columns) - set(dataset.column_names))
    if missing_columns:
        raise EvaluationError(f"Official dataset is missing columns: {', '.join(missing_columns)}.")

    metadata = DatasetMetadata(
        name=DATASET_NAME,
        split=split,
        revision=DATASET_REVISION,
        fingerprint=str(getattr(dataset, "_fingerprint", "unknown")),
    )
    # Avoid decoding the image column: scoring only needs these four fields.
    references = dataset.select_columns(reference_columns)
    return list(references), metadata


# -----------------------------------------------------------------------------
# Aggregation and reporting
# -----------------------------------------------------------------------------


def _new_counter() -> MetricCounter:
    return {"correct": 0, "total": 0}


def _update_counter(counter: MetricCounter, score: int) -> None:
    counter["correct"] += score
    counter["total"] += 1


def run_evaluation(predictions: Mapping[str, str], dataset: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Evaluate predictions with every reference sample in the denominator."""

    references = validate_references(dataset)
    reference_ids = {reference.sample_id for reference in references}
    unknown_ids = sorted(set(predictions) - reference_ids)
    if unknown_ids:
        preview = ", ".join(repr(sample_id) for sample_id in unknown_ids[:5])
        suffix = " ..." if len(unknown_ids) > 5 else ""
        raise EvaluationError(f"Found {len(unknown_ids)} unknown prediction IDs: {preview}{suffix}")

    overall = _new_counter()
    by_split: Dict[str, MetricCounter] = {}
    by_question_type: Dict[str, MetricCounter] = {}
    by_split_and_type: Dict[str, Dict[str, MetricCounter]] = {}
    missing_prediction_ids: List[str] = []

    for reference in references:
        if reference.sample_id in predictions:
            score = evaluate_sample(
                predictions[reference.sample_id],
                reference.answer_eval,
                reference.question_type,
            )
        else:
            score = 0
            missing_prediction_ids.append(reference.sample_id)

        _update_counter(overall, score)
        _update_counter(by_split.setdefault(reference.split, _new_counter()), score)
        _update_counter(by_question_type.setdefault(reference.question_type, _new_counter()), score)
        split_types = by_split_and_type.setdefault(reference.split, {})
        _update_counter(split_types.setdefault(reference.question_type, _new_counter()), score)

    return {
        "overall": overall,
        "by_split": by_split,
        "by_question_type": by_question_type,
        "by_split_and_question_type": by_split_and_type,
        "provided_predictions": len(predictions),
        "missing_prediction_ids": missing_prediction_ids,
    }


def _metric_output(counter: MetricCounter) -> Dict[str, Union[int, float]]:
    total = counter["total"]
    accuracy = 100.0 * counter["correct"] / total if total else 0.0
    return {
        "accuracy": round(accuracy, 2),
        "correct": counter["correct"],
        "total": total,
    }


def serialize_results(results: Mapping[str, Any], metadata: DatasetMetadata) -> Dict[str, Any]:
    """Create the deterministic public JSON result schema."""

    total = results["overall"]["total"]
    provided = results["provided_predictions"]
    coverage = 100.0 * provided / total if total else 0.0

    by_split = {
        split: _metric_output(results["by_split"][split])
        for split in SPLIT_ORDER
        if split in results["by_split"]
    }
    by_question_type = {
        question_type: _metric_output(results["by_question_type"][question_type])
        for question_type in QUESTION_TYPE_ORDER
        if question_type in results["by_question_type"]
    }
    detailed: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {}
    for split in SPLIT_ORDER:
        if split not in results["by_split_and_question_type"]:
            continue
        detailed[split] = {
            question_type: _metric_output(results["by_split_and_question_type"][split][question_type])
            for question_type in QUESTION_TYPE_ORDER
            if question_type in results["by_split_and_question_type"][split]
        }

    return {
        "evaluator": {"name": EVALUATOR_NAME, "version": EVALUATOR_VERSION},
        "dataset": {
            "name": metadata.name,
            "split": metadata.split,
            "revision": metadata.revision,
            "fingerprint": metadata.fingerprint,
            "samples": total,
        },
        "coverage": {
            "accuracy": round(coverage, 2),
            "provided": provided,
            "missing": len(results["missing_prediction_ids"]),
            "total": total,
        },
        "overall": _metric_output(results["overall"]),
        "by_split": by_split,
        "by_question_type": by_question_type,
        "by_split_and_question_type": detailed,
        "missing_prediction_ids": list(results["missing_prediction_ids"]),
    }


def _format_metric(label: str, metric: Mapping[str, Union[int, float]]) -> str:
    return f"  {label:12s} {metric['accuracy']:6.2f}%  ({metric['correct']}/{metric['total']})"


def print_results(output: Mapping[str, Any]) -> None:
    """Print the official metrics and their exact denominators."""

    print("\n" + "=" * 64)
    print(f"MMhops Official Evaluation v{output['evaluator']['version']}")
    print("=" * 64)
    dataset = output["dataset"]
    print(f"Dataset:  {dataset['name']} ({dataset['split']}, {dataset['samples']} samples)")
    print(f"Revision: {dataset['revision']}")
    print(f"Fingerprint: {dataset['fingerprint']}")

    coverage = output["coverage"]
    print(f"\nCoverage: {coverage['accuracy']:.2f}% ({coverage['provided']}/{coverage['total']})")
    if coverage["missing"]:
        missing_ids = output["missing_prediction_ids"]
        preview = ", ".join(missing_ids[:5])
        suffix = " ..." if len(missing_ids) > 5 else ""
        print(f"Missing:  {coverage['missing']} (counted as incorrect; first IDs: {preview}{suffix})")

    # The paper reports the three Bridge answer types followed by aggregate
    # Bridge and Compare accuracy. Keep those numbers together and prominent.
    print("\nPaper Metrics")
    bridge_types = output["by_split_and_question_type"].get("Bridge", {})
    for question_type in QUESTION_TYPE_ORDER:
        if question_type in bridge_types:
            print(_format_metric(question_type, bridge_types[question_type]))
    for split in SPLIT_ORDER:
        if split in output["by_split"]:
            print(_format_metric(split, output["by_split"][split]))

    print("\nMicro Average (all samples)")
    print(_format_metric("Overall", output["overall"]))
    print("=" * 64)


def save_results_json(output: Mapping[str, Any], output_path: Union[str, Path]) -> None:
    """Write the complete, auditable result object to JSON."""

    path = Path(output_path)
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise EvaluationError(f"Unable to write result JSON to {path}: {exc}") from exc
    print(f"\nResults saved to: {path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate final answers on the official MMhops dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Prediction JSONL format:
  {"id": "MMhops_test_00001", "prediction": "174th"}
  {"id": "MMhops_test_00002", "prediction": "12"}

Missing predictions are counted as incorrect. Duplicate and unknown IDs are errors.
The prediction field must contain the final answer, not a model trajectory.
""",
    )
    parser.add_argument(
        "--prediction-file",
        "-p",
        required=True,
        help="Prediction JSONL file.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED_SPLIT_SIZES),
        default="test",
        help="Official dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default=None,
        help="Optional path for the complete result JSON.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        print(f"Loading predictions: {args.prediction_file}")
        predictions = load_predictions(args.prediction_file)
        print(f"Loaded {len(predictions)} predictions")

        print(f"Loading dataset: {DATASET_NAME} ({args.split})")
        dataset, metadata = load_official_dataset(
            split=args.split,
            cache_dir=args.cache_dir,
        )
        results = run_evaluation(predictions, dataset)
        output = serialize_results(results, metadata)
        print_results(output)
        if args.output_json is not None:
            save_results_json(output, args.output_json)
    except EvaluationError as exc:
        parser.exit(2, f"error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
