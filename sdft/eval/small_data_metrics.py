import re

_TASK_LABELS = {
    "copa": ["choice1", "choice2"],
    "cb": ["entailment", "contradiction", "neutral"],
    "wsc": ["True", "False"],
}

_FINAL_LABEL_PATTERN = re.compile(r"final\s*label\s*:\s*([^\n\r]+)", re.IGNORECASE)
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9_ ]+")


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in _TASK_LABELS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {sorted(_TASK_LABELS)}")
    return normalized


def _normalize_label(task: str, raw_text: str) -> str | None:
    cleaned = _NON_ALNUM_PATTERN.sub(" ", raw_text.strip().lower())
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    labels = _TASK_LABELS[task]
    for label in labels:
        label_lower = label.lower()
        if cleaned == label_lower:
            return label
        if cleaned.startswith(label_lower + " "):
            return label
    return None


def _parse_prediction_label(prediction: str, task: str) -> tuple[str | None, bool]:
    # Primary parse route: explicit final label marker.
    matches = _FINAL_LABEL_PATTERN.findall(prediction)
    for candidate in reversed(matches):
        normalized = _normalize_label(task, candidate)
        if normalized is not None:
            return normalized, True
    # Strict parsing only: no fallback to free-text label mentions.
    return None, False


def score_small_data_predictions(
    predictions: list[str],
    references: list[str],
    tasks: list[str],
) -> dict[str, list[float]]:
    if len(predictions) != len(references) or len(predictions) != len(tasks):
        raise ValueError("Predictions, references, and tasks must have the same length.")

    accuracy_scores: list[float] = []
    parse_scores: list[float] = []

    for prediction, reference, task in zip(predictions, references, tasks):
        normalized_task = _normalize_task(task)
        normalized_reference = _normalize_label(normalized_task, reference)
        if normalized_reference is None:
            raise ValueError(f"Invalid reference label '{reference}' for task '{task}'.")

        parsed_label, parse_success = _parse_prediction_label(prediction, normalized_task)
        parse_scores.append(1.0 if parse_success else 0.0)
        accuracy_scores.append(1.0 if parsed_label == normalized_reference else 0.0)

    return {
        "accuracy": accuracy_scores,
        "parse_success": parse_scores,
    }
