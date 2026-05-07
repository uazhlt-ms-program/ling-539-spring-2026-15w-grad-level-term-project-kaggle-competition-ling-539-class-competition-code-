from __future__ import annotations

import argparse
from collections import Counter
from html import unescape
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_io import load_submission_ids, load_test_by_id, load_train, write_submission


RANDOM_STATE = 539


def normalize_text(text: str) -> str:
    return unescape(text).replace("<br />", " ").replace("<br/>", " ")


def build_model(max_features: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def format_confusion_matrix(matrix: list[list[int]], labels: list[int]) -> str:
    header = "| actual \\ predicted | " + " | ".join(str(label) for label in labels) + " |"
    separator = "| " + " | ".join("---" for _ in range(len(labels) + 1)) + " |"
    rows = [
        "| "
        + " | ".join([str(label)] + [str(value) for value in matrix[index]])
        + " |"
        for index, label in enumerate(labels)
    ]
    return "\n".join([header, separator, *rows])


def write_results_report(
    output_path: Path,
    macro_f1: float,
    validation_size: float,
    max_features: int,
    labels: list[int],
    report: str,
    matrix: list[list[int]],
    train_label_counts: Counter[int],
    submission_path: Path,
) -> None:
    label_rows = "\n".join(
        f"| {label} | {train_label_counts[label]:,} |" for label in sorted(train_label_counts)
    )
    markdown = f"""# Baseline Model Results

## Model

- Features: TF-IDF word unigrams and bigrams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features={max_features:,}`.
- Classifier: scikit-learn `LogisticRegression`, a course-covered linear classifier, with balanced class weights.
- Validation: stratified train/validation split with `{validation_size:.0%}` of training data held out.
- Metric: macro F1, matching the Kaggle emphasis on all three classes.

## Training Label Counts

| Label | Count |
| --- | --- |
{label_rows}

## Validation

- Macro F1: `{macro_f1:.4f}`

```text
{report.strip()}
```

Confusion matrix:

{format_confusion_matrix(matrix, labels)}

## Submission

Generated first baseline submission at `{submission_path}`.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF logistic regression baseline and write submission.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/baseline_tfidf_logreg.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/baseline_results.md"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=100_000)
    args = parser.parse_args()

    texts, labels, _ = load_train(args.train)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=args.validation_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    validation_model = build_model(args.max_features)
    validation_model.fit(train_texts, train_labels)
    validation_predictions = validation_model.predict(validation_texts)
    macro_f1 = f1_score(validation_labels, validation_predictions, average="macro")
    sorted_labels = sorted(set(labels))
    report = classification_report(validation_labels, validation_predictions, labels=sorted_labels, digits=4)
    matrix = confusion_matrix(validation_labels, validation_predictions, labels=sorted_labels).tolist()

    full_model = build_model(args.max_features)
    full_model.fit(texts, labels)

    test_text_by_id = load_test_by_id(args.test)
    submission_ids = load_submission_ids(args.sample_submission)
    missing_ids = [submission_id for submission_id in submission_ids if submission_id not in test_text_by_id]
    if missing_ids:
        raise ValueError(f"Sample submission contains IDs missing from test data: {missing_ids[:5]}")

    submission_texts = [test_text_by_id[submission_id] for submission_id in submission_ids]
    submission_predictions = full_model.predict(submission_texts)
    write_submission(args.output, submission_ids, submission_predictions)
    write_results_report(
        args.report,
        macro_f1,
        args.validation_size,
        args.max_features,
        sorted_labels,
        report,
        matrix,
        Counter(labels),
        args.output,
    )

    print(f"Validation macro F1: {macro_f1:.4f}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
