from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

from data_io import (
    REQUIRED_SUBMISSION_COLUMNS,
    REQUIRED_TEST_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    read_header,
    read_rows,
    validate_columns,
)


LABEL_NAMES = {
    0: "not a movie/TV review",
    1: "positive movie/TV review",
    2: "negative movie/TV review",
}


def text_length_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    lengths = [len(row["TEXT"].split()) for row in rows]
    return {
        "min": min(lengths),
        "median": median(lengths),
        "mean": mean(lengths),
        "max": max(lengths),
    }


def null_counts(rows: list[dict[str, str]], columns: list[str]) -> dict[str, int]:
    return {column: sum(1 for row in rows if row.get(column, "") == "") for column in columns}


def duplicate_id_count(rows: list[dict[str, str]]) -> int:
    ids = [row["ID"] for row in rows]
    return len(ids) - len(set(ids))


def write_label_distribution_svg(label_counts: Counter[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 360
    margin_left = 210
    margin_right = 40
    bar_height = 48
    gap = 34
    max_count = max(label_counts.values())
    chart_width = width - margin_left - margin_right

    rows = []
    for index, label in enumerate(sorted(label_counts)):
        count = label_counts[label]
        bar_width = int(chart_width * count / max_count)
        y = 82 + index * (bar_height + gap)
        label_text = f"{label}: {LABEL_NAMES[label]}"
        rows.append(
            f'<text x="24" y="{y + 31}" font-size="15">{label_text}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#4f81bd" />'
            f'<text x="{margin_left + bar_width + 10}" y="{y + 31}" font-size="15">{count:,}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white" />'
        '<text x="24" y="36" font-size="22" font-weight="bold">Training Label Distribution</text>'
        '<text x="24" y="60" font-size="14" fill="#555">Counts by Kaggle class label</text>'
        f'{"".join(rows)}'
        "</svg>\n"
    )
    output_path.write_text(svg, encoding="utf-8")


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header, separator, body])


def write_markdown_report(
    output_path: Path,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    figure_path: Path,
) -> None:
    label_counts = Counter(int(row["LABEL"]) for row in train_rows)
    total_train = len(train_rows)
    train_ids = {row["ID"] for row in train_rows}
    test_ids = {row["ID"] for row in test_rows}
    sample_ids = {row["ID"] for row in sample_rows}
    train_lengths = text_length_summary(train_rows)
    test_lengths = text_length_summary(test_rows)

    dataset_rows = [
        [
            "train",
            str(train_path),
            f"{len(train_rows):,}",
            ", ".join(read_header(train_path)),
            str(null_counts(train_rows, read_header(train_path))),
            str(duplicate_id_count(train_rows)),
        ],
        [
            "test",
            str(test_path),
            f"{len(test_rows):,}",
            ", ".join(read_header(test_path)),
            str(null_counts(test_rows, read_header(test_path))),
            str(duplicate_id_count(test_rows)),
        ],
        [
            "sample submission",
            str(sample_path),
            f"{len(sample_rows):,}",
            ", ".join(read_header(sample_path)),
            str(null_counts(sample_rows, read_header(sample_path))),
            str(duplicate_id_count(sample_rows)),
        ],
    ]

    label_rows = [
        [
            str(label),
            LABEL_NAMES[label],
            f"{label_counts[label]:,}",
            f"{label_counts[label] / total_train:.2%}",
        ]
        for label in sorted(label_counts)
    ]

    text_rows = [
        [
            "train",
            str(train_lengths["min"]),
            f"{train_lengths['median']:.1f}",
            f"{train_lengths['mean']:.1f}",
            str(train_lengths["max"]),
        ],
        [
            "test",
            str(test_lengths["min"]),
            f"{test_lengths['median']:.1f}",
            f"{test_lengths['mean']:.1f}",
            str(test_lengths["max"]),
        ],
    ]

    relative_figure_path = Path("..") / figure_path
    markdown = f"""# Data Analysis

This report documents the Kaggle files used for the first baseline submission.

## Column Checks

{format_table(["Dataset", "Path", "Rows", "Columns", "Empty values", "Duplicate IDs"], dataset_rows)}

## Label Distribution

{format_table(["Label", "Meaning", "Count", "Percent"], label_rows)}

![Training label distribution]({relative_figure_path.as_posix()})

## Text Lengths

Word-count summary after splitting on whitespace:

{format_table(["Dataset", "Min", "Median", "Mean", "Max"], text_rows)}

## Submission ID Check

- Sample submission IDs all appear in `data/test.csv`: `{sample_ids.issubset(test_ids)}`.
- `data/test.csv` has `{len(test_ids - sample_ids):,}` IDs that are not present in `data/sample_submission.csv`.
- Train/test ID overlap count: `{len(train_ids & test_ids):,}`.

Because Kaggle validates the uploaded file against the sample-submission shape, submission generation predicts labels only for the IDs in `data/sample_submission.csv` and preserves that order.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Kaggle data files and write a markdown report.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/data_analysis.md"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/label_distribution.svg"))
    args = parser.parse_args()

    validate_columns(args.train, REQUIRED_TRAIN_COLUMNS)
    validate_columns(args.test, REQUIRED_TEST_COLUMNS)
    validate_columns(args.sample_submission, REQUIRED_SUBMISSION_COLUMNS)

    train_rows = read_rows(args.train)
    test_rows = read_rows(args.test)
    sample_rows = read_rows(args.sample_submission)
    label_counts = Counter(int(row["LABEL"]) for row in train_rows)

    write_label_distribution_svg(label_counts, args.figure)
    write_markdown_report(
        args.report,
        train_rows,
        test_rows,
        sample_rows,
        args.train,
        args.test,
        args.sample_submission,
        args.figure,
    )
    print(f"Wrote {args.report}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
