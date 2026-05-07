from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


REQUIRED_TRAIN_COLUMNS = ("ID", "TEXT", "LABEL")
REQUIRED_TEST_COLUMNS = ("ID", "TEXT")
REQUIRED_SUBMISSION_COLUMNS = ("ID", "LABEL")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        return next(reader)


def validate_columns(path: Path, required_columns: Iterable[str]) -> None:
    header = read_header(path)
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_train(path: Path) -> tuple[list[str], list[int], list[str]]:
    validate_columns(path, REQUIRED_TRAIN_COLUMNS)
    rows = read_rows(path)
    ids = [row["ID"] for row in rows]
    texts = [row["TEXT"] for row in rows]
    labels = [int(row["LABEL"]) for row in rows]
    return texts, labels, ids


def load_test_by_id(path: Path) -> dict[str, str]:
    validate_columns(path, REQUIRED_TEST_COLUMNS)
    return {row["ID"]: row["TEXT"] for row in read_rows(path)}


def load_submission_ids(path: Path) -> list[str]:
    validate_columns(path, REQUIRED_SUBMISSION_COLUMNS)
    return [row["ID"] for row in read_rows(path)]


def write_submission(path: Path, ids: list[str], labels: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ID", "LABEL"])
        writer.writerows(zip(ids, labels))
