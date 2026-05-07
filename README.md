[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uhZ6joRH)
# LING 539 Kaggle Text Classification Project

This repository contains a reproducible text classification system for the [LING 539 Spring 2026 class Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026).

The task is a three-class NLP classification problem over text:

- `0`: not a movie/TV review
- `1`: positive movie/TV review
- `2`: negative movie/TV review

Kaggle scores submissions with macro F1, so the model is evaluated on balanced performance across all three labels rather than only overall accuracy.

## Final Model

The submitted system uses a scikit-learn pipeline:

- `TfidfVectorizer` with word unigrams and bigrams
- HTML line-break cleanup and HTML entity unescaping
- `LogisticRegression` with balanced class weights

Logistic regression is a course-covered classifier and is well suited to sparse TF-IDF features for text classification.

Final recorded results:

- Validation macro F1: `0.9173`
- Kaggle public leaderboard score: `0.92189`
- Submission file: `submissions/baseline_tfidf_logreg.csv`

## Repository Structure

```text
data/
  train.csv
  test.csv
  sample_submission.csv
docs/
  data_analysis.md
  baseline_results.md
  blog_notes.md
reports/
  figures/label_distribution.svg
src/
  data_io.py
  inspect_data.py
  train_baseline.py
submissions/
  baseline_tfidf_logreg.csv
```

## Data

The Kaggle files should be placed in `data/`:

- `data/train.csv` with columns `ID,TEXT,LABEL`
- `data/test.csv` with columns `ID,TEXT`
- `data/sample_submission.csv` with columns `ID,LABEL`

The data analysis report is in `docs/data_analysis.md`, with a label-distribution visual in `reports/figures/label_distribution.svg`.

Training label distribution:

| Label | Meaning | Count | Percent |
| --- | --- | ---: | ---: |
| 0 | not a movie/TV review | 32,289 | 45.92% |
| 1 | positive movie/TV review | 19,139 | 27.22% |
| 2 | negative movie/TV review | 18,889 | 26.86% |

## Reproduce the Submission

Install dependencies with a supported Python environment:

```bash
python -m pip install -r requirements.txt
```

Inspect the data and regenerate the analysis report:

```bash
python src/inspect_data.py
```

Train the final TF-IDF + logistic regression model and regenerate the Kaggle submission CSV:

```bash
python src/train_baseline.py
```

This writes:

```text
docs/baseline_results.md
submissions/baseline_tfidf_logreg.csv
```

The training script:

- loads `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv`
- performs a stratified validation split
- reports macro F1, per-class F1, and a confusion matrix
- retrains on the full training set
- writes predictions in the exact sample-submission ID order

## Submit to Kaggle

If using the Kaggle CLI, install it and configure Kaggle credentials locally:

```bash
python -m pip install kaggle
```

Then upload the generated submission:

```bash
kaggle competitions submit \
  -c ling-539-competition-2026 \
  -f submissions/baseline_tfidf_logreg.csv \
  -m "TF-IDF logistic regression baseline"
```

The first baseline upload completed successfully with public score `0.92189`.

## Reports

- `docs/data_analysis.md`: schema checks, label distribution, text-length summary, and submission ID validation
- `docs/baseline_results.md`: final validation metrics, classification report, confusion matrix, and Kaggle result
- `docs/blog_notes.md`: blog-post draft content for the required course write-up
