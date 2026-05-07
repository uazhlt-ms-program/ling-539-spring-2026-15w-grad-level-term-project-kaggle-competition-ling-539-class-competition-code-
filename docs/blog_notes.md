# Blog Post Draft

This draft can be adapted for the required GitHub Pages course blog post.

## Classifying Movie and TV Review Text with TF-IDF and Logistic Regression

For the LING 539 class Kaggle competition, I built a three-class text classifier for short and long review-like documents. The labels were:

- `0`: not a movie/TV review
- `1`: positive movie/TV review
- `2`: negative movie/TV review

Kaggle evaluates submissions with macro F1. That metric is important for this task because it gives each class equal weight, even though the training set is not perfectly balanced. A model that performs well only on the most common class can still score poorly if it misses positive or negative movie/TV reviews.

## Data Analysis

The training file has 70,317 rows with columns `ID`, `TEXT`, and `LABEL`. The test file has 17,580 rows with columns `ID` and `TEXT`. The sample submission has 17,580 rows with columns `ID` and `LABEL`.

Before training, I checked the expected columns, missing values, duplicate IDs, label distribution, text lengths, and test/sample-submission ID alignment. The sample submission IDs matched the test IDs, so the final prediction script preserves that ID order exactly when writing the Kaggle CSV.

Training label distribution:

| Label | Meaning | Count | Percent |
| --- | --- | --- | --- |
| 0 | not a movie/TV review | 32,289 | 45.92% |
| 1 | positive movie/TV review | 19,139 | 27.22% |
| 2 | negative movie/TV review | 18,889 | 26.86% |

The label distribution is somewhat imbalanced, with class `0` appearing most often. Because the competition uses macro F1, I evaluated the model with macro F1 and also inspected per-class F1 scores.

The text lengths varied widely. The median training example had about 104 whitespace-separated words, while the longest examples were much longer. This made TF-IDF a good baseline feature representation because it can handle both short and long documents efficiently.

## Model Approach

The submitted model is a scikit-learn pipeline with TF-IDF features and logistic regression:

- Text preprocessing: unescape HTML entities and replace `<br />` line breaks with spaces
- Features: `TfidfVectorizer` with word unigrams and bigrams
- Feature filtering: `min_df=2`, `max_df=0.95`, and up to 100,000 features
- Weighting: sublinear term frequency scaling
- Classifier: `LogisticRegression` with balanced class weights

I chose logistic regression because it is a course-covered classifier and is a strong baseline for sparse text features. TF-IDF captures which words and short phrases are distinctive for each class, while logistic regression learns a linear decision boundary over those features. Balanced class weights help account for the fact that class `0` is more common than classes `1` and `2`.

## Validation Results

I used an 80/20 stratified train/validation split. Stratification keeps the class proportions similar in the training and validation portions, which is important for a three-class macro F1 task.

Validation macro F1: `0.9173`.

Per-class validation F1:

| Label | F1 |
| --- | --- |
| 0 | 0.9690 |
| 1 | 0.8840 |
| 2 | 0.8989 |

The model performed best on class `0`, the non-movie/TV-review class. The main remaining errors were between positive and negative movie/TV reviews. That makes sense because classes `1` and `2` are semantically closer to each other than either one is to unrelated non-review text.

Validation confusion matrix:

| actual \ predicted | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 6287 | 108 | 63 |
| 1 | 158 | 3383 | 287 |
| 2 | 73 | 335 | 3370 |

## Reproducibility

The repository includes scripts for both data inspection and submission generation. To reproduce the submission:

```bash
python src/inspect_data.py
python src/train_baseline.py
```

The output file is `submissions/baseline_tfidf_logreg.csv`.

The training script performs validation, retrains on the full training set, loads the test set, and writes predictions using the exact IDs from `data/sample_submission.csv`.

## Kaggle Result

The first Kaggle upload used `submissions/baseline_tfidf_logreg.csv`.

- Submission message: `TF-IDF logistic regression baseline`
- Public leaderboard score: `0.92189`

## What I Learned

This project showed why validation design matters for Kaggle competitions. Since the leaderboard metric is macro F1, accuracy alone would not be enough to judge the model. Looking at per-class F1 and the confusion matrix made it clear that the hardest part was separating positive and negative reviews, not identifying non-review text.

It also showed that a simple, well-documented baseline can be very competitive. TF-IDF plus logistic regression is fast, reproducible, and interpretable enough to explain clearly. The final repository includes the code and commands needed to regenerate the submission.

## Future Improvements

If I continued iterating, I would try tuning the logistic regression regularization strength, comparing against Multinomial Naive Bayes or linear SVM models, and adding character n-gram TF-IDF features. Those changes might help with misspellings, punctuation, and subtle positive-versus-negative sentiment cues.
