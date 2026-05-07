# Baseline Model Results

## Model

- Features: TF-IDF word unigrams and bigrams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=100,000`.
- Classifier: scikit-learn `LogisticRegression`, a course-covered linear classifier, with balanced class weights.
- Validation: stratified train/validation split with `20%` of training data held out.
- Metric: macro F1, matching the Kaggle emphasis on all three classes.

## Training Label Counts

| Label | Count |
| --- | --- |
| 0 | 32,289 |
| 1 | 19,139 |
| 2 | 18,889 |

## Validation

- Macro F1: `0.9173`

```text
precision    recall  f1-score   support

           0     0.9646    0.9735    0.9690      6458
           1     0.8842    0.8838    0.8840      3828
           2     0.9059    0.8920    0.8989      3778

    accuracy                         0.9272     14064
   macro avg     0.9182    0.9164    0.9173     14064
weighted avg     0.9269    0.9272    0.9270     14064
```

Confusion matrix:

| actual \ predicted | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 6287 | 108 | 63 |
| 1 | 158 | 3383 | 287 |
| 2 | 73 | 335 | 3370 |

## Submission

Generated first baseline submission at `submissions/baseline_tfidf_logreg.csv`.

Kaggle upload completed with public leaderboard score `0.92189`.
