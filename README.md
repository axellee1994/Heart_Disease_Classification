# Heart Disease Classification

A compact, end‑to‑end machine learning workflow for predicting heart disease from clinical variables using scikit‑learn. The workflow covers EDA, model baselines, hyperparameter tuning, evaluation beyond accuracy, and exporting a trained model. See the notebook: [end-to-end-heart-disease-classification.ipynb](./end-to-end-heart-disease-classification.ipynb).

Dataset
- Source: UCI Cleveland Heart Disease dataset (via Kaggle mirror).
  - UCI: https://archive.ics.uci.edu/dataset/45/heart+disease
  - Kaggle: https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset
- CSV expected at: [data/heart-disease-UCI.csv](./data/heart-disease-UCI.csv)

Results summary (current)
- Best model: tuned RandomForestClassifier
- Test accuracy ≈ 0.74–0.78; ROC‑AUC ≈ 0.861
- High recall on positives (~94%) with more false positives; 95% accuracy target not met
- Primary signal from stress‑test/chest‑pain features (cp, thal, thalach, oldpeak, slope, exang, ca)

Note: This is a proof‑of‑concept, not clinical advice or a production system.

## Techniques demonstrated

- Stratified data splitting (80/10/10) with `train_test_split` (stratify)  
  scikit‑learn: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Baseline comparison across models (LogisticRegression, KNN, RandomForest)  
  - LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
  - KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
  - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Pipelined preprocessing for linear/KNN models with `Pipeline` + `StandardScaler`  
  Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html  
  StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Hyperparameter search
  - RandomizedSearchCV (distributions from `scipy.stats`)  
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html  
    SciPy distributions: https://docs.scipy.org/doc/scipy/reference/stats.html
  - GridSearchCV  
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- Evaluation beyond accuracy
  - Cross‑validation with multiple scorers via `cross_val_score`  
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
  - ROC curve and ROC‑AUC (`roc_curve`, `roc_auc_score`)  
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html  
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
  - Confusion matrix with `ConfusionMatrixDisplay` and `classification_report`  
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html  
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- Feature importance via `RandomForest.feature_importances_` (tree‑based impurity)  
  https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
- EDA with `pandas.crosstab`, correlation matrix, and `seaborn` heatmaps  
  pandas: https://pandas.pydata.org/docs/  
  seaborn heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html
- Model export with `joblib.dump` to [models/](./models/)  
  joblib: https://joblib.readthedocs.io/en/latest/persistence.html

## Notable libraries and components

- scikit‑learn (models, pipelines, metrics, CV search) — https://scikit-learn.org/
- pandas (data wrangling) — https://pandas.pydata.org/
- NumPy (numerics) — https://numpy.org/
- seaborn and matplotlib (visualization) — https://seaborn.pydata.org/ | https://matplotlib.org/
- SciPy stats for randomized search distributions — https://scipy.org/
- joblib for artifact persistence — https://joblib.readthedocs.io/

## Project structure

```text
.
├── README.md
├── end-to-end-heart-disease-classification.ipynb
├── data/
└── models/
```

- data/: input CSVs (e.g., heart-disease-UCI.csv)
- models/: serialized model artifacts (e.g., best_random_forest_model.joblib)

## Reproducibility and artifacts

- Random seeds set where possible (`np.random.seed(42)` and `random_state=42`).
- Best estimator exported via `joblib.dump(clf, "models/best_random_forest_model.joblib")`.
- Consider tracking search spaces, best params, and test metrics alongside the model (e.g., JSON next to the artifact).

## Limitations and next steps

- Accuracy target (95%) not met. Current model favors recall over precision.
- Threshold tuning and probability calibration can adjust the precision/recall trade‑off.
- Potential gains from gradient‑boosted trees and richer feature engineering.