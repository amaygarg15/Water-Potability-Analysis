# Water Potability Analysis (SMOTE-Based)

This project predicts whether water is safe to drink using water quality features and a supervised machine learning pipeline.
The focus of this repository is one clean, reproducible workflow: the SMOTE-based analysis in the Water Quality Dataset folder.

## Why This Project

In real-world water datasets, class imbalance is common. If a model mostly sees one class, it can report high accuracy but still miss the minority class when it matters most.
To handle that, this project uses SMOTE (Synthetic Minority Oversampling Technique) and evaluates models with metrics that go beyond plain accuracy.

## What Is Included

- A complete notebook pipeline for data cleaning, EDA, preprocessing, model benchmarking, SMOTE balancing, and evaluation.
- Processed output files used for reproducibility.
- Final comparison of multiple classifiers, with emphasis on minority-class performance.

## Project Structure

```text
.
├── README.md
└── Water Quality Dataset/
	├── water_quality_analysis.ipynb
	└── processed_outputs/
		├── water_quality_cleaned.csv
		├── X_train_scaled.csv
		├── X_test_scaled.csv
		├── y_train.csv
		└── y_test.csv
```

## Workflow Summary

1. Load dataset (local CSV or optional Kaggle download).
2. Clean and prepare data (fix invalid values, handle missing values, remove duplicates).
3. Scale features using StandardScaler.
4. Split data into train and test sets.
5. Benchmark multiple models on the original split.
6. Apply SMOTE on training data.
7. Re-train and evaluate models using Accuracy, Weighted F1, and Minority Class Recall.
8. Review confusion matrix and sample predictions.

## Models Used

- Logistic Regression
- K-Nearest Neighbors (baseline phase)
- Random Forest
- Extra Trees
- Gradient Boosting

## Key Results

### Baseline (before SMOTE)

- Best model: GradientBoosting
- Accuracy: 0.9587
- Train accuracy: 0.9712
- Test accuracy: 0.9587

### SMOTE Benchmark

| Model | Overall Accuracy | Weighted F1 | Minority Class (1) Recall |
|---|---:|---:|---:|
| GradientBoosting (SMOTE) | 0.9481 | 0.9497 | 0.8462 |
| RandomForest (SMOTE) | 0.9563 | 0.9568 | 0.8352 |
| ExtraTrees (SMOTE) | 0.9294 | 0.9310 | 0.7473 |
| LogisticRegression (SMOTE) | 0.8031 | 0.8312 | 0.7088 |

### Interpretation

- RandomForest (SMOTE) gives the strongest overall Accuracy and Weighted F1.
- GradientBoosting (SMOTE) gives the best minority recall.
- This shows the practical trade-off between overall correctness and minority-class sensitivity.

## How To Run

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn kagglehub jupyter
```

3. Open and run:

```text
Water Quality Dataset/water_quality_analysis.ipynb
```

4. Generated files will be saved in:

```text
Water Quality Dataset/processed_outputs/
```

## Notes

- Kaggle download is optional. If unavailable, keep the CSV locally and the notebook will use local file fallback.
- The repository intentionally keeps one final project workflow to stay focused and easy to review.

## Next Improvements

- Add cross-validation and hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
- Add ROC-AUC and Precision-Recall curves.
- Package inference into a small API or CLI for reusable predictions.