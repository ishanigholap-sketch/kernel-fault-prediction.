
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.pipeline import Pipeline

# Dataset URL: UCI AI4I Predictive Maintenance
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00601/ai4i2020.csv"
)

print("Loading dataset...")
df = pd.read_csv(DATA_URL)

# Features used for machine-failure classification
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

X = df[FEATURES]
y = df[TARGET]

print(f"Dataset shape: {df.shape}")
print(f"Total observations: {len(df)}")
print("\nClass distribution:")
print(y.value_counts())

# Keep the test set separate from training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Compare linear and RBF kernel SVMs
models = {
    "Linear SVM": SVC(kernel="linear", class_weight="balanced"),
    "RBF SVM": SVC(kernel="rbf", class_weight="balanced"),
}

results = {}

for name, classifier in models.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", classifier),
    ])

    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results[name] = {
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
    }

    print(f"Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall:    {results[name]['recall']:.4f}")
    print(f"F1 score:  {results[name]['f1']:.4f}")

# Select the model with the highest failure-class F1 score
best_name = max(results, key=lambda name: results[name]["f1"])
best = results[best_name]

print(f"\nBest model by F1 score: {best_name}")
print("\nClassification report:")
print(classification_report(
    y_test,
    best["predictions"],
    labels=[0, 1],
    target_names=["Normal", "Failure"],
    zero_division=0,
))

# Display the confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    best["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"],
)

plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Type I and Type II errors for failure as the positive class
tn, fp, fn, tp = best["confusion_matrix"].ravel()

print(f"\nType I error (false positives): {fp}")
print(f"Type II error (false negatives): {fn}")
print("\nModel training and evaluation complete!")
