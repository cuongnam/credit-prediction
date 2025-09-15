# XGBoost - Give Me Some Credit (Credit Scoring)
# Full pipeline: load data, preprocess, train, evaluate, optimize threshold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import platform

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import xgboost as xgb
import joblib

# -----------------------
# 1) Multiprocessing fix
# -----------------------
if platform.system() != "Windows":  # Windows không hỗ trợ "fork"
    multiprocessing.set_start_method("fork", force=True)

# -----------------------
# 2) Load dataset
# -----------------------
df = pd.read_csv("cs-training.csv")
print("Data shape:", df.shape)
print(df.head())

# -----------------------
# 3) Preprocessing
# -----------------------
# Drop ID column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Handle missing values (simple: fill with median)
df = df.fillna(df.median(numeric_only=True))

# -----------------------
# 4) Split train/test
# -----------------------
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# -----------------------
# 5) Train XGBoost model
# -----------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -----------------------
# 6) Save model
# -----------------------
joblib.dump(model, "xgb_final_model.joblib")
print("Model saved: xgb_final_model.joblib")

# -----------------------
# 7) Load model (check)
# -----------------------
model = joblib.load("xgb_final_model.joblib")

# -----------------------
# 8) Predictions
# -----------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# -----------------------
# 9) Evaluation
# -----------------------
print("\n[Train] Classification report:\n", classification_report(y_train, y_train_pred))
print("\n[Test] Classification report:\n", classification_report(y_test, y_test_pred))

# -----------------------
# 10) ROC-AUC
# -----------------------
auc = roc_auc_score(y_test, y_test_proba)
print("ROC-AUC:", auc)

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -----------------------
# 11) Confusion matrix
# -----------------------
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------
# 11b) Optimize threshold for precision
# -----------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)

valid_idx = np.where(recalls[:-1] >= 0.5)[0]  # điều kiện recall ≥ 0.5

if len(valid_idx) > 0:
    best_threshold = thresholds[valid_idx][np.argmax(precisions[valid_idx])]
    best_precision = precisions[valid_idx].max()
    best_recall = recalls[valid_idx][np.argmax(precisions[valid_idx])]

    print(f"\n[Threshold Optimization]")
    print(f"Ngưỡng tối ưu: {best_threshold:.3f}")
    print(f"Precision tại ngưỡng này: {best_precision:.3f}")
    print(f"Recall tại ngưỡng này: {best_recall:.3f}")

    # Dự đoán lại với threshold tối ưu
    y_test_pred_opt = (y_test_proba >= best_threshold).astype(int)
    print("\nClassification report (threshold optimized):")
    print(classification_report(y_test, y_test_pred_opt))
else:
    print("Không tìm thấy threshold nào thỏa recall >= 0.5")

# -----------------------
# 12) Feature importance
# -----------------------
xgb.plot_importance(model, importance_type="weight")
plt.title("Feature Importance (by weight)")
plt.show()

xgb.plot_importance(model, importance_type="gain")
plt.title("Feature Importance (by gain)")
plt.show()
