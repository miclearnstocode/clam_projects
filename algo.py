import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import joblib
import json
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("clam_dataset_nose.csv")
print(f"Original dataset: {len(df)} samples")

# Check class balance
print(f"Alive (Label=1): {len(df[df['Label']==1])}")
print(f"Dead (Label=0): {len(df[df['Label']==0])}")

# Separate features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
# IMPORTANT: Increase Alive weight to 10.0! This forces the model to REALLY look for pink noses.
model = RandomForestClassifier(
    n_estimators=300,          # More trees = more stable
    max_depth=15,              # Deeper trees to learn complex pink patterns
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight={0: 0.5, 1: 10.0},  # CRITICAL: Alive is 10x more important than Dead
    oob_score=True
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))

# Calculate AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1 (Alive)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.3f}")

#AUC interpretation
if auc_score >= 0.90:
    auc_interpretation = "Excellent"
elif auc_score >= 0.80:
    auc_interpretation = "Good"
elif auc_score >= 0.70:
    auc_interpretation = "Fair"
elif auc_score >= 0.60:
    auc_interpretation = "Poor"
else:
    auc_interpretation = "Failed"

print(f"AUC Interpretation: {auc_interpretation}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop 5 Most Important Features:")
for i in range(5):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.3f}")

# Save model
joblib.dump(model, 'clam_model_engineered.pkl')
print("\n✅ Model saved as 'clam_model_engineered.pkl'")

# Save feature names
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("✅ Feature names saved as 'feature_names.json'")

# Save AUC and other metrics
metrics = {
    'accuracy': accuracy,
    'auc_score': auc_score,
    'auc_interpretation': auc_interpretation,
    'n_estimators': 300,
    'max_depth': 15,
    'class_weight': {0: 0.5, 1: 10.0}
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Model metrics saved as 'model_metrics.json'")