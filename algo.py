import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("clam_dataset_nose.csv")
print(f"Original dataset: {len(df)} samples")

# Check class balance
print(f"Healthy (Label=1): {len(df[df['Label']==1])}")
print(f"At Risk (Label=0): {len(df[df['Label']==0])}")

# Separate features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=300,     
    max_depth=15,        
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight={0: 0.5, 1: 10.0}, 
    oob_score=True
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['At Risk', 'Healthy']))

# Calculate AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1 (Healthy)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.3f}")

# AUC interpretation
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

# Calculate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("              At Risk  Healthy")
print(f"Actual At Risk  {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"Actual Healthy  {cm[1][0]:5d}  {cm[1][1]:5d}")

# Detailed confusion matrix interpretation
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix Details:")
print(f"True Negatives (At Risk correctly classified):     {tn}")
print(f"False Positives (At Risk misclassified as Healthy): {fp}")
print(f"False Negatives (Healthy misclassified as At Risk): {fn}")
print(f"True Positives (Healthy correctly classified):     {tp}")

# Calculate additional metrics from confusion matrix
total = tn + fp + fn + tp
accuracy_from_cm = (tn + tp) / total
sensitivity = tp / (tp + fn)  # Recall for Healthy
specificity = tn / (tn + fp)  # Recall for At Risk
precision_healthy = tp / (tp + fp)
precision_at_risk = tn / (tn + fn)

print("\nMetrics derived from Confusion Matrix:")
print(f"Overall Accuracy: {accuracy_from_cm:.3f}")
print(f"Sensitivity (Recall for Healthy): {sensitivity:.3f}")
print(f"Specificity (Recall for At Risk): {specificity:.3f}")
print(f"Precision for Healthy: {precision_healthy:.3f}")
print(f"Precision for At Risk: {precision_at_risk:.3f}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['At Risk', 'Healthy'], 
            yticklabels=['At Risk', 'Healthy'])
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Condition')
plt.ylabel('Actual Condition')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")

# ROC Curve visualization
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ ROC curve saved as 'roc_curve.png'")

# Print feature importance
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
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'precision_healthy': float(precision_healthy),
    'precision_at_risk': float(precision_at_risk),
    'n_estimators': 300,
    'max_depth': 15,
    'class_weight': {0: 0.5, 1: 10.0}
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Model metrics saved as 'model_metrics.json'")


print("\n" + "="*50)
print("SUMMARY OF MODEL PERFORMANCE")
print("="*50)
print(f"Confusion Matrix (Test Set, n={len(y_test)}):")
print("                 Predicted")
print("              At Risk  Healthy")
print(f"Actual At Risk  {tn:5d}  {fp:5d}")
print(f"Actual Healthy  {fn:5d}  {tp:5d}")
print(f"\nAUC-ROC Score: {auc_score:.3f} ({auc_interpretation})")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Sensitivity (Recall for Healthy): {sensitivity:.3f}")
print(f"Specificity (Recall for At Risk): {specificity:.3f}")
print("="*50)