import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("clam_dataset.csv")

# --- FEATURE ENGINEERING (Keep only proven features) ---
print("Original features:", len(df.columns) - 1)

# Color ratios
df['Blue_Green_Ratio'] = df['Mean_Blue'] / (df['Mean_Green'] + 1)
df['Green_Red_Ratio'] = df['Mean_Green'] / (df['Mean_Red'] + 1)

# Color differences
df['Blue_Minus_Green'] = df['Mean_Blue'] - df['Mean_Green']

# Shape combinations
df['Aspect_Circularity'] = df['Aspect_Ratio'] * df['Circularity']
df['Shape_Score'] = df['Aspect_Ratio'] / (df['Circularity'] + 0.01)

# Variation features
df['Total_Color_Variation'] = df['Std_Blue'] + df['Std_Green'] + df['Std_Red']
df['Color_Variation_Product'] = df['Std_Blue'] * df['Std_Green'] * df['Std_Red']

# Normalized features
total_color = df['Mean_Blue'] + df['Mean_Green'] + df['Mean_Red']
df['Mean_Green_Normalized'] = df['Mean_Green'] / total_color

print("New features added:", len(df.columns) - 1 - 9)
print("Total features:", len(df.columns) - 1)

# Remove any NaN or infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Separate features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- HYPERPARAMETER TUNING ---
print("\n🔍 Finding optimal hyperparameters...")

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [4, 5, 6],
    'min_samples_split': [5, 8, 10],
    'min_samples_leaf': [3, 4, 5],
    'class_weight': ['balanced', {0: 1.0, 1: 1.5}]
}

rf = RandomForestClassifier(random_state=42, oob_score=True)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_ * 100:.2f}%")

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# OOB score
print(f"\nOut-of-Bag Score: {best_model.oob_score_ * 100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the best model
joblib.dump(best_model, 'clam_model_final.pkl')
print("\n✅ Final model saved as 'clam_model_final.pkl'")

# Save feature names for later use
import json
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("✅ Feature names saved as 'feature_names.json'")