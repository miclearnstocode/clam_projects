import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Load dataset
df = pd.read_csv("clam_dataset_202.csv")
print(f"Original dataset: {len(df)} samples")

# --- ENGINEER FEATURES ---
df['Blue_Green_Ratio'] = df['Mean_Blue'] / (df['Mean_Green'] + 1)
df['Green_Red_Ratio'] = df['Mean_Green'] / (df['Mean_Red'] + 1)
df['Blue_Minus_Green'] = df['Mean_Blue'] - df['Mean_Green']
df['Aspect_Circularity'] = df['Aspect_Ratio'] * df['Circularity']
df['Shape_Score'] = df['Aspect_Ratio'] / (df['Circularity'] + 0.01)
df['Total_Color_Variation'] = df['Std_Blue'] + df['Std_Green'] + df['Std_Red']
df['Color_Variation_Product'] = df['Std_Blue'] * df['Std_Green'] * df['Std_Red']
df['Mean_Green_Normalized'] = df['Mean_Green'] / (df['Mean_Blue'] + df['Mean_Green'] + df['Mean_Red'])

print(f"After feature engineering: {len(df.columns)} features")

# Check class balance
print(f"Alive (Label=1): {len(df[df['Label']==1])}")
print(f"Dead (Label=0): {len(df[df['Label']==0])}")
print(f"No Clam (Label=2): {len(df[df['Label']==2])}")

# Separate features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight={0: 1.5, 1: 1.5, 2: 2.5},  # Give 'No Clam' a higher weight
    oob_score=True
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dead', 'Alive', 'No Clam']))

# Save model
joblib.dump(model, 'clam_model_engineered.pkl')
print("\n✅ Model saved as 'clam_model_engineered.pkl'")

# Save feature names
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("✅ Feature names saved as 'feature_names.json'")