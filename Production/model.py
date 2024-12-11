import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with the correct path
dataset_path = '/Users/zarakali/Desktop/COM774miniProject/data/hardata.csv'
df = pd.read_csv(dataset_path)

# Print basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nSample of the data:")
print(df.head())

# 1. Data Preprocessing
# Separate features (X) and target variable (y)
X = df.drop(['subject', 'Activity'], axis=1)
y = df['Activity']

# Print unique activities
print("\nUnique activities in the dataset:", y.unique())

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# 3. Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Make predictions
y_pred = rf_model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Create confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 7. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot top 20 most important features
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# 8. Save model performance metrics
output_dir = '/Users/zarakali/Desktop/COM774miniProject/outputs'
# Create directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

model_metrics = {
    'accuracy': accuracy,
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Save metrics
metrics_path = os.path.join(output_dir, 'model_metrics.csv')
metrics_df = pd.DataFrame([model_metrics])
metrics_df.to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path))

# 9. Save the model
model_path = os.path.join(output_dir, 'random_forest_har_model.joblib')
import joblib
joblib.dump(rf_model, model_path)

print(f"\nModel and metrics saved in: {output_dir}")