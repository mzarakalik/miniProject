import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def parse_args():
   parser = argparse.ArgumentParser(description='Train Random Forest model on HAR dataset')
   parser.add_argument('--data_path', type=str, default='/Users/zarakali/Desktop/COM774miniProject/data/hardata.csv',
                     help='Path to dataset CSV file')
   parser.add_argument('--output_dir', type=str, default='/Users/zarakali/Desktop/COM774miniProject/outputs',
                     help='Directory to save model and metrics')
   parser.add_argument('--test_size', type=float, default=0.2,
                     help='Test set size ratio (default: 0.2)')
   parser.add_argument('--n_estimators', type=int, default=100,
                     help='Number of trees in Random Forest (default: 100)')
   return parser.parse_args()

def main():
   args = parse_args()
   
   # Load dataset
   df = pd.read_csv(args.data_path)
   
   # Data preprocessing
   X = df.drop(['subject', 'Activity'], axis=1)
   y = df['Activity']
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=args.test_size, random_state=42)
   
   # Train model
   rf_model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
   rf_model.fit(X_train, y_train)
   
   # Predictions
   y_pred = rf_model.predict(X_test)
   
   # Evaluate
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy:.4f}")
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))
   
   # Visualizations
   plt.figure(figsize=(10, 8))
   cm = confusion_matrix(y_test, y_pred)
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.title('Confusion Matrix')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.show()
   
   # Feature importance
   feature_importance = pd.DataFrame({
       'feature': X.columns,
       'importance': rf_model.feature_importances_
   }).sort_values('importance', ascending=False)
   
   plt.figure(figsize=(12, 6))
   sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
   plt.title('Top 20 Most Important Features')
   plt.xlabel('Feature Importance')
   plt.tight_layout()
   plt.show()
   
   # Save outputs
   os.makedirs(args.output_dir, exist_ok=True)
   
   model_metrics = {
       'accuracy': accuracy,
       'model_type': 'RandomForest',
       'n_estimators': args.n_estimators,
       'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
   }
   
   metrics_path = os.path.join(args.output_dir, 'model_metrics.csv')
   metrics_df = pd.DataFrame([model_metrics])
   metrics_df.to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path))
   
   model_path = os.path.join(args.output_dir, 'random_forest_har_model.joblib')
   joblib.dump(rf_model, model_path)
   
   print(f"\nModel and metrics saved in: {args.output_dir}")

if __name__ == '__main__':
   main()