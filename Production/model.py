import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Train Random Forest model on HAR dataset')
    parser.add_argument('--data_path', type=str, 
                       default='/Users/zarakali/Desktop/COM774miniProject/data/hardata.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/zarakali/Desktop/COM774miniProject/outputs',
                       help='Directory to save model and metrics')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio (default: 0.2)')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees in Random Forest (default: 100)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at: {args.data_path}")
        
    # Load dataset
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print("Data loaded successfully")
    
    # Data preprocessing
    X = df.drop(['subject', 'Activity'], axis=1)
    y = df['Activity']
    print("Data preprocessing completed")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)
    print(f"Data split: training set size={len(X_train)}, test set size={len(X_test)}")
    
    # Train model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Model training completed")
    
    # Predictions
    print("Making predictions...")
    y_pred = rf_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save classification report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    # Confusion Matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add numbers to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 6))
    top_20 = feature_importance.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    
    # Save feature importance plot
    fi_path = os.path.join(args.output_dir, 'feature_importance.png')
    plt.savefig(fi_path)
    plt.close()
    
    # Save model
    model_path = os.path.join(args.output_dir, 'random_forest_har_model.joblib')
    joblib.dump(rf_model, model_path)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForest',
        'n_estimators': args.n_estimators,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = os.path.join(args.output_dir, 'model_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    
    print(f"\nOutputs saved in: {args.output_dir}")
    print(f"- Model: {model_path}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Classification Report: {report_path}")
    print(f"- Confusion Matrix: {cm_path}")
    print(f"- Feature Importance: {fi_path}")

if __name__ == '__main__':
    main()