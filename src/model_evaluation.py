"""
Model Evaluation Module
=======================
Comprehensive evaluation of the intent classification model with visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, List, Optional
import joblib


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          labels: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_confidence_distribution(confidences: np.ndarray, 
                                  threshold: float = 0.85,
                                  save_path: Optional[str] = None):
    """
    Plot histogram of confidence scores.
    
    Args:
        confidences: Array of confidence scores
        threshold: Threshold line to draw
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = plt.hist(
        confidences, 
        bins=20, 
        edgecolor='black',
        alpha=0.7,
        color='steelblue'
    )
    
    # Color bars based on threshold
    for i, patch in enumerate(patches):
        if bins[i] >= threshold:
            patch.set_facecolor('green')
        elif bins[i] >= 0.60:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')
    
    # Add threshold line
    plt.axvline(x=threshold, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'High Confidence ({threshold})')
    plt.axvline(x=0.60, color='darkorange', linestyle='--', linewidth=2,
                label='Medium Confidence (0.60)')
    
    plt.title('Confidence Score Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(confidences):.3f}\nStd: {np.std(confidences):.3f}'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution saved to: {save_path}")
    
    plt.show()


def plot_intent_distribution(df: pd.DataFrame, intent_column: str = 'intent',
                              save_path: Optional[str] = None):
    """
    Plot bar chart of intent distribution.
    
    Args:
        df: DataFrame with intent column
        intent_column: Name of intent column
        save_path: Path to save the plot
    """
    intent_counts = df[intent_column].value_counts()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(intent_counts)),
        intent_counts.values,
        color='steelblue',
        edgecolor='black'
    )
    
    plt.xticks(range(len(intent_counts)), intent_counts.index, rotation=45, ha='right')
    plt.title('Intent Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Intent Category', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, intent_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Intent distribution saved to: {save_path}")
    
    plt.show()


def plot_metrics_per_class(report: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot precision, recall, F1-score per class.
    
    Args:
        report: Classification report dictionary
        save_path: Path to save the plot
    """
    # Extract per-class metrics
    classes = []
    precision = []
    recall = []
    f1_score = []
    
    for key, value in report.items():
        if isinstance(value, dict) and 'precision' in value:
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(key)
                precision.append(value['precision'])
                recall.append(value['recall'])
                f1_score.append(value['f1-score'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Intent Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, and F1-Score per Intent', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics per class saved to: {save_path}")
    
    plt.show()


def evaluate_model_comprehensive(model_path: str, vectorizer_path: str,
                                   test_data_path: str, output_dir: str = 'docs'):
    """
    Comprehensive model evaluation with all visualizations.
    
    Args:
        model_path: Path to saved model
        vectorizer_path: Path to saved vectorizer
        test_data_path: Path to test data CSV
        output_dir: Directory to save plots
    """
    from data_preprocessing import preprocess_text
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv(test_data_path)
    df['processed_query'] = df['query'].apply(preprocess_text)
    
    # Transform and predict
    X_test = vectorizer.transform(df['processed_query'])
    y_true = df['intent']
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidences = y_proba.max(axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nWeighted Averages:")
    print(f"  Precision: {report['weighted avg']['precision']:.4f}")
    print(f"  Recall: {report['weighted avg']['recall']:.4f}")
    print(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean Confidence: {np.mean(confidences):.4f}")
    print(f"  Std Confidence: {np.std(confidences):.4f}")
    print(f"  High Confidence (>=85%): {(confidences >= 0.85).sum()} ({(confidences >= 0.85).mean()*100:.1f}%)")
    print(f"  Medium Confidence (60-85%): {((confidences >= 0.60) & (confidences < 0.85)).sum()}")
    print(f"  Low Confidence (<60%): {(confidences < 0.60).sum()}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    labels = sorted(y_true.unique())
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_true, y_pred, labels,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Confidence Distribution
    plot_confidence_distribution(
        confidences,
        save_path=os.path.join(output_dir, 'confidence_distribution.png')
    )
    
    # Metrics per Class
    plot_metrics_per_class(
        report,
        save_path=os.path.join(output_dir, 'metrics_per_class.png')
    )
    
    # Save evaluation report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred))
        f.write(f"\n\nConfidence Statistics:\n")
        f.write(f"Mean: {np.mean(confidences):.4f}\n")
        f.write(f"Std: {np.std(confidences):.4f}\n")
        f.write(f"High Confidence Rate: {(confidences >= 0.85).mean()*100:.1f}%\n")
    
    print(f"Evaluation report saved to: {report_path}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confidences': confidences
    }


if __name__ == "__main__":
    # Run evaluation if models exist
    model_path = 'models/intent_classifier.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    test_data_path = 'data/processed/customer_queries.csv'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        if os.path.exists(test_data_path):
            evaluate_model_comprehensive(
                model_path=model_path,
                vectorizer_path=vectorizer_path,
                test_data_path=test_data_path,
                output_dir='docs'
            )
        else:
            print(f"Test data not found: {test_data_path}")
    else:
        print("Model files not found. Train the model first.")
