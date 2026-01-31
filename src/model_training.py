"""
Model Training Module
=====================
Handles training of the intent classification model using TF-IDF and various classifiers.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from typing import Tuple, Dict, Any, Optional

from data_preprocessing import load_and_preprocess_data, get_data_statistics


class IntentClassifier:
    """
    Intent Classification Model using TF-IDF and various ML classifiers.
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of classifier ('logistic_regression', 'svm', 'naive_bayes', 'random_forest')
        """
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.intent_labels = None
        self.is_trained = False
        
    def _get_model(self, model_type: str):
        """Get the appropriate model based on type."""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                C=1.0, 
                random_state=42,
                solver='lbfgs'
            ),
            'svm': SVC(
                kernel='linear', 
                C=1.0, 
                probability=True, 
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
        
        return models[model_type]
    
    def train(self, X_train: pd.Series, y_train: pd.Series, 
              max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 3)) -> Dict[str, Any]:
        """
        Train the intent classifier.
        
        Args:
            X_train: Training queries (preprocessed)
            y_train: Training labels (intents)
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            
        Returns:
            Dictionary with training metrics
        """
        print(f"Training {self.model_type} classifier...")
        print(f"Training samples: {len(X_train)}")
        
        # Store intent labels
        self.intent_labels = list(y_train.unique())
        print(f"Number of intents: {len(self.intent_labels)}")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
        
        # Initialize and train model
        self.model = self._get_model(self.model_type)
        self.model.fit(X_train_tfidf, y_train)
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train_tfidf)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_tfidf, y_train, cv=5)
        
        self.is_trained = True
        
        metrics = {
            'model_type': self.model_type,
            'train_accuracy': train_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'num_features': X_train_tfidf.shape[1],
            'num_intents': len(self.intent_labels)
        }
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return metrics
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent for a single text query.
        
        Args:
            text: Preprocessed query text
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([text])
        
        # Predict
        predicted_intent = self.model.predict(text_tfidf)[0]
        
        # Get confidence scores
        confidence_scores = self.model.predict_proba(text_tfidf)[0]
        max_confidence = float(confidence_scores.max())
        
        # Get top 3 intents
        top_indices = confidence_scores.argsort()[-3:][::-1]
        top_intents = [
            {
                'intent': self.model.classes_[i],
                'confidence': float(confidence_scores[i])
            }
            for i in top_indices
        ]
        
        return {
            'predicted_intent': predicted_intent,
            'confidence': max_confidence,
            'top_intents': top_intents
        }
    
    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test queries (preprocessed)
            y_test: Test labels (intents)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Predict
        y_pred = self.model.predict(X_test_tfidf)
        y_proba = self.model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confidence distribution
        max_confidences = y_proba.max(axis=1)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'avg_confidence': float(max_confidences.mean()),
            'predictions': y_pred.tolist(),
            'confidences': max_confidences.tolist()
        }
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def save(self, model_dir: str = 'models'):
        """
        Save the trained model and vectorizer.
        
        Args:
            model_dir: Directory to save models
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'intent_classifier.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'intent_labels': self.intent_labels
        }
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"Metadata saved to: {metadata_path}")
    
    def load(self, model_dir: str = 'models'):
        """
        Load a trained model and vectorizer.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = os.path.join(model_dir, 'intent_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        
        # Check files exist
        for path in [model_path, vectorizer_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load model
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load vectorizer
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded from: {vectorizer_path}")
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        self.model_type = metadata['model_type']
        self.intent_labels = metadata['intent_labels']
        print(f"Metadata loaded from: {metadata_path}")
        
        self.is_trained = True
        print(f"Model type: {self.model_type}")
        print(f"Intent labels: {self.intent_labels}")


def train_and_save_model(data_path: str, model_dir: str = 'models', 
                         model_type: str = 'logistic_regression',
                         test_size: float = 0.2) -> Tuple[IntentClassifier, Dict[str, Any]]:
    """
    Complete training pipeline: load data, train model, evaluate, and save.
    
    Args:
        data_path: Path to training data CSV
        model_dir: Directory to save models
        model_type: Type of classifier
        test_size: Fraction of data for testing
        
    Returns:
        Tuple of (trained classifier, evaluation metrics)
    """
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # Get data statistics
    stats = get_data_statistics(df)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of intents: {stats['num_intents']}")
    print(f"  Average query length: {stats['avg_query_length']:.1f} words")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_query'],
        df['intent'],
        test_size=test_size,
        random_state=42,
        stratify=df['intent']
    )
    
    print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)}")
    
    # Create and train classifier
    classifier = IntentClassifier(model_type=model_type)
    train_metrics = classifier.train(X_train, y_train)
    
    # Evaluate
    eval_metrics = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save(model_dir)
    
    return classifier, {**train_metrics, **eval_metrics}


if __name__ == "__main__":
    # Test with sample data
    import sys
    
    data_path = 'data/processed/customer_queries.csv'
    
    if os.path.exists(data_path):
        classifier, metrics = train_and_save_model(
            data_path=data_path,
            model_dir='models',
            model_type='svm'
        )
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    else:
        print(f"Training data not found at: {data_path}")
        print("Please create or generate the dataset first.")
        print("Use: python generate_dataset.py")
