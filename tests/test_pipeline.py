"""
Test Pipeline
=============
Tests for the Voice Customer Service System.
"""

import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


class TestDataPreprocessing(unittest.TestCase):
    """Tests for data preprocessing module."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        from data_preprocessing import clean_text
        
        # Test lowercase
        self.assertEqual(clean_text("HELLO WORLD"), "hello world")
        
        # Test special characters removal
        self.assertEqual(clean_text("Hello! How are you?"), "hello how are you")
        
        # Test URL removal
        result = clean_text("Visit https://example.com for more info")
        self.assertNotIn("https", result)
        
        # Test email removal
        result = clean_text("Contact test@example.com for help")
        self.assertNotIn("@", result)
    
    def test_preprocess_text(self):
        """Test text preprocessing function."""
        from data_preprocessing import preprocess_text, download_nltk_data
        
        download_nltk_data()
        
        # Test basic preprocessing
        result = preprocess_text("I want to cancel my order!!!")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Test lemmatization
        result = preprocess_text("Running and jumped")
        # Should contain lemmatized forms
        self.assertIsInstance(result, str)


class TestDecisionEngine(unittest.TestCase):
    """Tests for decision engine module."""
    
    def test_high_confidence_auto_respond(self):
        """Test high confidence leads to auto_respond."""
        from decision_engine import DecisionEngine
        
        engine = DecisionEngine()
        prediction = {
            'predicted_intent': 'order_cancellation',
            'confidence': 0.92
        }
        
        result = engine.process_query(prediction, "Cancel my order")
        self.assertEqual(result['action'], 'auto_respond')
        self.assertIn('response', result)
    
    def test_medium_confidence_clarify(self):
        """Test medium confidence leads to clarify."""
        from decision_engine import DecisionEngine
        
        engine = DecisionEngine()
        prediction = {
            'predicted_intent': 'order_status',
            'confidence': 0.70
        }
        
        result = engine.process_query(prediction, "Order")
        self.assertEqual(result['action'], 'clarify')
    
    def test_low_confidence_escalate(self):
        """Test low confidence leads to escalate."""
        from decision_engine import DecisionEngine
        
        engine = DecisionEngine()
        prediction = {
            'predicted_intent': 'general_inquiry',
            'confidence': 0.40
        }
        
        result = engine.process_query(prediction, "blah blah")
        self.assertEqual(result['action'], 'escalate')
    
    def test_response_templates(self):
        """Test that response templates exist for intents."""
        from decision_engine import DecisionEngine
        
        engine = DecisionEngine()
        intents = engine.get_supported_intents()
        
        self.assertGreater(len(intents), 0)
        self.assertIn('order_cancellation', intents)
        self.assertIn('order_status', intents)


class TestDatasetGenerator(unittest.TestCase):
    """Tests for dataset generator."""
    
    def test_intent_data_exists(self):
        """Test that intent data is defined."""
        from generate_dataset import INTENT_DATA
        
        self.assertGreater(len(INTENT_DATA), 0)
        
        for intent, queries in INTENT_DATA.items():
            self.assertGreater(len(queries), 0)
            for query in queries:
                self.assertIsInstance(query, str)
    
    def test_generate_variations(self):
        """Test query variation generation."""
        from generate_dataset import generate_variations
        
        queries = ["Hello", "Hi there"]
        variations = generate_variations(queries, num_variations=2)
        
        self.assertGreater(len(variations), len(queries))


class TestTextToSpeech(unittest.TestCase):
    """Tests for TTS module."""
    
    def test_tts_initialization(self):
        """Test TTS can be initialized."""
        from text_to_speech import TextToSpeech
        
        tts = TextToSpeech(language='en')
        self.assertIsNotNone(tts)


class TestSpeechToText(unittest.TestCase):
    """Tests for STT module."""
    
    def test_stt_initialization(self):
        """Test STT can be initialized (may fail if Whisper not installed)."""
        try:
            from speech_to_text import SpeechToText
            stt = SpeechToText(model_size='tiny')
            # If Whisper is available
            self.assertIsNotNone(stt)
        except ImportError:
            # Whisper not installed, skip
            self.skipTest("Whisper not installed")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Check if model exists
        cls.model_exists = (
            os.path.exists('models/intent_classifier.pkl') and
            os.path.exists('models/vectorizer.pkl')
        )
    
    def test_full_text_pipeline(self):
        """Test full text processing pipeline."""
        if not self.model_exists:
            self.skipTest("Model not trained yet")
        
        import joblib
        from data_preprocessing import preprocess_text
        from decision_engine import DecisionEngine
        
        # Load model
        model = joblib.load('models/intent_classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        engine = DecisionEngine()
        
        # Test query
        query = "I want to cancel my order"
        processed = preprocess_text(query)
        
        # Predict
        vector = vectorizer.transform([processed])
        intent = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()
        
        # Get response
        prediction = {
            'predicted_intent': intent,
            'confidence': confidence
        }
        result = engine.process_query(prediction, query)
        
        self.assertIn('response', result)
        self.assertIn('action', result)
        self.assertIn('intent', result)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestTextToSpeech))
    suite.addTests(loader.loadTestsFromTestCase(TestSpeechToText))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
