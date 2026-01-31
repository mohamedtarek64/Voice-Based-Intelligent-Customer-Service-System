"""
Decision Engine Module
======================
Handles intent-based decision making and response generation.
"""

import random
from typing import Dict, Any, Optional, List


# Response templates for each intent
RESPONSE_TEMPLATES = {
    'order_cancellation': [
        "I understand you want to cancel your order. Your cancellation request has been processed successfully. You should receive a refund within 5-7 business days.",
        "Your order has been cancelled. We're sorry to see you go! Is there anything else I can help you with?",
        "Order cancellation confirmed. If you were charged, the refund will be processed automatically.",
        "I've initiated the cancellation for your order. You'll receive a confirmation email shortly."
    ],
    
    'order_status': [
        "Let me check your order status. Your order is currently being processed and should ship within 1-2 business days.",
        "I can see your order is on its way! It should arrive within the next 3-5 business days.",
        "Your order has been shipped and is currently in transit. You can track it using the tracking number sent to your email.",
        "Good news! Your order is out for delivery today. Please ensure someone is available to receive it."
    ],
    
    'payment_issues': [
        "I'm sorry to hear about the payment issue. Our finance team has been notified and will investigate this within 24 hours. You'll receive an update via email.",
        "Payment failures can occur due to several reasons. Have you tried using a different payment method or checking with your bank?",
        "I see there was an issue with your payment. Let me help you resolve this. Can you try the payment again?",
        "We're looking into your payment issue. If you were charged incorrectly, a refund will be processed automatically."
    ],
    
    'product_information': [
        "I'd be happy to help you with product information! What specific details would you like to know?",
        "Our products are of the highest quality. What product are you interested in learning more about?",
        "I can provide detailed specifications, availability, and pricing information. What would you like to know?",
        "Let me check the product details for you. We have a wide range of options available."
    ],
    
    'complaint': [
        "I'm truly sorry to hear about your experience. Your feedback is important to us, and I've logged this complaint for immediate review by our quality team.",
        "We sincerely apologize for any inconvenience. A senior representative will contact you within 24 hours to resolve this issue.",
        "Thank you for bringing this to our attention. We take all complaints seriously and will work to make this right.",
        "I understand your frustration, and I apologize. Let me escalate this to ensure it gets resolved promptly."
    ],
    
    'return_exchange': [
        "I can help you with a return or exchange. Our policy allows returns within 30 days of purchase. I'll initiate the process for you.",
        "No problem! I'll set up a return for you. You'll receive a prepaid shipping label via email within the next hour.",
        "I understand you want to return or exchange your item. Let me guide you through the process.",
        "Returns are easy! I'll send you the return instructions and a shipping label right away."
    ],
    
    'account_issues': [
        "I can help you with your account. For security purposes, I'll send a password reset link to your registered email address.",
        "Let me help you access your account. Have you tried the 'Forgot Password' option on the login page?",
        "Account security is our priority. I'll help you regain access safely.",
        "I understand you're having trouble with your account. Let's get this resolved quickly."
    ],
    
    'general_inquiry': [
        "Thank you for reaching out! How can I assist you today?",
        "I'm here to help! What would you like to know?",
        "Our customer service is available 24/7. How may I be of assistance?",
        "I'd be happy to help with your inquiry. What information do you need?"
    ],
    
    'shipping_inquiry': [
        "We offer various shipping options including standard (5-7 days), express (2-3 days), and next-day delivery. Prices vary by location.",
        "Shipping is free for orders over $50! Otherwise, standard shipping is $5.99.",
        "Your shipping options and estimated delivery times will be displayed at checkout based on your location.",
        "We ship to most locations worldwide. Delivery times depend on your shipping address."
    ],
    
    'refund_status': [
        "I'll check your refund status. Refunds typically take 5-10 business days to appear in your account after processing.",
        "Your refund has been processed and should appear in your account shortly. If you don't see it within 10 business days, please contact us again.",
        "Let me look into your refund. I can see it was processed on our end. Please check with your bank if it hasn't arrived yet.",
        "Refund tracking is available! Your refund was initiated and is being processed by your payment provider."
    ]
}

# Clarification questions for medium confidence
CLARIFICATION_QUESTIONS = {
    'order_cancellation': "Did you want to cancel an existing order? Please confirm and I'll process the cancellation.",
    'order_status': "Are you looking for information about a recent order? Please provide your order number if you have it.",
    'payment_issues': "Are you experiencing a problem with a payment or refund? Please provide more details so I can assist you.",
    'product_information': "Which product would you like more information about?",
    'complaint': "I'm sorry to hear you had an issue. Could you please tell me more about what happened?",
    'return_exchange': "Would you like to return or exchange an item? Please let me know the details.",
    'account_issues': "Are you having trouble accessing your account? What specific issue are you facing?",
    'general_inquiry': "I'd like to help! Could you please provide more details about your question?",
    'shipping_inquiry': "Do you have a question about shipping options or delivery times?",
    'refund_status': "Are you checking on a pending refund? Please provide your order or refund reference number."
}


class DecisionEngine:
    """
    Decision engine for customer service responses.
    Makes decisions based on intent prediction confidence.
    """
    
    def __init__(self, high_threshold: float = 0.85, medium_threshold: float = 0.60):
        """
        Initialize the decision engine.
        
        Args:
            high_threshold: Confidence threshold for auto-response
            medium_threshold: Confidence threshold for clarification
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.response_templates = RESPONSE_TEMPLATES
        self.clarification_questions = CLARIFICATION_QUESTIONS
    
    def process_query(self, prediction: Dict[str, Any], 
                      original_query: str = "") -> Dict[str, Any]:
        """
        Process prediction and generate appropriate response.
        
        Args:
            prediction: Model prediction with intent and confidence
            original_query: Original customer query
            
        Returns:
            Dictionary with response and action
        """
        intent = prediction['predicted_intent']
        confidence = prediction['confidence']
        
        # Determine action based on confidence
        if confidence >= self.high_threshold:
            action = 'auto_respond'
            response = self._generate_response(intent)
            
        elif confidence >= self.medium_threshold:
            action = 'clarify'
            response = self._get_clarification(intent)
            
        else:
            action = 'escalate'
            response = self._get_escalation_message()
        
        return {
            'original_query': original_query,
            'intent': intent,
            'confidence': confidence,
            'confidence_level': self._get_confidence_level(confidence),
            'action': action,
            'response': response,
            'top_intents': prediction.get('top_intents', [])
        }
    
    def _generate_response(self, intent: str) -> str:
        """
        Generate response for high-confidence prediction.
        
        Args:
            intent: Predicted intent
            
        Returns:
            Response text
        """
        templates = self.response_templates.get(
            intent, 
            ["I understand your concern. Let me help you with that."]
        )
        return random.choice(templates)
    
    def _get_clarification(self, intent: str) -> str:
        """
        Get clarification question for medium-confidence prediction.
        
        Args:
            intent: Predicted intent
            
        Returns:
            Clarification question
        """
        return self.clarification_questions.get(
            intent,
            "I'm not entirely sure what you need. Could you please provide more details?"
        )
    
    def _get_escalation_message(self) -> str:
        """
        Get message for low-confidence escalation.
        
        Returns:
            Escalation message
        """
        messages = [
            "I want to make sure you get the best help possible. Let me connect you with a specialist who can assist you better.",
            "I'm having trouble understanding your request. A human agent will be with you shortly to provide personalized assistance.",
            "For this request, I'll connect you with our customer service team who can help you more effectively."
        ]
        return random.choice(messages)
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Get human-readable confidence level.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Confidence level string
        """
        if confidence >= self.high_threshold:
            return 'high'
        elif confidence >= self.medium_threshold:
            return 'medium'
        else:
            return 'low'
    
    def add_response_template(self, intent: str, templates: List[str]):
        """
        Add or update response templates for an intent.
        
        Args:
            intent: Intent name
            templates: List of response templates
        """
        self.response_templates[intent] = templates
    
    def add_clarification(self, intent: str, question: str):
        """
        Add or update clarification question for an intent.
        
        Args:
            intent: Intent name
            question: Clarification question
        """
        self.clarification_questions[intent] = question
    
    def get_supported_intents(self) -> List[str]:
        """
        Get list of intents with response templates.
        
        Returns:
            List of intent names
        """
        return list(self.response_templates.keys())


def process_customer_query(query_text: str, model, vectorizer, 
                           preprocessor=None) -> Dict[str, Any]:
    """
    Process a customer query through the full pipeline.
    
    Args:
        query_text: Raw customer query
        model: Trained classifier model
        vectorizer: Fitted TF-IDF vectorizer
        preprocessor: Optional preprocessing function
        
    Returns:
        Dictionary with complete response
    """
    # Preprocess if function provided
    if preprocessor:
        processed_query = preprocessor(query_text)
    else:
        processed_query = query_text.lower().strip()
    
    # Vectorize
    query_vector = vectorizer.transform([processed_query])
    
    # Predict
    predicted_intent = model.predict(query_vector)[0]
    confidence_scores = model.predict_proba(query_vector)[0]
    max_confidence = float(confidence_scores.max())
    
    # Get top intents
    top_indices = confidence_scores.argsort()[-3:][::-1]
    top_intents = [
        {
            'intent': model.classes_[i],
            'confidence': float(confidence_scores[i])
        }
        for i in top_indices
    ]
    
    prediction = {
        'predicted_intent': predicted_intent,
        'confidence': max_confidence,
        'top_intents': top_intents
    }
    
    # Use decision engine
    engine = DecisionEngine()
    result = engine.process_query(prediction, query_text)
    
    return result


if __name__ == "__main__":
    # Test decision engine
    engine = DecisionEngine()
    
    test_predictions = [
        {'predicted_intent': 'order_cancellation', 'confidence': 0.92},
        {'predicted_intent': 'order_status', 'confidence': 0.75},
        {'predicted_intent': 'general_inquiry', 'confidence': 0.45},
    ]
    
    print("Testing Decision Engine:")
    print("=" * 60)
    
    for pred in test_predictions:
        result = engine.process_query(pred, "Test query")
        print(f"\nIntent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Level: {result['confidence_level']}")
        print(f"Action: {result['action']}")
        print(f"Response: {result['response']}")
        print("-" * 60)
