"""
Dataset Generator
=================
Generates synthetic customer service dataset for training.
"""

import csv
import random
import os


# Intent categories with sample queries
INTENT_DATA = {
    'order_cancellation': [
        "I want to cancel my order",
        "Cancel my order please",
        "Please cancel my recent order",
        "Can I cancel my order?",
        "I need to cancel an order",
        "How do I cancel my order?",
        "I'd like to cancel my purchase",
        "Cancel order",
        "I changed my mind, cancel my order",
        "Please process a cancellation",
        "I want to cancel the order I just placed",
        "Can you cancel my order for me?",
        "I need this order cancelled",
        "Stop my order",
        "Don't ship my order, cancel it",
        "Cancellation request for my order",
        "I no longer want this order",
        "Please void my order",
        "I want to revoke my order",
        "Cancel the items in my order",
        "I'd like to cancel before it ships",
        "Can I still cancel my order?",
        "Is it too late to cancel?",
        "I need to cancel ASAP",
        "Cancel my order immediately",
        "I accidentally ordered, please cancel",
        "Wrong order, please cancel",
        "Cancel my latest purchase",
        "I want my order cancelled today",
        "Please help me cancel my order"
    ],
    
    'order_status': [
        "Where is my order?",
        "Track my order",
        "What's the status of my order?",
        "When will my order arrive?",
        "Has my order shipped?",
        "I want to track my package",
        "Check my order status",
        "Is my order on the way?",
        "When can I expect my delivery?",
        "Order tracking",
        "Where is my package?",
        "Did my order ship yet?",
        "Any updates on my order?",
        "I need to know where my order is",
        "Can you track my order?",
        "What happened to my order?",
        "My order is taking too long",
        "When will I receive my order?",
        "Delivery status please",
        "Has my package been dispatched?",
        "I'm waiting for my order",
        "Package tracking request",
        "Where is my delivery?",
        "Is my package lost?",
        "My order hasn't arrived",
        "Shipment status",
        "Track my recent order",
        "I haven't received my order yet",
        "Expected delivery date?",
        "When does my order arrive?"
    ],
    
    'payment_issues': [
        "Payment failed",
        "My payment didn't go through",
        "I was charged twice",
        "Payment error",
        "Card declined",
        "Can't complete payment",
        "Payment problem",
        "My card isn't working",
        "Transaction failed",
        "Payment not accepted",
        "Double charged on my card",
        "Payment unsuccessful",
        "Error during payment",
        "Credit card issue",
        "Payment keeps failing",
        "Can't process my payment",
        "Billing issue",
        "My payment was rejected",
        "Problem with checkout",
        "Payment gateway error",
        "Money deducted but order failed",
        "Charged but no confirmation",
        "Payment stuck",
        "Card not accepted",
        "Debit card declined",
        "Payment method not working",
        "Checkout error",
        "Can't pay for my order",
        "Payment issue with my account",
        "Help with payment problem"
    ],
    
    'product_information': [
        "Tell me about this product",
        "Product specifications",
        "Is this item available?",
        "What are the product details?",
        "Product features",
        "Do you have this in stock?",
        "Item availability",
        "Product description please",
        "What colors does it come in?",
        "What sizes are available?",
        "Is this product good?",
        "Product reviews",
        "Compare products",
        "What's the warranty on this?",
        "Product dimensions",
        "Material information",
        "How does this product work?",
        "Is this compatible with...?",
        "What's included in the package?",
        "Product quality",
        "More info about the item",
        "Technical specifications",
        "Is this genuine product?",
        "Product origin",
        "Brand information",
        "How to use this product?",
        "Product recommendations",
        "Best seller products",
        "New arrivals",
        "What products do you have?"
    ],
    
    'complaint': [
        "I'm not satisfied",
        "This is unacceptable",
        "I want to file a complaint",
        "Very disappointed with the service",
        "Bad experience",
        "Poor quality product",
        "I received a damaged item",
        "Product is defective",
        "Terrible customer service",
        "I'm very unhappy",
        "This is not what I ordered",
        "Product doesn't work",
        "I want to speak to a manager",
        "Horrible experience",
        "I'm frustrated",
        "The product is broken",
        "Wrong item received",
        "Item arrived damaged",
        "Quality is very poor",
        "Not as described",
        "False advertising",
        "Missing items in my order",
        "Product malfunction",
        "Very bad service",
        "I need to complain",
        "This is fraud",
        "I feel cheated",
        "Worst experience ever",
        "Dissatisfied customer",
        "I demand compensation"
    ],
    
    'return_exchange': [
        "I want to return this",
        "How do I return an item?",
        "Return policy",
        "I need to exchange this",
        "Can I return this product?",
        "Return request",
        "Exchange for different size",
        "Initiate return",
        "I want to send this back",
        "Return my order",
        "How to return?",
        "Exchange policy",
        "I'd like to exchange",
        "Return shipping",
        "Free returns?",
        "Start a return",
        "Replace my item",
        "Return label request",
        "30 day return",
        "Return window",
        "Can I swap this?",
        "Wrong size, need exchange",
        "Item doesn't fit, return please",
        "Quality issue return",
        "Refund and return",
        "Return damaged product",
        "Exchange for store credit",
        "Return process",
        "Return pickup",
        "Schedule a return"
    ],
    
    'account_issues': [
        "I can't log in",
        "Reset my password",
        "Account locked",
        "Forgot password",
        "Can't access my account",
        "Login problem",
        "Password reset",
        "Account recovery",
        "Update my email",
        "Change password",
        "Account not working",
        "Sign in issues",
        "My account is blocked",
        "Authentication failed",
        "Verification code not received",
        "Account suspended",
        "Create new password",
        "Email not recognized",
        "Two factor authentication issue",
        "Unlock my account",
        "Login credentials not working",
        "Account security concern",
        "Delete my account",
        "Update account information",
        "Change my phone number",
        "Account hacked",
        "Someone else accessing my account",
        "Merge accounts",
        "Account settings",
        "Profile update"
    ],
    
    'general_inquiry': [
        "Hello",
        "Hi there",
        "I have a question",
        "Can you help me?",
        "I need assistance",
        "Customer support",
        "General question",
        "I need help",
        "How can I contact you?",
        "Business hours",
        "Contact information",
        "Speak to representative",
        "Human agent please",
        "General help",
        "Information needed",
        "Quick question",
        "Can someone help me?",
        "I need information",
        "Support request",
        "Help me please",
        "Good morning",
        "Good afternoon",
        "Is anyone there?",
        "I need customer service",
        "Connect me to support",
        "How does this work?",
        "First time customer",
        "New here",
        "General assistance",
        "Thank you"
    ],
    
    'shipping_inquiry': [
        "Shipping options",
        "How much is shipping?",
        "Free shipping?",
        "Shipping cost",
        "Delivery options",
        "Express shipping",
        "Next day delivery",
        "International shipping",
        "Shipping time",
        "Delivery fee",
        "Do you ship to my area?",
        "Shipping rates",
        "Standard shipping",
        "Priority shipping",
        "Shipping methods",
        "Free delivery threshold",
        "Delivery days",
        "Weekend delivery?",
        "Shipping zones",
        "Expedited shipping",
        "Same day delivery",
        "Shipping charges",
        "Delivery cost estimate",
        "Ship to different address",
        "Change shipping address",
        "Fastest shipping option",
        "Economy shipping",
        "Ground shipping",
        "Air shipping",
        "Shipping discount"
    ],
    
    'refund_status': [
        "Where is my refund?",
        "Refund status",
        "When will I get my refund?",
        "Refund not received",
        "Check refund status",
        "How long for refund?",
        "Refund pending",
        "My refund is delayed",
        "I haven't received my refund",
        "Refund timeline",
        "Refund processing time",
        "Waiting for refund",
        "Refund to credit card",
        "Refund to bank account",
        "Track my refund",
        "Refund amount incorrect",
        "Full refund please",
        "Partial refund received",
        "Refund request status",
        "Expedite my refund",
        "When was refund issued?",
        "Refund confirmation",
        "I need my money back",
        "Refund not showing",
        "Refund taking too long",
        "Refund inquiry",
        "Credit not received",
        "Money back please",
        "Reimburse me",
        "When is refund processed?"
    ]
}

# Additional variations to expand the dataset
VARIATIONS = {
    'prefixes': [
        "", "Hi, ", "Hello, ", "Hey, ", "Excuse me, ", "Please, ",
        "I need to ", "I would like to ", "I want to ", "Can I ",
        "Could you ", "Would you ", "Is it possible to ", "How do I "
    ],
    'suffixes': [
        "", ".", "!", "?", " please", " thanks", " thank you",
        " ASAP", " urgently", " immediately", " right away", " now"
    ]
}


def generate_variations(queries: list, num_variations: int = 3) -> list:
    """Generate variations of queries with prefixes and suffixes."""
    variations = []
    
    for query in queries:
        # Original query
        variations.append(query)
        
        # Add variations
        for _ in range(num_variations):
            prefix = random.choice(VARIATIONS['prefixes'])
            suffix = random.choice(VARIATIONS['suffixes'])
            
            # Don't add prefix if query already starts with one
            if query[0].isupper() and prefix:
                new_query = prefix.lower() + query.lower() + suffix
            else:
                new_query = prefix + query + suffix
            
            # Clean up
            new_query = new_query.strip()
            if new_query and new_query not in variations:
                variations.append(new_query)
    
    return variations


def generate_dataset(output_path: str = "data/processed/customer_queries.csv",
                     variations_per_query: int = 3,
                     min_per_intent: int = 100) -> str:
    """
    Generate synthetic customer service dataset.
    
    Args:
        output_path: Path to save the CSV file
        variations_per_query: Number of variations per base query
        min_per_intent: Minimum samples per intent
        
    Returns:
        Path to generated dataset
    """
    print("Generating customer service dataset...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_data = []
    
    for intent, queries in INTENT_DATA.items():
        # Generate variations
        expanded_queries = generate_variations(queries, variations_per_query)
        
        # Ensure minimum samples
        while len(expanded_queries) < min_per_intent:
            # Add more variations
            base_query = random.choice(queries)
            prefix = random.choice(VARIATIONS['prefixes'])
            suffix = random.choice(VARIATIONS['suffixes'])
            new_query = f"{prefix}{base_query}{suffix}".strip()
            if new_query not in expanded_queries:
                expanded_queries.append(new_query)
        
        # Add to dataset
        for query in expanded_queries:
            all_data.append({
                'query': query,
                'intent': intent
            })
    
    # Shuffle data
    random.shuffle(all_data)
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'intent'])
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Dataset generated: {output_path}")
    print(f"Total samples: {len(all_data)}")
    
    # Print distribution
    intent_counts = {}
    for item in all_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nIntent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    
    return output_path


def split_dataset(input_path: str, train_ratio: float = 0.8) -> tuple:
    """
    Split dataset into train and test sets.
    
    Args:
        input_path: Path to full dataset
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (train_path, test_path)
    """
    import pandas as pd
    
    df = pd.read_csv(input_path)
    
    # Stratified split
    train_samples = []
    test_samples = []
    
    for intent in df['intent'].unique():
        intent_df = df[df['intent'] == intent]
        intent_data = intent_df.to_dict('records')
        random.shuffle(intent_data)
        
        split_idx = int(len(intent_data) * train_ratio)
        train_samples.extend(intent_data[:split_idx])
        test_samples.extend(intent_data[split_idx:])
    
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    # Save splits
    base_dir = os.path.dirname(input_path)
    train_path = os.path.join(base_dir, 'train.csv')
    test_path = os.path.join(base_dir, 'test.csv')
    
    pd.DataFrame(train_samples).to_csv(train_path, index=False)
    pd.DataFrame(test_samples).to_csv(test_path, index=False)
    
    print(f"Training set: {len(train_samples)} samples -> {train_path}")
    print(f"Test set: {len(test_samples)} samples -> {test_path}")
    
    return train_path, test_path


if __name__ == "__main__":
    # Generate dataset
    dataset_path = generate_dataset(
        output_path="data/processed/customer_queries.csv",
        variations_per_query=10,
        min_per_intent=500
    )
    
    # Split into train/test
    split_dataset(dataset_path)
    
    print("\n✅ Dataset generation complete!")
