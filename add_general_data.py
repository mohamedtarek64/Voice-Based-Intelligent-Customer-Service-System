
import pandas as pd
import os

new_data = [
    # GREETING & SMALL TALK
    ("hi", "greeting"), ("hello", "greeting"), ("hey", "greeting"), ("hi there", "greeting"),
    ("good morning", "greeting"), ("good afternoon", "greeting"), ("how are you", "greeting"),
    ("how is it going", "greeting"), ("what's up", "greeting"), ("are you there", "greeting"),
    ("hello bot", "greeting"), ("hey friend", "greeting"), ("is anyone online", "greeting"),
    
    # GOODBYE
    ("bye", "goodbye"), ("goodbye", "goodbye"), ("see you", "goodbye"), ("later", "goodbye"),
    ("talk to you later", "goodbye"), ("exit", "goodbye"), ("stop", "goodbye"),
    ("bye bye", "goodbye"), ("have a good day", "goodbye"), ("see you soon", "goodbye"),
    
    # THANKS (Adding more variations to fix low confidence)
    ("thanks", "thanks"), ("thank you", "thanks"), ("thank you so much", "thanks"),
    ("thanks a lot", "thanks"), ("i appreciate it", "thanks"), ("cool thanks", "thanks"),
    ("cheers", "thanks"), ("great thanks", "thanks"), ("awesome thank you", "thanks"),
    ("you've been helpful", "thanks"), ("thanks for the help", "thanks"),
    ("thank you assistant", "thanks"), ("much appreciated", "thanks"), ("thanks!", "thanks"),
    ("thx", "thanks"), ("many thanks", "thanks"), ("that's helpful", "thanks"),
    
    # BOT INFO
    ("who are you", "bot_info"), ("what is your name", "bot_info"), ("what are you", "bot_info"),
    ("are you a robot", "bot_info"), ("tell me about yourself", "bot_info"),
    ("who am i talking to", "bot_info"), ("are you ai", "bot_info"), ("what is your identity", "bot_info"),
    
    # CAPABILITIES
    ("what can you do", "capabilities"), ("how can you help", "capabilities"),
    ("help me", "capabilities"), ("show me what you can do", "capabilities"),
    ("give me some help", "capabilities"), ("what are your features", "capabilities"),
    
    # WEATHER
    ("weather", "weather"), ("how is the weather", "weather"), ("weather update", "weather"),
    ("is it raining", "weather"), ("whats the temperature", "weather"),
    
    # HUMOR
    ("joke", "humor"), ("tell me a joke", "humor"), ("make me laugh", "humor"),
    ("do you know any jokes", "humor"), ("another joke", "humor")
]

# Multiply to get enough samples (increasing to 20x to dominate the sparse space)
expanded_data = new_data * 20

df_new = pd.DataFrame(expanded_data, columns=['query', 'intent'])

file_path = 'data/processed/customer_queries.csv'
if os.path.exists(file_path):
    # Just overwrite with fresh data + original data to prevent duplicates if run multiple times
    # Or append but we need to be careful. Let's append but check for duplicates if possible, 
    # though for ML more samples is fine.
    df_old = pd.read_csv(file_path)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_csv(file_path, index=False)
    print(f"Added {len(df_new)} new samples to {file_path}")
else:
    print(f"Error: {file_path} not found")
