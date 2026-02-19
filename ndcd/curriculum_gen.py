import random
import json
import urllib.request
import os

def generate_simple_lowercase(filename="ndcd/data/simple_lower.txt", size=10000):
    """
    Trivial task: Lowecase alphabet sequence ONLY.
    a -> b -> c ... -> z -> a
    """
    print(f"Generating {filename} (Lowercase Alphabet Only)...")
    chars = "abcdefghijklmnopqrstuvwxyz"
    
    with open(filename, "w") as f:
        # Just repeat the alphabet sequence
        for _ in range(size // 26 + 1):
            f.write(chars + " ")
            
            # Occasionally repeat single letters to anchor them (a a a b b b)
            if random.random() < 0.1:
                char = random.choice(chars)
                f.write((char * 5) + " ")

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {url} to {target_path}...")
        try:
            urllib.request.urlretrieve(url, target_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def generate_chars(filename="ndcd/data/level1_chars.txt", size=20000):
    """
    Level 1: Simple character sequence learning.
    Repeats: "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
    """
    print(f"Generating {filename}...")
    pattern = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"

    with open(filename, "w") as f:
        current_size = 0
        while current_size < size:
            f.write(pattern)
            current_size += len(pattern)
            
def generate_toddler_words(filename="ndcd/data/level2_words.txt", source_file="ndcd/data/google-10000-english.txt", size=150000):
    """
    Level 2: Toddler Vocabulary Expansion (Progressive).
    Source: https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
    
    Strategy:
    1.  Stage 1: Top 50 words (nouns/verbs). Dense repetition.
    2.  Stage 2: Top 100 words. Simple 2-word pairs.
    3.  Stage 3: Top 300 words. 3-word pairings.
    4.  Stage 4: Top 1000 words. Mixed sentences.
    """
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
    download_file(url, source_file)
    
    print(f"Generating {filename} from {source_file} (Toddler Mode)...")
    
    vocab = []
    if os.path.exists(source_file):
        with open(source_file, "r") as f:
            vocab = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {source_file} not found. Using fallback.")
        vocab = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i"] # Fallback

    with open(filename, "w") as f:
        # --- Stage 1: The First 50 Words (Naming) ---
        # Focus: Nouns, simple verbs. "Ball", "Mom", "Go", "No".
        # We assume the list is frequency sorted.
        stage1_vocab = vocab[:50]
        f.write("---STAGE1--- ")
        for _ in range(size // 10): # 10% of data
            # Heavy repetition of single words: "Ball. Ball. Ball."
            w = random.choice(stage1_vocab)
            f.write(f"{w} {w} {w} ")
            
        # --- Stage 2: Top 100 Words (Attributes) ---
        # Focus: Adjectives + Noun. "Big Ball", "Good Boy".
        stage2_vocab = vocab[:100]
        f.write("---STAGE2--- ")
        for _ in range(size // 5): # 20% of data
            w1 = random.choice(stage2_vocab)
            w2 = random.choice(stage2_vocab)
            f.write(f"{w1} {w2} ")
            
        # --- Stage 3: Top 300 Words (Simple Subject-Verb-Object) ---
        stage3_vocab = vocab[:300]
        f.write("---STAGE3--- ")
        for _ in range(size // 3): # 30% of data
             # "I go home", "You see dog"
             w1 = random.choice(stage3_vocab)
             w2 = random.choice(stage3_vocab)
             w3 = random.choice(stage3_vocab)
             f.write(f"{w1} {w2} {w3} ")

        # --- Stage 4: Top 1000 Words (Explosion) ---
        stage4_vocab = vocab[:1000]
        f.write("---STAGE4--- ")
        remaining_size = size - (size//10 + size//5 + size//3)
        # Just generate tokens roughly to fill
        # It's okay if exact size isn't perfect, just need lots of data.
        for _ in range(remaining_size // 5): 
             # Random 5-word "proto-sentences"
             s = " ".join([random.choice(stage4_vocab) for _ in range(5)])
             f.write(s + " ")

def generate_quotes(filename="ndcd/data/level3_quotes.txt", source_file="ndcd/data/english.json", size=200000):
    """
    Level 3: Complex Quotes and Proverbs (from english.json).
    Source: https://github.com/monkeytypegame/monkeytype/blob/master/frontend/static/quotes/english.json
    """
    url = "https://raw.githubusercontent.com/monkeytypegame/monkeytype/master/frontend/static/quotes/english.json"
    download_file(url, source_file)
    
    print(f"Generating {filename} from {source_file}...")
    
    quotes = []
    if os.path.exists(source_file):
        try:
            with open(source_file, "r") as f:
                data = json.load(f)
                # Structure: { "quotes": [ { "text": "...", ... }, ... ] }
                if "quotes" in data:
                    quotes = [item["text"] for item in data["quotes"]]
        except Exception as e:
            print(f"Error parsing json: {e}")
            
    if not quotes:
        print("Warning: No quotes found. Using fallback.")
        quotes = ["The quick brown fox jumps over the lazy dog."]

    with open(filename, "w") as f:
        # Write repeated quotes to learn grammar structure
        # We want a lot of data. 
        # Randomly sample quotes until size is reached
        current_size = 0
        while current_size < size:
             q = random.choice(quotes)
             f.write(q + " ")
             current_size += len(q) + 1

if __name__ == "__main__":
    os.makedirs("ndcd/data", exist_ok=True)
    generate_chars()
    generate_toddler_words()
    generate_quotes()
    print("Curriculum data generated.")
