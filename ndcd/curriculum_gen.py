import os
import random
import json
import urllib.request

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {url} to {target_path}...")
        try:
            urllib.request.urlretrieve(url, target_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def generate_chars(filename="ndcd/data/level1_chars.txt", size=20000):
    """
    Level 1: Advanced Character Learning.
    1. Forwards Alphabet: ABC...Z
    2. Backwards Alphabet: ZYX...A
    3. Random Repetitions: AAABBB...
    """
    print(f"Generating {filename}...")
    chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chars_lower = "abcdefghijklmnopqrstuvwxyz"
    all_chars = chars_upper + chars_lower
    
    with open(filename, "w") as f:
        # 1. Forward/Backward Sequences (repeated many times)
        for _ in range(50):
            f.write(chars_upper + " ")
            f.write(chars_lower + " ")
            f.write(chars_upper[::-1] + " ")
            f.write(chars_lower[::-1] + " ")
            f.write("0123456789 ")
            f.write("9876543210 ")
            
        # 2. Block Repetitions (Stability)
        for _ in range(size // 15):
            char = random.choice(all_chars)
            repeat = random.randint(5, 10)
            f.write(char * repeat + " ")
            
def generate_words(filename="ndcd/data/level2_words.txt", source_file="ndcd/data/google-10000-english.txt", size=100000):
    """
    Level 2: Vocabulary Expansion (Top 10,000 Common Words).
    Source: https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
    """
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
    download_file(url, source_file)
    
    print(f"Generating {filename} from {source_file}...")
    
    vocab = []
    if os.path.exists(source_file):
        with open(source_file, "r") as f:
            vocab = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {source_file} not found even after download attempt. Using fallback.")
        vocab = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i"] # Fallback
        
    with open(filename, "w") as f:
        # 1. Write disjoint words
        for _ in range(size // 10):
            word = random.choice(vocab)
            f.write(word + " ")
            
        # 2. Sequential snippets (if ordered) - The 10k list is frequency ordered.
        # Writing them in order helps learn frequency distribution
        for _ in range(5):
             for w in vocab[:1000]: # Top 1000 most frequent repeated
                 f.write(w + " ")
        
        # 3. Simple pairings
        for _ in range(size // 10):
            w1 = random.choice(vocab)
            w2 = random.choice(vocab)
            f.write(f"{w1} {w2} ")

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
    generate_words()
    generate_quotes()
    print("Curriculum data generated.")
