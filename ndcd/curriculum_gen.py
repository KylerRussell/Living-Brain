import random
import os
import itertools

def generate_holophrases(train_file="ndcd/data/train/level1_holophrases.txt", 
                         test_file="ndcd/data/eval/level1_holophrases.txt", 
                         train_size=20000, test_size=5000):
    """
    Phase 1: Holophrases and Concrete Chunks
    Early in development, children attempt to reproduce whole adult utterances 
    rather than isolated words.
    """
    print(f"Generating {train_file} and {test_file} (Phase 1: Holophrases)...")
    phrases = [
        "Look at that.",
        "I want it.",
        "Go away.",
        "Give me.",
        "Pick it up.",
        "Put it down.",
        "No more.",
        "All gone.",
        "Come here.",
        "Help me.",
        "What is that?",
        "Stop it."
    ]
    
    # Randomly shuffle for split
    random.shuffle(phrases)
    # 80/20 vocabulary split for Phase 1 to test basic extrapolation
    split_idx = int(len(phrases) * 0.8)
    train_phrases = phrases[:split_idx]
    test_phrases = phrases[split_idx:]
    
    with open(train_file, "w") as f:
        current_size = 0
        while current_size < train_size:
            phrase = random.choice(train_phrases)
            f.write(phrase + " ")
            current_size += len(phrase) + 1
            
    with open(test_file, "w") as f:
        current_size = 0
        while current_size < test_size:
            phrase = random.choice(test_phrases)
            f.write(phrase + " ")
            current_size += len(phrase) + 1

def generate_slot_and_frame(train_file="ndcd/data/train/level2_slot_frame.txt", 
                            test_file="ndcd/data/eval/level2_slot_frame.txt",
                            train_size=150000, test_size=30000):
    """
    Phase 2: Slot-and-Frame Patterns
    Humans generalize grammar by noticing variations within fixed, recurrent utterance frames.
    Implements Template Generalization and Maximum Compound Divergence (MCD).
    """
    print(f"Generating {train_file} and {test_file} (Phase 2: Slot-and-Frame)...")
    nouns = ["ball", "dog", "cat", "car", "apple", "cup", "book", "bear", "shoe", "bottle"]
    
    all_frames = [
        "Where is the {noun}?",
        "More {noun}, please.",
        "I see the {noun}.",
        "That is a big {noun}.",
        "Give me the {noun}.",
        "The {noun} fell down.",
        "My {noun} is gone."
    ]
    
    # 1. Template Generalization: Withhold one entire frame from training
    train_frames = all_frames[:-1]
    withheld_frame = all_frames[-1]
    
    # 2. Maximum Compound Divergence (MCD): 
    # Hold out 20% of noun-frame combinations for the eval set
    valid_combinations = list(itertools.product(train_frames, nouns))
    random.shuffle(valid_combinations)
    
    num_mcd_holdouts = int(len(valid_combinations) * 0.2)
    test_combinations = valid_combinations[:num_mcd_holdouts]
    train_combinations = valid_combinations[num_mcd_holdouts:]
    
    with open(train_file, "w") as f:
        current_size = 0
        while current_size < train_size:
            frame, noun = random.choice(train_combinations)
            sentence = frame.format(noun=noun)
            f.write(sentence + " ")
            current_size += len(sentence) + 1
            
    with open(test_file, "w") as f:
        current_size = 0
        while current_size < test_size:
            # 50% chance to test MCD holdouts, 50% chance to test pure Template Generalization
            if random.random() < 0.5:
                frame, noun = random.choice(test_combinations)
                sentence = frame.format(noun=noun)
            else:
                noun = random.choice(nouns)
                sentence = withheld_frame.format(noun=noun)
                
            f.write(sentence + " ")
            current_size += len(sentence) + 1

def generate_complex_constructions(train_file="ndcd/data/train/level3_complex.txt", 
                                   test_file="ndcd/data/eval/level3_complex.txt",
                                   train_size=200000, test_size=40000):
    """
    Phase 3: Complex Constructions and Hierarchical Expansion
    Implements Length & Structural Generalization (Productivity).
    Train on <= 2 clauses, test on recursively nested >= 3 clauses.
    """
    print(f"Generating {train_file} and {test_file} (Phase 3: Complex Constructions)...")
    
    # Clause primitives (1 clause)
    clause1 = [
        "I want the ball", "it is fun to bounce", "we go to the slide", "I will go first",
        "He was running fast", "he fell down", "We played on the swings", "it got dark",
        "she shared her toy", "we are friends", "you push me higher", "I hold on",
        "the bell rings", "we have to go inside", "The sandbox is full of wet sand today",
        "we can build a castle", "I need to brush my teeth", "I go to sleep",
        "Read me a story", "I am not tired yet", "the lights go out", "the stars shine bright",
        "My blanket is soft", "my pillow is too hard", "you sing a song", "I will close my eyes",
        "He drank some water", "he laid down", "it is late", "I cannot sleep",
        "We put the toys away", "getting into bed", "I am eating an apple", "I am hungry",
        "dinner is ready", "we will sit at the table", "you eat your vegetables", "you can have dessert",
        "She spilled the milk", "she cleaned it up", "The soup is too hot", "I am blowing on it",
        "We baked cookies", "it was raining outside", "I like pizza", "I asked for another slice",
        "He washed his hands", "he ate the sandwich"
    ]
    conjunctions = ["because", "if", "and then", "until", "so", "while", "when", "before", "but", "although"]

    def generate_sentence(num_clauses):
        selected_clauses = random.sample(clause1, num_clauses)
        selected_conjunctions = random.sample(conjunctions, num_clauses - 1)
        
        sentence = selected_clauses[0]
        for i in range(num_clauses - 1):
            sentence += f" {selected_conjunctions[i]} {selected_clauses[i+1]}"
        
        # Capitalize first letter, add period
        sentence = sentence[0].upper() + sentence[1:] + "."
        return sentence

    # Generate Training Sentences (Max 2 clauses)
    train_sentences = [generate_sentence(random.randint(1, 2)) for _ in range(50)]
    
    # Generate Test Sentences (Max 3-4 clauses)
    test_sentences = [generate_sentence(random.randint(3, 4)) for _ in range(20)]

    with open(train_file, "w") as f:
        current_size = 0
        while current_size < train_size:
            sentence = random.choice(train_sentences)
            f.write(sentence + " ")
            current_size += len(sentence) + 1
            
    with open(test_file, "w") as f:
        current_size = 0
        while current_size < test_size:
            sentence = random.choice(test_sentences)
            f.write(sentence + " ")
            current_size += len(sentence) + 1

def generate_contextual_continuity(train_file="ndcd/data/train/level4_contextual.txt",
                                   test_file="ndcd/data/eval/level4_contextual.txt", 
                                   train_size=200000, test_size=40000):
    """
    Phase 4: Contextual Continuity (Simulating Joint Attention)
    Test set contains completely novel sentences that share the same latent theme.
    """
    print(f"Generating {train_file} and {test_file} (Phase 4: Contextual Continuity)...")
    
    topics_train = {
        "playground": [
            "I want the ball because it is fun to bounce.",
            "If we go to the slide, I will go first.",
            "He was running fast and then he fell down.",
            "We played on the swings until it got dark.",
            "Because she shared her toy, we are friends.",
            "Can you push me higher while I hold on?",
            "When the bell rings, we have to go inside.",
            "The sandbox is full of wet sand today, so we can build a castle."
        ],
        "bedtime": [
            "I need to brush my teeth before I go to sleep.",
            "Read me a story because I am not tired yet.",
            "When the lights go out, the stars shine bright.",
            "My blanket is soft, but my pillow is too hard.",
            "If you sing a song, I will close my eyes.",
            "He drank some water, and then he laid down.",
            "Although it is late, I cannot sleep.",
            "We put the toys away before getting into bed."
        ],
        "food": [
            "I am eating an apple because I am hungry.",
            "When dinner is ready, we will sit at the table.",
            "If you eat your vegetables, you can have dessert.",
            "She spilled the milk, but she cleaned it up.",
            "The soup is too hot, so I am blowing on it.",
            "We baked cookies while it was raining outside.",
            "Because I like pizza, I asked for another slice.",
            "He washed his hands before he ate the sandwich."
        ]
    }
    
    # Completely novel sentences sharing the same vocabulary and theme
    topics_eval = {
        "playground": [
            "The slide is wet today so we played on the swings.",
            "She fell down fast because we were running.",
            "Can we build a castle if the sandbox has wet sand?",
            "I will hold on while you bounce the ball."
        ],
        "bedtime": [
            "Before getting into bed, I will read a story.",
            "I cannot sleep because my blanket is too hard.",
            "I am not tired but I need to close my eyes.",
            "Although the stars shine bright, I need to drink some water."
        ],
        "food": [
            "I like the sandwich but the milk is too hot.",
            "If I wash my hands I can have a slice of pizza.",
            "Because I am hungry I baked an apple.",
            "When I clean the table we will eat the soup."
        ]
    }
    
    theme_keys = list(topics_train.keys())
    
    with open(train_file, "w") as f:
        current_size = 0
        while current_size < train_size:
            topic = random.choice(theme_keys)
            sentences = topics_train[topic]
            
            repeats = random.randint(10, 20)
            for _ in range(repeats):
                sentence = random.choice(sentences)
                f.write(sentence + " ")
                current_size += len(sentence) + 1
                if current_size >= train_size:
                    break
                    
    with open(test_file, "w") as f:
        current_size = 0
        while current_size < test_size:
            topic = random.choice(theme_keys)
            sentences = topics_eval[topic]
            
            repeats = random.randint(10, 20)
            for _ in range(repeats):
                sentence = random.choice(sentences)
                f.write(sentence + " ")
                current_size += len(sentence) + 1
                if current_size >= test_size:
                    break

if __name__ == "__main__":
    os.makedirs("ndcd/data/train", exist_ok=True)
    os.makedirs("ndcd/data/eval", exist_ok=True)
    generate_holophrases()
    generate_slot_and_frame()
    generate_complex_constructions()
    generate_contextual_continuity()
    print("Curriculum train/eval data generated.")

