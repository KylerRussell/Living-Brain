import random
import os

def generate_holophrases(filename="ndcd/data/level1_holophrases.txt", size=20000):
    """
    Phase 1: Holophrases and Concrete Chunks
    Early in development, children attempt to reproduce whole adult utterances 
    rather than isolated words.
    """
    print(f"Generating {filename} (Phase 1: Holophrases)...")
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
    with open(filename, "w") as f:
        current_size = 0
        while current_size < size:
            phrase = random.choice(phrases)
            f.write(phrase + " ")
            current_size += len(phrase) + 1

def generate_slot_and_frame(filename="ndcd/data/level2_slot_frame.txt", size=150000):
    """
    Phase 2: Slot-and-Frame Patterns
    Humans generalize grammar by noticing variations within fixed, recurrent utterance frames.
    """
    print(f"Generating {filename} (Phase 2: Slot-and-Frame)...")
    nouns = ["ball", "dog", "cat", "car", "apple", "cup", "book", "bear", "shoe", "bottle"]
    
    frames = [
        "Where is the {noun}?",
        "More {noun}, please.",
        "I see the {noun}.",
        "That is a big {noun}.",
        "Give me the {noun}.",
        "The {noun} fell down.",
        "My {noun} is gone."
    ]
    
    with open(filename, "w") as f:
        current_size = 0
        while current_size < size:
            frame = random.choice(frames)
            noun = random.choice(nouns)
            sentence = frame.format(noun=noun)
            
            f.write(sentence + " ")
            current_size += len(sentence) + 1

def generate_complex_constructions(filename="ndcd/data/level3_complex.txt", size=200000):
    """
    Phase 3: Complex Constructions and Hierarchical Expansion
    Gradually introduce multi-clause sentences, varied verb tenses, and conjunctions.
    """
    print(f"Generating {filename} (Phase 3: Complex Constructions)...")
    
    sentences = [
        "I want the ball because it is fun to bounce.",
        "If we go to the slide, I will go first.",
        "He was running fast and then he fell down.",
        "We played on the swings until it got dark.",
        "Because she shared her toy, we are friends.",
        "Can you push me higher while I hold on?",
        "When the bell rings, we have to go inside.",
        "The sandbox is full of wet sand today, so we can build a castle.",
        "I need to brush my teeth before I go to sleep.",
        "Read me a story because I am not tired yet.",
        "When the lights go out, the stars shine bright.",
        "My blanket is soft, but my pillow is too hard.",
        "If you sing a song, I will close my eyes.",
        "He drank some water, and then he laid down.",
        "Although it is late, I cannot sleep.",
        "We put the toys away before getting into bed.",
        "I am eating an apple because I am hungry.",
        "When dinner is ready, we will sit at the table.",
        "If you eat your vegetables, you can have dessert.",
        "She spilled the milk, but she cleaned it up.",
        "The soup is too hot, so I am blowing on it.",
        "We baked cookies while it was raining outside.",
        "Because I like pizza, I asked for another slice.",
        "He washed his hands before he ate the sandwich."
    ]
    
    with open(filename, "w") as f:
        current_size = 0
        while current_size < size:
            sentence = random.choice(sentences)
            f.write(sentence + " ")
            current_size += len(sentence) + 1

def generate_contextual_continuity(filename="ndcd/data/level4_contextual.txt", size=200000):
    """
    Phase 4: Contextual Continuity (Simulating Joint Attention)
    Introduce "topic persistence." Generate 10 to 20 sentences in a row that share a latent theme 
    or overlapping vocabulary before switching to a new topic.
    """
    print(f"Generating {filename} (Phase 4: Contextual Continuity)...")
    
    topics = {
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
    
    theme_keys = list(topics.keys())
    
    with open(filename, "w") as f:
        current_size = 0
        while current_size < size:
            topic = random.choice(theme_keys)
            sentences = topics[topic]
            
            repeats = random.randint(10, 20)
            for _ in range(repeats):
                sentence = random.choice(sentences)
                f.write(sentence + " ")
                current_size += len(sentence) + 1
                if current_size >= size:
                    break

if __name__ == "__main__":
    os.makedirs("ndcd/data", exist_ok=True)
    generate_holophrases()
    generate_slot_and_frame()
    generate_complex_constructions()
    generate_contextual_continuity()
    print("Curriculum data generated.")
