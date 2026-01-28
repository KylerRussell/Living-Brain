
import numpy as np
from graph import DynamicGraph
from engine import DragonEngine
from curriculum import Curriculum
from language import LanguageInterface
import sys

def main():
    print("=== Initializing Language-Capable Dragon ===")
    
    # 1. Setup
    num_nodes = 200 # Larger brain for high-dim vectors
    # Note: Our random embeddings are dim=num_nodes.
    # In reality, we might map embedding (dim=50) to sensory nodes (first 50)
    # But for distributed representation, we can map to the whole state or a subset.
    # Let's map Input Word -> Clamped Input (Sensory Nodes)
    # And Output Word -> Readout from whole state (or subset)
    
    graph = DynamicGraph(num_nodes=num_nodes, m_edges=4, p_triad=0.1, seed=101)
    engine = DragonEngine(graph, dt=0.01)
    curriculum = Curriculum(engine)
    
    vocab = ["Hello", "Hi", "Cat", "Meow", "Dog", "Woof", "Good", "Bad", "Food", "Yum", "Poison", "Yuck"]
    lang = LanguageInterface(vocab, embedding_dim=num_nodes)
    
    # 2. Training Data (Association/Grounding)
    # We train the brain to associate Input -> Output
    pairs = [
        ("Hello", "Hi"),
        ("Cat", "Meow"),
        ("Dog", "Woof"),
        ("Food", "Yum"),
        ("Poison", "Yuck")
    ]
    
    print(f"Training on pairs: {pairs}")
    
    # Convert to vectors
    dataset = []
    for word_in, word_out in pairs:
        vec_in = lang.encode(word_in)
        vec_out = lang.encode(word_out)
        dataset.append((vec_in, vec_out))
        
    # 3. Train (Grounding Phase)
    # We use the existing run_grounding method
    # Note: run_grounding expects (input, target). 
    # For EqProp, we clamp input, settle, then nudge output towards target.
    # Our simple implementation in curriculum.py supports this format.
    
    curriculum.run_grounding(dataset, epochs=50, beta=0.5, learning_rate=0.05)
    
    # 4. Chat Loop
    print("\n=== Chat with the Dragon (Type 'exit' to quit) ===")
    print("Known words:", vocab)
    
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
            
        if user_input.lower() == 'exit':
            break
            
        if user_input not in vocab:
            print(f"Dragon: <Confused> (I don't know '{user_input}')")
            continue
            
        # Encode
        input_vec = lang.encode(user_input)
        
        # Settle (Inference)
        # We clamp the input and let the system settle to an attractor
        engine.settle(input_vec, duration_steps=100)
        
        # Decode output
        # ideally we decode the state of the *output* nodes (or whole state if auto-associative)
        # Our training nudged the WHOLE state towards vec_out?
        # Wait, run_grounding nudges 'target_vec'.
        # If vec_out is full dimension, then yes, we trained the whole state to look like vec_out.
        output_state = graph.states
        response = lang.decode(output_state)
        
        print(f"Dragon: {response}")

if __name__ == "__main__":
    main()
