
import numpy as np

class LanguageInterface:
    def __init__(self, vocab_words, embedding_dim, seed=42):
        """
        Simple Language Interface using static random embeddings.
        
        Args:
            vocab_words: List of strings (vocabulary).
            embedding_dim: Dimension of the state vector (must match num_nodes).
        """
        self.vocab = vocab_words
        self.word_to_idx = {w: i for i, w in enumerate(vocab_words)}
        self.dim = embedding_dim
        
        np.random.seed(seed)
        # Random orthogonal-ish vectors for each word
        # We normalize them to have unit length or similar scale as neural activity
        self.embeddings = np.random.normal(0, 1, (len(vocab_words), embedding_dim))
        
        # Normalize embeddings to be unit vectors for cosine similarity
        norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norm + 1e-9)
        
    def encode(self, word):
        """Returns the embedding vector for a word."""
        if word not in self.word_to_idx:
            print(f"Warning: Word '{word}' not in vocab, returning zero vector.")
            return np.zeros(self.dim)
        idx = self.word_to_idx[word]
        return self.embeddings[idx]
        
    def decode(self, vector):
        """Finds the nearest word in the vocabulary to the input vector."""
        # Cosine similarity: (A . B) / (|A| |B|)
        # Our embeddings B are already normalized.
        # We need to normalize A (the input vector)
        vec_norm = np.linalg.norm(vector)
        if vec_norm < 1e-9:
            return "<Silence>"
            
        unit_vec = vector / vec_norm
        
        scores = self.embeddings @ unit_vec
        best_idx = np.argmax(scores)
        return self.vocab[best_idx]
