
import numpy as np
import torch

class ByteSensoryInterface:
    def __init__(self, device='cpu'):
        self.device = device
        
    def encode(self, byte_val, num_nodes):
        """
        One-hot encodes a byte (0-255) into a vector of size num_nodes.
        First 256 nodes are sensory.
        """
        vec = torch.zeros(num_nodes, device=self.device)
        # Ensure byte_val is in 0-255
        idx = int(byte_val) % 256
        vec[idx] = 1.0 
        return vec
        
    def decode(self, state_vector):
        """
        Decodes the first 256 nodes into a byte (0-255).
        Returns the byte value (int).
        """
        # Take first 256 nodes
        sensory_activity = state_vector[:256]
        # Argmax
        idx = torch.argmax(sensory_activity).item()
        return idx
        
    def decode_text(self, state_vector_batch):
        # Placeholder for batch decoding if needed
        pass
