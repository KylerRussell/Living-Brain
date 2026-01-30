
import numpy as np
import torch

class ByteSensoryInterface:
    def __init__(self, input_offset=0, output_offset=256, device='cpu'):
        self.device = device
        self.input_offset = input_offset
        self.output_offset = output_offset
        
    def encode(self, byte_val, num_nodes):
        """
        One-hot encodes a byte (0-255) into a vector of size num_nodes.
        Writes to [input_offset : input_offset+256].
        """
        vec = torch.zeros(num_nodes, device=self.device)
        # Ensure byte_val is in 0-255
        idx = (int(byte_val) % 256) + self.input_offset
        if idx < num_nodes:
            vec[idx] = 1.0 
        return vec
        
    def decode(self, state_vector):
        """
        Decodes nodes [output_offset : output_offset+256].
        Returns the byte value (int) relative to the block.
        """
        # Take the output block
        # Ensure we don't go out of bounds
        end_idx = self.output_offset + 256
        if end_idx > state_vector.shape[0]:
             # Fallback or error, but let's just slice safely
             end_idx = state_vector.shape[0]
             
        sensory_activity = state_vector[self.output_offset : end_idx]
        
        if sensory_activity.numel() == 0:
            return 0 # Default if out of bounds
            
        # Argmax
        idx = torch.argmax(sensory_activity).item()
        return idx
        
    def decode_text(self, state_vector_batch):
        # Placeholder for batch decoding if needed
        pass
