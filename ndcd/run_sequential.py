import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import os
import time
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ndcd.engine_torch import MultiCompartmentSTSPNet

from ndcd.curriculum_gen import (
    generate_holophrases,
    generate_slot_and_frame,
    generate_complex_constructions,
    generate_contextual_continuity
)

def ensure_data(data_path, phase_name):
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found. Generating for {phase_name}...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if "level1" in data_path: generate_holophrases(data_path)
        elif "level2" in data_path: generate_slot_and_frame(data_path)
        elif "level3" in data_path: generate_complex_constructions(data_path)
        elif "level4" in data_path: generate_contextual_continuity(data_path)

class ByteSequenceDataset(Dataset):
    def __init__(self, data_path, seq_len=64):
        with open(data_path, 'rb') as f:
            self.data = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).copy()).long()
        self.seq_len = seq_len
        self.length = len(self.data) - seq_len
        
    def __len__(self):
        return self.length // self.seq_len
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y

def apply_fisher_pruning(model, optimizer, prune_ratio=0.01):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor) and module.weight.requires_grad:
            param = module.weight_orig if hasattr(module, 'weight_orig') else module.weight
            state = optimizer.state.get(param)
            
            if state is not None and 'exp_avg_sq' in state:
                fisher = state['exp_avg_sq']
                importance = fisher * (param.data ** 2)
                
                # Prune least important synapses based on Fisher information
                prune.l1_unstructured(module, name='weight', amount=prune_ratio, importance_scores=importance)

class SequentialBPTTTrainer:
    def __init__(self, hidden_dim=256, num_layers=3, device='cuda'):
        self.device = device
        self.model = MultiCompartmentSTSPNet(vocab_size=256, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        
        # JIT compilation for reduced overhead and fused kernels (training only)
        # Dynamic sequence lengths in generate() will use uncompiled self.model to avoid recompilation cache misses.
        self.compiled_model = torch.compile(self.model, mode="reduce-overhead")
        
        # BPTT Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.sleep_interval = 500  # Trigger sleep phase every N steps
        self.pruning_ratio_per_sleep = 0.05
        
    def train_phase(self, phase_name, data_path, epochs=1, batch_size=32, seq_len=64):
        print(f"\n=== Starting Phase: {phase_name} ===")
        ensure_data(data_path, phase_name)
        
        dataset = ByteSequenceDataset(data_path, seq_len=seq_len)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        
        self.model.train()
        step = 0
        total_steps = len(loader) * epochs
        start_time = time.time()
        
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # --- Wake Phase (Online Learning) ---
                self.optimizer.zero_grad()
                
                logits, pred_loss, _, layer_acts_stacked = self.compiled_model(x)
                
                # Cross Entropy over sequence
                loss_ce = self.criterion(logits.view(-1, 256), y.view(-1))
                
                # Total Loss incorporates layer-wise predictive coding
                loss = loss_ce + 0.1 * pred_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                # Constrain parameters (e.g., temporal taus via sigmoid, Dale's law handled in forward)
                
                # Apply Intrinsic Plasticity (IP) outside autograd graph
                self.model.apply_intrinsic_plasticity(layer_acts_stacked, seq_len=seq_len, target_rate=0.1, lr=0.01)
                
                if step % 100 == 0:
                    pred = logits.argmax(dim=-1)
                    acc = (pred == y).float().mean().item()
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{total_steps} | Time: {elapsed:.0f}s | Loss: {loss.item():.4f} | CE: {loss_ce.item():.4f} | PredLoss: {pred_loss.item():.4f} | Acc: {acc:.2%}")
                
                # --- Sleep Phase (Offline Consolidation) ---
                if step > 0 and step % self.sleep_interval == 0:
                    print("\n--- Initiating Sleep Phase ---")
                    # Global power-law downscaling (SHY)
                    self.model.apply_sleep_homeostasis(eta_sleep=0.05)
                    
                    # Generative Replay (Input Noise)
                    with torch.no_grad():
                        dummy_x = torch.randint(0, 256, (batch_size, seq_len), device=self.device)
                        self.model(dummy_x, use_sleep_noise=True) # Let network dream and settle
                        
                    # Target redundant synapses with Fisher Pruning
                    apply_fisher_pruning(self.model, self.optimizer, prune_ratio=self.pruning_ratio_per_sleep)
                    pruning_percentage = self.pruning_ratio_per_sleep * 100
                    print(f"  Fisher Pruning: Masked lowest {pruning_percentage:.1f}% of Fisher-importance synapses.")
                    print("--- Waking Up ---\n")
                    
                step += 1

    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        self.model.eval()
        
        curr_text = start_text
        device = self.device
        
        with torch.no_grad():
            for _ in range(length):
                x = torch.tensor([ord(c) if ord(c) < 256 else 0 for c in curr_text[-64:]], device=device).unsqueeze(0)
                logits, _, _, _ = self.model(x)
                
                # Get last token prediction
                next_logit = logits[0, -1, :]
                probs = torch.softmax(next_logit, dim=0)
                
                next_byte = torch.multinomial(probs, 1).item()
                char = chr(next_byte) if 0 <= next_byte < 128 else '?'
                curr_text += char
                
        print(curr_text)
        print("--------------------------------------")
        self.model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Using device: {args.device}")

    trainer = SequentialBPTTTrainer(hidden_dim=256, num_layers=3, device=args.device)

    # Phase 1: Holophrases
    trainer.train_phase("Holophrases", "ndcd/data/level1_holophrases.txt", epochs=2)
    trainer.generate(start_text="L")

    # Phase 2: Slot-and-Frame
    trainer.train_phase("Slot-and-Frame", "ndcd/data/level2_slot_frame.txt", epochs=2)
    trainer.generate(start_text="W")

    # Phase 3: Complex Constructions
    trainer.train_phase("Complex Constructions", "ndcd/data/level3_complex.txt", epochs=2)
    trainer.generate(start_text="I")

    # Phase 4: Contextual Continuity
    trainer.train_phase("Contextual Continuity", "ndcd/data/level4_contextual.txt", epochs=2)
    trainer.generate(start_text="I", length=200)

if __name__ == "__main__":
    main()