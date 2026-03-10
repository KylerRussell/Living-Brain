import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import os
import stat
import tempfile
import shutil

# Create a GCC wrapper to suppress Triton's main.c compilation warnings
_system_cc = os.environ.get("CC", shutil.which("gcc") or shutil.which("clang") or "gcc")
_wrapper_fd, _wrapper_path = tempfile.mkstemp(prefix="cc_wrapper_", suffix=".sh")
with os.fdopen(_wrapper_fd, "w") as _f:
    _f.write(f"#!/bin/bash\nexec {_system_cc} -w -Wno-builtin-macro-redefined \"$@\"\n")
os.chmod(_wrapper_path, os.stat(_wrapper_path).st_mode | stat.S_IEXEC)
os.environ["CC"] = _wrapper_path

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
    def __init__(self, data_path, seq_len=128):
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
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor) and getattr(module.weight, 'requires_grad', False):
            param = module.weight
            state = optimizer.state.get(param)
            
            if state is not None:
                fisher = state.get('fisher_ema', state.get('exp_avg_sq'))
                if fisher is not None:
                    importance = fisher * (param.data ** 2)
                
                # Custom pruning to avoid PyTorch forward hooks that break CUDA graphs
                threshold = torch.quantile(importance.view(-1).float(), prune_ratio)
                mask = (importance >= threshold).float()
                
                # Apply to parameters and Adam states
                param.data.mul_(mask)
                state['exp_avg'].mul_(mask)
                state['exp_avg_sq'].mul_(mask)
                
                # Save mask to enforce it without hooks
                if not hasattr(module, 'fisher_mask'):
                    module.register_buffer('fisher_mask', mask)
                else:
                    module.fisher_mask.mul_(mask)

class SequentialBPTTTrainer:
    def __init__(self, hidden_dim=512, num_layers=3, device='cuda'):
        self.device = device
        self.model = MultiCompartmentSTSPNet(vocab_size=256, hidden_dim=hidden_dim, num_layers=num_layers, sparsity_alpha=0.15).to(device)
        
        # JIT compilation for reduced overhead and fused kernels (training only)
        # Dynamic sequence lengths in generate() will use uncompiled self.model to avoid recompilation cache misses.
        # Removed mode="reduce-overhead" to prevent CUDA graph capture memory explosion during unrolling
        self.compiled_model = torch.compile(self.model)
        
        # BPTT Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=8e-4, weight_decay=0.0)
        self.criterion = nn.CrossEntropyLoss()
        
        self.sleep_interval = 500  # Trigger sleep phase every N steps
        self.pruning_ratio_per_sleep = 0.0
        
        self.saliency_ema = 0.0 # Track saliency history
        
        
    def train_phase(self, phase_name, data_path, epochs=1, batch_size=8, seq_len=128):
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
                
                if step == 0 and epoch == 0:
                    print("\n[INFO] Triggering first forward pass. PyTorch Inductor is now compiling the computational graph.")
                    print("[INFO] This process (JIT compilation) can take several minutes and appear stalled. Please wait...")
                    compile_t0 = time.time()
                
                progress = step / total_steps
                logits, pred_loss, _, layer_acts_stacked = self.compiled_model(x, progress=progress)
                
                if step == 0 and epoch == 0:
                    print(f"[INFO] Compilation finished successfully in {time.time() - compile_t0:.2f} seconds! Training is now running at full speed.\n")
                
                # Cross Entropy over sequence
                loss_ce = self.criterion(logits.view(-1, 256), y.view(-1))
                
                # Total Loss incorporates layer-wise predictive coding
                # Use log-scale to govern explosion
                loss = loss_ce + 0.001 * torch.log1p(pred_loss)
                
                # Quick batch accuracy check for tracking
                with torch.no_grad():
                    batch_acc = (logits.argmax(dim=-1) == y).float().mean().item()
                self.current_acc = 0.9 * getattr(self, 'current_acc', 0.0) + 0.1 * batch_acc
                
                # Neuromodulation (Saliency-Gated Learning)
                # Scale learning rate by prediction error (surprise)
                base_lr = 8e-4
                if getattr(self, 'current_acc', 0.0) < 0.15:
                    raw_saliency = torch.clamp(pred_loss.detach() * 1.5, min=0.3, max=2.0).item()
                else:
                    raw_saliency = torch.clamp(torch.exp(pred_loss.detach()) - 0.5, min=0.3, max=5.0).item()
                    
                self.saliency_ema = 0.9 * self.saliency_ema + 0.1 * raw_saliency
                saliency = raw_saliency / (1.0 + 0.5 * max(0, self.saliency_ema - 2.0))
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr * saliency
                    # Adaptive Weight Decay: prune non-essential connections when bored
                    param_group['weight_decay'] = 1e-5 if saliency < 1.0 else 0.0
                
                loss.backward()
                
                # Fisher-Gated Metaplasticity (Synaptic Importance)
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            state = self.optimizer.state.get(param)
                            if state and 'exp_avg_sq' in state:
                                current_fisher = state['exp_avg_sq']
                                if 'fisher_ema' not in state:
                                    state['fisher_ema'] = torch.zeros_like(param.data)
                                state['fisher_ema'] = 0.90 * state['fisher_ema'] + 0.10 * current_fisher
                                importance = state['fisher_ema'] * (param.data ** 2)
                                param.grad.div_(importance.clamp(min=1.0))
                                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # Enforce manual Fisher pruning masks
                with torch.no_grad():
                    for module in self.model.modules():
                        if hasattr(module, 'fisher_mask'):
                            module.weight.mul_(module.fisher_mask)

                # Constrain parameters (e.g., temporal taus via sigmoid, Dale's law handled in forward)
                
                # Apply Intrinsic Plasticity (IP) outside autograd graph
                progress = step / total_steps
                ip_lr_scale = 1.0 - 0.9 * progress # Decays from 1.0 to 0.1
                self.model.apply_intrinsic_plasticity(layer_acts_stacked, seq_len=seq_len, target_rate=0.04, lr=0.0001 * ip_lr_scale)
                
                if step % 10 == 0 or step == 1:
                    pred = logits.argmax(dim=-1)
                    acc = (pred == y).float().mean().item()
                    elapsed = time.time() - start_time
                    log_str = f"Step {step}/{total_steps} | Time: {elapsed:.0f}s | Loss: {loss.item():.4f} | CE: {loss_ce.item():.4f} | PredLoss: {pred_loss.item():.4f} | Acc: {acc:.2%}"
                    print(log_str)
                    with open("ndcd/training_log.txt", "a") as f:
                        f.write(log_str + "\n")
                
                # --- Sleep Phase (Offline Consolidation) ---
                if step > 0 and step % self.sleep_interval == 0:
                    print("\n--- Initiating Sleep Phase ---")
                    # Global power-law downscaling (SHY)
                    # Skip sleep homeostasis to prevent weight shocks during unstable early learning.
                    # self.model.apply_sleep_homeostasis(eta_sleep=0.05)
                    
                    # Generative Replay (Active Attractor Consolidation)
                    self.model.eval()
                    with torch.no_grad():
                        dummy_x = torch.randint(0, 256, (batch_size, 1), device=self.device)
                        replay_seq = [dummy_x]
                        
                        # Generate a short sequence (length 32)
                        for _ in range(32):
                            logits_step, _, _, _ = self.model(torch.cat(replay_seq, dim=1))
                            next_token_logits = logits_step[:, -1, :]
                            probs = torch.softmax(next_token_logits / 1.0, dim=-1)
                            
                            # Fallback if probabilities contain NaN or negative values
                            if torch.isnan(probs).any() or (probs < 0).any():
                                probs = torch.ones_like(probs) / probs.size(-1)
                                
                            next_token = torch.multinomial(probs, 1)
                            replay_seq.append(next_token)
                            
                        replay_x = torch.cat(replay_seq[:-1], dim=1) # (batch, 32)
                        replay_y = torch.cat(replay_seq[1:], dim=1)  # (batch, 32)
                    self.model.train()
                    
                    # Contrastive Generative Replay: Generate noise
                    noise_x = torch.randint(0, 256, (batch_size, 31), device=self.device)
                    
                    # Consolidate on generated sequences with a very low learning rate
                    self.optimizer.zero_grad()
                    
                    # Dream pass
                    logits_replay, pred_loss_replay, _, _ = self.model(replay_x) # Use uncompiled for dynamic seq len
                    loss_ce_replay = self.criterion(logits_replay.view(-1, 256), replay_y.view(-1))
                    
                    # Contrastive Noise pass
                    logits_noise, _, _, _ = self.model(noise_x)
                    probs_noise = torch.softmax(logits_noise, dim=-1)
                    # Maximize entropy on random noise
                    loss_entropy_noise = (probs_noise * torch.log(probs_noise + 1e-9)).sum(dim=-1).mean()
                    
                    if loss_ce_replay.item() > 3.5:
                        print(f"  [Dream Filter] Rejecting dream sequence with CE: {loss_ce_replay.item():.4f} > 3.5")
                    else:
                        # Minimize CE on dream, Maximize entropy on noise
                        loss_replay = loss_ce_replay + 0.1 * pred_loss_replay + 0.1 * loss_entropy_noise
                        loss_replay.backward()
                        
                        # Metaplastic Anchor: Fisher-Gated updates during sleep
                        with torch.no_grad():
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    state = self.optimizer.state.get(param)
                                    if state and 'exp_avg_sq' in state:
                                        current_fisher = state['exp_avg_sq']
                                        if 'fisher_ema' not in state:
                                            state['fisher_ema'] = torch.zeros_like(param.data)
                                        state['fisher_ema'] = 0.90 * state['fisher_ema'] + 0.10 * current_fisher
                                        importance = state['fisher_ema'] * (param.data ** 2)
                                        param.grad.div_(importance.clamp(min=1.0))
                        
                        # Use a specifically small consolidation LR
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = 1e-5
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        
                        print(f"  Generative Replay Loss: {loss_replay.item():.4f}")
                        
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
                x = torch.tensor([ord(c) if ord(c) < 256 else 0 for c in curr_text[-128:]], device=device).unsqueeze(0)
                logits, _, _, _ = self.model(x)
                
                # Get last token prediction
                next_logit = logits[0, -1, :]
                # Repetition penalty (within the generation loop)
                if 'prev_byte' in locals():
                    logits[0, -1, prev_byte] -= 2.0 # Penalty for repeating the exact same byte
                
                # Use 0.7 to sharpen the distribution (closer to 0.0 is more "greedy")
                probs = torch.softmax(next_logit / 0.7, dim=0)
                
                next_byte = torch.multinomial(probs, 1).item()
                prev_byte = next_byte
                
                char = chr(next_byte) if 0 <= next_byte < 128 else '?'
                curr_text += char
                
        print(curr_text)
        print("--------------------------------------")
        with open("ndcd/training_log.txt", "a") as f:
            f.write(f"\n--- Generating: {start_text} ... ---\n")
            f.write(curr_text + "\n")
            f.write("--------------------------------------\n")
            
        self.model.train()

def main():
    # Clear the old log file
    with open("ndcd/training_log.txt", "w") as f:
        f.write("=== Living-Brain NDCD Training Log ===\n")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Using device: {args.device}")

    trainer = SequentialBPTTTrainer(hidden_dim=512, num_layers=3, device=args.device)

    # Phase 1: Holophrases
    trainer.train_phase("Holophrases", "ndcd/data/level1_holophrases.txt", epochs=20, seq_len=128)
    trainer.generate(start_text="L")

    # Phase 2: Slot-and-Frame
    trainer.train_phase("Slot-and-Frame", "ndcd/data/level2_slot_frame.txt", epochs=20, seq_len=128)
    trainer.generate(start_text="W")

    # Phase 3: Complex Constructions
    trainer.train_phase("Complex Constructions", "ndcd/data/level3_complex.txt", epochs=20, seq_len=128)
    trainer.generate(start_text="I")

    # Phase 4: Contextual Continuity
    trainer.train_phase("Contextual Continuity", "ndcd/data/level4_contextual.txt", epochs=20, seq_len=128)
    trainer.generate(start_text="I", length=200)

if __name__ == "__main__":
    main()