
import torch
import numpy as np
import os
import time
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import PredictiveCodingEngine
from ndcd.curriculum_gen import generate_chars, generate_toddler_words, generate_quotes

def ensure_data(data_path, phase_name):
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found. Generating for {phase_name}...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if "level1" in data_path: generate_chars(data_path)
        elif "level2" in data_path: generate_toddler_words(data_path)
        elif "level3" in data_path: generate_quotes(data_path)
        elif "sherlock" in data_path:
             import urllib.request
             url = "https://www.gutenberg.org/files/1661/1661-0.txt"
             try:
                 urllib.request.urlretrieve(url, data_path)
             except Exception as e:
                 print(f"Failed to download Sherlock: {e}")

class SequentialTrainer:
    def __init__(self, num_nodes=50000, device='cpu', num_modules=50):
        self.device = device
        self.num_nodes = num_nodes
        if num_nodes < 512:
            raise ValueError(f"num_nodes ({num_nodes}) must be >= 512 to support 256 input + 256 output nodes.")

        # 1. Initialize Hierarchical Modular Graph
        print("Initializing Hierarchical Modular Graph...")
        self.graph = DynamicGraph(
            num_nodes=num_nodes,
            m_edges=20,  # Legacy param, unused in modular topology
            p_triad=0.1,
            seed=42,
            num_modules=num_modules,
            num_levels=4,
        )
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus

        self.indices = indices
        self.initial_values = values

        # Input scale factor (fixed to 1.0)
        self.input_scale_factor = 1.0

        # Spectral Radius Tuning
        self.tune_spectral_radius(target_radius=0.95)

        # 2. Initialize Predictive Coding Engine
        # dt=0.5: IMEX is stable for large dt; converges in 10-20 steps
        module_ranges = self.graph.get_module_ranges()
        module_levels = self.graph.module_levels
        hier_pairs = self.graph.hier_pairs

        self.engine = PredictiveCodingEngine(
            num_nodes,
            indices,
            self.initial_values,
            biases,
            taus,
            module_ranges=module_ranges,
            module_levels=module_levels,
            hier_pairs=hier_pairs,
            positions=self.graph.pos,
            dt=0.5,
            device=device,
            temporal_alpha=0.5,
        )

        # 3. Define I/O Masks
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))

        # Pre-compute One-Hot Identity Matrices
        self.eye = torch.eye(256, device=device)

    def tune_spectral_radius(self, target_radius=0.95):
        """Tunes the spectral radius of the weight matrix to a target value."""
        print(f"Tuning Spectral Radius to {target_radius:.2f}...")

        if hasattr(self, 'engine'):
            w_tensor = self.engine.weight_values.cpu().numpy()
            indices = self.engine.indices.cpu().numpy()
        else:
            w_tensor = self.initial_values
            indices = self.indices

        row = indices[0]
        col = indices[1]
        w_sparse = sp.csr_matrix((w_tensor, (row, col)), shape=(self.num_nodes, self.num_nodes))

        try:
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            print(f"Current Spectral Radius: {max_eig:.4f}")

            scale_factor = target_radius / (max_eig + 1e-8)
            w_tensor = w_tensor * scale_factor
            print(f"Scaled weights by {scale_factor:.4f}")

            self.input_scale_factor = 1.0

            if hasattr(self, 'engine'):
                self.engine.weight_values = torch.tensor(w_tensor, dtype=torch.float32, device=self.device)
            else:
                self.initial_values = w_tensor

        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}). Using default.")

    def train_phase(self, phase_name, data_path, iterations, steps_per_iter, lr=0.01):
        """
        Predictive Coding training loop.

        For each token:
        1. Clamp input byte at level-0 input nodes
        2. Store previous state for temporal prediction
        3. Settle dynamics (single phase, 10-20 IMEX steps)
        4. Compute prediction errors (spatial + temporal)
        5. Update weights using LOCAL Hebbian rule
        6. Update short-term plasticity (from Phase 1)
        7. Log metrics

        No nudging, no beta, no three-phase settling.
        """
        print(f"\n=== Starting Phase: {phase_name} ===")
        print(f"Run started at: {time.ctime()}")
        ensure_data(data_path, phase_name)

        if not os.path.exists(data_path):
            print(f"Skipping {phase_name} (Data missing)")
            return

        with open(data_path, 'rb') as f:
            data = f.read()

        data_len = len(data)
        curr_idx = 0

        start_time = time.time()

        total_steps = iterations * steps_per_iter

        loss_accum = 0.0
        energy_accum = 0.0
        acc_window = []
        acc_top3_window = []

        for step in range(total_steps):
            # 1. Get Data Stream
            if curr_idx >= data_len - 1:
                curr_idx = 0

            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1

            # 2. Input Setup
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[input_byte] = 5.0 * self.input_scale_factor

            # 3. Store previous state for temporal prediction
            self.engine.store_previous_state()

            # 4. Reset fast nodes; slow nodes carry context
            fast_mask = self.engine.taus < 0.5
            self.engine.state[fast_mask] *= 0.1

            # 5. Single-phase settle (IMEX, ~20 steps)
            self.engine.settle(
                input_vec,
                input_mask=input_mask,
                max_steps=20,
                tol=1e-3,
            )

            # 6. Compute prediction errors
            energy = self.engine.compute_prediction_errors()
            energy_accum += energy

            # 7. Measure Prediction (before weight update)
            output_activity = torch.tanh(self.engine.state[256:512])
            probs = torch.softmax(output_activity * 10.0, dim=0)
            pred_idx = torch.argmax(probs).item()

            state_norm = torch.norm(self.engine.state) / np.sqrt(self.num_nodes)

            is_correct = (pred_idx == target_byte)
            acc_window.append(1.0 if is_correct else 0.0)

            _, top3_indices = torch.topk(probs, 3)
            if target_byte in top3_indices.tolist():
                acc_top3_window.append(1.0)
            else:
                acc_top3_window.append(0.0)

            loss = -torch.log(probs[target_byte] + 1e-8).item()
            loss_accum += loss

            # 8. Update weights using local predictive coding rule
            # Cosine LR schedule
            lr_mult = 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
            effective_lr = lr * max(lr_mult, 0.1)
            self.engine.update_weights_predictive(learning_rate=effective_lr)

            # 9. VICReg Regularization — per-module to avoid OOM
            # Global covariance over ~49K hidden nodes would be 49K×49K ≈ 9GB.
            # Per-module covariance is ~1000×1000 ≈ 4MB each.
            hidden_act = torch.tanh(self.engine.state[512:])
            if not hasattr(self, '_vicreg_buffer'):
                self._vicreg_buffer = []
            self._vicreg_buffer.append(hidden_act.detach())
            if len(self._vicreg_buffer) >= 32:
                batch = torch.stack(self._vicreg_buffer)  # [32, num_hidden]
                vicreg_lr = 0.001

                for mod_idx, (start, end) in enumerate(self.engine.module_ranges):
                    if end <= 512:
                        continue
                    mod_batch = batch[:, start - 512:end - 512]

                    std = torch.sqrt(mod_batch.var(dim=0) + 1e-4)
                    var_loss = torch.relu(1.0 - std).mean()

                    mod_centered = mod_batch - mod_batch.mean(dim=0)
                    cov = (mod_centered.T @ mod_centered) / (mod_batch.shape[0] - 1)
                    cov_loss = cov.fill_diagonal_(0).pow(2).sum() / mod_batch.shape[1]

                    with torch.no_grad():
                        self.engine.biases[start:end] -= vicreg_lr * (var_loss + cov_loss)

                self._vicreg_buffer = []

            # 10. Logging
            if step % 100 == 0:
                if len(acc_window) > 1000: acc_window = acc_window[-1000:]
                if len(acc_top3_window) > 1000: acc_top3_window = acc_top3_window[-1000:]

                elapsed = time.time() - start_time

                if step % 1000 == 0:
                    acc_1k = sum(acc_window) / len(acc_window) if acc_window else 0.0
                    acc3_1k = sum(acc_top3_window) / len(acc_top3_window) if acc_top3_window else 0.0
                    avg_loss = loss_accum / max(step, 1)
                    avg_energy = energy_accum / max(step, 1)

                    # Per-level error breakdown
                    level_errors = self.engine.get_prediction_error_by_level()
                    err_str = " | ".join(
                        f"L{l}: s={d['spatial']:.4f} t={d['temporal']:.4f}"
                        for l, d in sorted(level_errors.items())
                    )

                    print(f"Step {step}/{total_steps} | Time: {elapsed:.0f}s | "
                          f"AvgLoss: {avg_loss:.4f} | Energy: {avg_energy:.4f} | "
                          f"Acc@1k: {acc_1k:.2%} | Top3@1k: {acc3_1k:.2%} | "
                          f"||s||/√N: {state_norm:.4f}")
                    print(f"  PredErr: {err_str}")
                else:
                    recent_acc = sum(acc_window[-100:]) / min(len(acc_window), 100)
                    recent_acc3 = sum(acc_top3_window[-100:]) / min(len(acc_top3_window), 100)
                    print(f"  step {step}/{total_steps} | Loss: {loss:.4f} | "
                          f"Energy: {energy:.4f} | Acc: {recent_acc:.2%} | "
                          f"Top3: {recent_acc3:.2%} | ||s||/√N: {state_norm:.4f}        ", end='\r')

        final_acc = sum(acc_window)/len(acc_window) if len(acc_window) > 0 else 0.0
        final_acc3 = sum(acc_top3_window)/len(acc_top3_window) if len(acc_top3_window) > 0 else 0.0
        print(f"\nPhase Complete. Avg Loss: {loss_accum/total_steps:.4f} | "
              f"Avg Energy: {energy_accum/total_steps:.4f} | Final Acc: {final_acc:.2%}")

    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        curr_text = start_text

        # Prime
        for char in start_text:
            val = ord(char)
            if val > 255: val = 0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[val] = 1.0 * self.input_scale_factor
            self.engine.settle(input_vec, max_steps=10)

        for _ in range(length):
            state = self.engine.state
            out_act = torch.tanh(state[256:512])
            probs = torch.softmax(out_act * 10.0, dim=0)

            next_byte = torch.multinomial(probs, 1).item()
            char = chr(next_byte) if 0 <= next_byte < 128 else '?'
            curr_text += char

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[next_byte] = 1.0 * self.input_scale_factor
            self.engine.settle(input_vec, max_steps=10)

        print(curr_text)
        print("--------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nodes", type=int, default=50000)
    parser.add_argument("--modules", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if torch.backends.mps.is_available() and device == 'cpu':
        device = 'mps'
    if torch.cuda.is_available() and device == 'cpu':
        device = 'cuda'

    print(f"Using device: {device}")

    trainer = SequentialTrainer(num_nodes=args.nodes, device=device, num_modules=args.modules)

    # Reset state before curriculum begins
    trainer.engine.state.zero_()

    # Phase 1: Chars
    # lr=0.05: compensates for tiny activation products
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt",
                        iterations=500, steps_per_iter=100, lr=0.05)
    trainer.generate(start_text="A")

    # Phase 2: Words
    trainer.train_phase("Words", "ndcd/data/level2_words.txt",
                        iterations=200, steps_per_iter=100, lr=0.05)
    trainer.generate()

    # Phase 3: Quotes
    trainer.train_phase("Quotes", "ndcd/data/level3_quotes.txt",
                        iterations=200, steps_per_iter=200, lr=0.05)
    trainer.generate()

    # Phase 4: Literature
    trainer.train_phase("Literature", "ndcd/data/sherlock.txt",
                        iterations=500, steps_per_iter=500, lr=0.05)
    trainer.generate(start_text="Sherlock", length=200)

if __name__ == "__main__":
    main()
