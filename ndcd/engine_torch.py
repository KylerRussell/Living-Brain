import numpy as np
import torch
from typing import Optional, Tuple, List


@torch.jit.script
def get_soma(v_m: torch.Tensor, i_exc: torch.Tensor, i_inh: torch.Tensor, a: torch.Tensor, cahva: torch.Tensor, 
             is_neg_pe: torch.Tensor, threshold: torch.Tensor, is_sst: torch.Tensor, is_pv: torch.Tensor, 
             is_dg: torch.Tensor, ip_gain: torch.Tensor, ip_bias: torch.Tensor, nmda_ratio: torch.Tensor,
             apical_beta: torch.Tensor) -> torch.Tensor:
    # Solution 2: State-Locked NMDAR Bistability
    # Ties magnesium block to current membrane potential (v_m)
    v_m_scaled = v_m * 100.0
    mg_block = 1.0 / (1.0 + (1.2 / 3.57) * torch.exp(-0.062 * v_m_scaled))
    
    # Fast AIS kinetics for PV interneurons (lower threshold)
    effective_threshold = torch.where(is_pv, threshold * 0.7, threshold)
    
    # Proposal 2: Shunting Inhibition (Divisive Gain Control)
    i_soma_base = (i_exc * ip_gain + ip_bias) / (1.0 + i_inh)
    
    # NMDA/AMPA blend (Phase 2)
    # AMPA component (1-ratio) has no mg_block, NMDA component (ratio) is blocked
    nmda_drive_scale = (1.0 - nmda_ratio) + nmda_ratio * mg_block
    base_drive = torch.tanh(i_soma_base) * nmda_drive_scale
    
    # Proposal 3: Two-Compartment Coincidence Logic (BAC Firing)
    # Sharp gate (gain=15.0, threshold=7.5) for precise coincident detection
    somatic_spike_detected = torch.sigmoid(base_drive * 15.0 - 7.5)
    
    # Non-linear Apical XOR Logic (wider gate)
    # Scaled by apical_beta (per-module sensitivity)
    apical_gate_fast = torch.sigmoid(a * apical_beta * 8.0 - 3.0)  # Opens at ~0.37
    apical_gate_slow = torch.sigmoid(a * apical_beta * 12.0 - 10.0) # Closes at ~0.83
    apical_xor_logic = apical_gate_fast - 0.4 * apical_gate_slow # Relaxed XOR
    
    # Coincidence detection gating
    bac_burst_trigger = somatic_spike_detected * apical_xor_logic
    
    # burst(coincidence) + high-voltage calcium plateaus (CaHVA)
    # Increased burst amp to 25.0 for clearer diagnostic signatures
    burst_amp = 1.0 + 25.0 * bac_burst_trigger * (1.1 + cahva)
    
    # Omission signaling (PE-)
    omission_diff = torch.clamp(a - i_exc, min=0.0)
    omission_drive = torch.sigmoid(omission_diff * 12.0 - 6.0) * is_neg_pe.float() * 10.0
    
    # Final somatic drive
    i_drive = (base_drive * burst_amp + omission_drive)
    
    # Phase 4: Bidirectional Prediction Error Wiring
    npe_base = (a * ip_gain + ip_bias) / (1.0 + i_exc.clamp(min=0.0) + i_inh)
    npe_drive = torch.tanh(npe_base) * 3.0 + omission_drive
    
    ppe_drive = base_drive * burst_amp
    i_drive = torch.where(is_neg_pe, npe_drive, ppe_drive)
    
    # Supralinear Activation (n=2 for PV, n=3 for DG)
    pos_drive = torch.clamp(i_drive - effective_threshold, min=0.0)
    
    if is_pv.any() or is_dg.any():
        exponent = torch.ones_like(i_drive)
        exponent = torch.where(is_pv, torch.tensor(2.0, device=i_drive.device), exponent)
        exponent = torch.where(is_dg, torch.tensor(3.0, device=i_drive.device), exponent)
        
        p_out = pos_drive ** exponent
        s_out = torch.tanh(i_drive / effective_threshold.clamp(min=0.1)) + pos_drive * 0.1
        return torch.where(is_pv | is_dg, p_out, s_out)
    
    i_final = i_drive / effective_threshold.clamp(min=0.1)
    return torch.tanh(i_final) + pos_drive * 0.1

@torch.jit.script
def jit_solve_dynamics_imex(
    initial_basal: torch.Tensor,
    initial_apical: torch.Tensor,
    weight_values: torch.Tensor,
    indices: torch.Tensor,
    basal_intra_mask: torch.Tensor,
    basal_inter_mask: torch.Tensor,
    apical_intra_mask: torch.Tensor,
    apical_inter_mask: torch.Tensor,
    biases: torch.Tensor,
    taus: torch.Tensor,
    input_vector: torch.Tensor,
    dt: float,
    max_steps: int,
    tolerance: float,
    input_mask: Optional[torch.Tensor],
    module_starts: torch.Tensor,
    module_ends: torch.Tensor,
    is_pv: torch.Tensor,
    is_sst: torch.Tensor,
    is_vip: torch.Tensor,
    is_lts: torch.Tensor,
    is_inhibitory: torch.Tensor,
    is_l23: torch.Tensor,
    is_l56: torch.Tensor,
    is_neg_pe: torch.Tensor,
    is_dg: torch.Tensor,
    activation_ema: torch.Tensor,
    ip_gain: torch.Tensor,
    ip_bias: torch.Tensor,
    node_levels: torch.Tensor,
    sparsity_alpha: torch.Tensor,
    sfa_states: torch.Tensor,
    cahva_states: torch.Tensor,
    rho_slow_states: torch.Tensor,
    ais_distance: torch.Tensor,
    threshold_adaptation: torch.Tensor,
    inh_depression_soma: torch.Tensor,
    inh_depression_dend: torch.Tensor,
    u_facil: torch.Tensor,
    x_depress: torch.Tensor,
    nmda_ratio: torch.Tensor,
    tau_facil: float,
    tau_depress: float,
    U0: float,
    ee_mask: torch.Tensor,
    gap_junction_indices: torch.Tensor,
    gap_junction_weights: torch.Tensor,
    apical_bandpass_state1: torch.Tensor,
    apical_bandpass_state2: torch.Tensor,
    lifetime_firing: torch.Tensor,
    g_gap: float = 0.5,
    damping: float = 0.15,
    implicit_damping: float = 1.2,
    sigma_noise: float = 0.05,
    apical_beta: torch.Tensor = torch.ones(1), # Default to scalar 1.0 if not provided
    calcium_store: torch.Tensor = torch.zeros(1),
    purkinje_mask: torch.Tensor = torch.zeros(1, dtype=torch.bool),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Semi-implicit (IMEX) dynamics solver for Multi-Compartment Predictive Coding.

    Neurons now have segregated compartments:
    - Basal: integrates feedforward (bottom-up + lateral) signals (AMPA-like)
    - Apical: Feedback feedback (top-down) with non-linear quadratic integration.
    - CaHVA: High-Voltage-Activated Calcium plateau for BAC firing
    - Soma (output): Bi-compartment coincidence detector with coincidence gating.
    - Adaptive Threshold: Homeostatic scaling via diffuse messenger (Nitric Oxide proxy).
    - Interneurons: Diversified PV+ (fast, perisomatic) and SOM+ (slow, dendritic).

    Returns (basal, apical, somatic, diff, steps, dt, sfa, cahva, rho_slow, threshold, ais, inh_soma, inh_dend, u, x, ip_gain, ip_bias, state1, state2, lifetime_firing, nmda_ratio)
    """
    # Physical Constants for Conductance Model
    E_L: float = -0.5   # Leak reversal (normalized slightly higher to prevent silence)
    E_E: float = 1.2    # Exc reversal (extra drive)
    E_I: float = -1.2   # Inh reversal (extra suppression)
    C_m: float = 1.0    # Membrane capacitance
    G_L: float = 0.02   # Leak conductance (reduced for integration)
    num_nodes = initial_basal.size(0)
    # Initial states
    current_u = u_facil.clone()
    current_x = x_depress.clone()
    current_b = initial_basal.clone()
    current_a = initial_apical.clone()
    
    # Adaptive threshold initialization
    current_threshold = threshold_adaptation.clone()
    current_inh_s = inh_depression_soma.clone()
    current_inh_d = inh_depression_dend.clone()
    current_ais = ais_distance.clone()
    current_ip_gain = ip_gain.clone() # NEW
    current_ip_bias = ip_bias.clone() # NEW
    current_bp1 = apical_bandpass_state1.clone()
    current_bp2 = apical_bandpass_state2.clone()
    current_lifetime = lifetime_firing.clone()

    # Proposal 1: Dynamic Thresholds with Diffusive IP
    # Δθ_i = η_IP * ( (ν_i - ν_target) + κ * Σ (ν_j - ν_target) )
    with torch.no_grad():
        eta_ip = 0.015 # η_IP
        kappa = 0.4    # κ (diffusive term)
        for m in range(module_starts.size(0)):
            ms = module_starts[m].item()
            me = module_ends[m].item()
            
            # ν_i - ν_target
            individual_error = activation_ema[ms:me] - sparsity_alpha[ms:me]
            # neighborhood term: module average error
            local_avg_error = individual_error.mean()
            
            # Update firing threshold θ_i
            d_theta = eta_ip * (individual_error + kappa * local_avg_error)
            current_threshold[ms:me] = torch.clamp(current_threshold[ms:me] + d_theta, 0.5, 5.0)
            
            # Proposal 5: Structural Heterogeneity (LTS Units)
            # LTS units have lower thresholds and higher sensitivity
            lts_mask_mod = is_lts[ms:me]
            if lts_mask_mod.any():
                current_threshold[ms:me][lts_mask_mod] *= 0.98

    indices_basal_intra = indices[:, basal_intra_mask]
    indices_basal_inter = indices[:, basal_inter_mask]
    indices_apical_intra = indices[:, apical_intra_mask]
    indices_apical_inter = indices[:, apical_inter_mask]
    
    weights_basal_intra = weight_values[basal_intra_mask]
    weights_basal_inter = weight_values[basal_inter_mask]
    weights_apical_intra = weight_values[apical_intra_mask]
    weights_apical_inter = weight_values[apical_inter_mask]

    # Pre-split E and I weights
    src_inh_basal_intra = is_inhibitory[indices_basal_intra[0]].float()
    src_inh_basal_inter = is_inhibitory[indices_basal_inter[0]].float()
    src_inh_apical_intra = is_inhibitory[indices_apical_intra[0]].float()
    src_inh_apical_inter = is_inhibitory[indices_apical_inter[0]].float()
    
    w_basal_intra_E = torch.sparse_coo_tensor(indices_basal_intra, weights_basal_intra * (1.0 - src_inh_basal_intra), (num_nodes, num_nodes))
    w_basal_intra_I = torch.sparse_coo_tensor(indices_basal_intra, weights_basal_intra * src_inh_basal_intra, (num_nodes, num_nodes))
    
    # Basal Inter weights also contribute to E and I conductances
    w_basal_inter_E = torch.sparse_coo_tensor(indices_basal_inter, weights_basal_inter * (1.0 - src_inh_basal_inter), (num_nodes, num_nodes))
    w_basal_inter_I = torch.sparse_coo_tensor(indices_basal_inter, weights_basal_inter * src_inh_basal_inter, (num_nodes, num_nodes))
    
    # Apical weights
    w_apical_intra_E = torch.sparse_coo_tensor(indices_apical_intra, weights_apical_intra * (1.0 - src_inh_apical_intra), (num_nodes, num_nodes))
    w_apical_intra_I = torch.sparse_coo_tensor(indices_apical_intra, weights_apical_intra * src_inh_apical_intra, (num_nodes, num_nodes))
    
    w_apical_inter_E = torch.sparse_coo_tensor(indices_apical_inter, weights_apical_inter * (1.0 - src_inh_apical_inter), (num_nodes, num_nodes))
    w_apical_inter_I = torch.sparse_coo_tensor(indices_apical_inter, weights_apical_inter * src_inh_apical_inter, (num_nodes, num_nodes))

    # Sparse versions for SR control
    w_basal_intra_sparse = torch.sparse_coo_tensor(indices_basal_intra, weights_basal_intra, (num_nodes, num_nodes))
    w_basal_inter_sparse = torch.sparse_coo_tensor(indices_basal_inter, weights_basal_inter, (num_nodes, num_nodes))
    w_apical_intra_sparse = torch.sparse_coo_tensor(indices_apical_intra, weights_apical_intra, (num_nodes, num_nodes))
    w_apical_inter_sparse = torch.sparse_coo_tensor(indices_apical_inter, weights_apical_inter, (num_nodes, num_nodes))

    current_dt = dt
    dynamic_damping = implicit_damping 
    imex_denom = 1.0 + dynamic_damping * current_dt / taus
    
    # --- Item 1: Apical Hierarchical Scaling ---
    # Feedforward is fast, feedback/apical is slow (NMDA-like)
    # Target tau_apical range: 50-100ms.
    # We use a narrower gradient than basal tau to preserve prediction precision.
    apical_tau_mult = 2.0 + 2.0 * node_levels.float() # L0 -> 2x (50ms), L3 -> 8x (200ms)
    imex_denom_apical = 1.0 + dynamic_damping * current_dt / (taus * apical_tau_mult)

    min_dt: float = 0.05

    # Pre-split E and I weights
    src_inh_basal_intra = is_inhibitory[indices_basal_intra[0]].float()
    src_inh_basal_inter = is_inhibitory[indices_basal_inter[0]].float()
    src_inh_apical_intra = is_inhibitory[indices_apical_intra[0]].float()
    src_inh_apical_inter = is_inhibitory[indices_apical_inter[0]].float()
    
    w_basal_intra_E = torch.sparse_coo_tensor(indices_basal_intra, weights_basal_intra * (1.0 - src_inh_basal_intra), (num_nodes, num_nodes))
    w_basal_intra_I = torch.sparse_coo_tensor(indices_basal_intra, weights_basal_intra * src_inh_basal_intra, (num_nodes, num_nodes))
    
    w_basal_inter_E = torch.sparse_coo_tensor(indices_basal_inter, weights_basal_inter * (1.0 - src_inh_basal_inter), (num_nodes, num_nodes))
    w_basal_inter_I = torch.sparse_coo_tensor(indices_basal_inter, weights_basal_inter * src_inh_basal_inter, (num_nodes, num_nodes))
    
    w_apical_intra_E = torch.sparse_coo_tensor(indices_apical_intra, weights_apical_intra * (1.0 - src_inh_apical_intra), (num_nodes, num_nodes))
    w_apical_intra_I = torch.sparse_coo_tensor(indices_apical_intra, weights_apical_intra * src_inh_apical_intra, (num_nodes, num_nodes))
    
    w_apical_inter_E = torch.sparse_coo_tensor(indices_apical_inter, weights_apical_inter * (1.0 - src_inh_apical_inter), (num_nodes, num_nodes))
    w_apical_inter_I = torch.sparse_coo_tensor(indices_apical_inter, weights_apical_inter * src_inh_apical_inter, (num_nodes, num_nodes))
    
    step_count = 0
    diff = tolerance + 1.0
    prev_diff: float = 1e6

    # Initial conductances
    i_exc = torch.zeros_like(current_b)
    i_inh = torch.zeros_like(current_b)
    
    current_s = get_soma(current_b, i_exc, i_inh, current_a, cahva_states, is_neg_pe, current_threshold, is_sst, is_pv, is_dg, current_ip_gain, current_ip_bias, nmda_ratio, apical_beta)

    while step_count < max_steps and diff > tolerance:
        # Update PV/SST interneuron activity for somatic/dendritic inhibition
        # 1. VIP interneurons driven by feedback (apical) inhibit SOM/PV cells.
        vip_drive = torch.relu(current_a).mean() if current_a.any() else torch.tensor(0.0, device=current_b.device)
        vip_activity = torch.sigmoid(vip_drive * 5.0 - 2.0)
        
        # 2. SOM+ Martinotti: slow, target apical dendrites.
        sst_activity = torch.relu(current_s[is_sst]).mean() if is_sst.any() else torch.tensor(0.0, device=current_b.device)
        conscience_factor = 1.0 + 0.5 * current_lifetime[is_sst].mean() if is_sst.any() else torch.tensor(1.0, device=current_b.device)
        sst_drive = (sst_activity * conscience_factor) / (1.0 + 10.0 * vip_activity)
        
        # 3. PV+ Basket: fast, perisomatic (basal targeting).
        pv_activity = torch.relu(current_s[is_pv]).mean() if is_pv.any() else torch.tensor(0.0, device=current_b.device)
        pv_drive = pv_activity / (1.0 + 5.0 * vip_activity)

        # Hard clamp input nodes
        if input_mask is not None:
            current_b = current_b * (1.0 - input_mask) + input_vector * input_mask
            current_s = get_soma(current_b, i_exc, i_inh, current_a, cahva_states, is_neg_pe, current_threshold, is_sst, is_pv, is_dg, current_ip_gain, current_ip_bias, nmda_ratio, apical_beta)


        old_soma = current_s.clone()

        # Proposal 2: Shunting Inhibition in Activation Function
        rho = get_soma(current_b, i_exc, i_inh, current_a, cahva_states, is_neg_pe, current_threshold, is_sst, is_pv, is_dg, current_ip_gain, current_ip_bias, nmda_ratio, apical_beta)
        
        # Add fluctuation-driven noise at the somatic output level (from Phase 1)
        # This preserves AI-state variability during dynamics without biasing diagnostics.
        threshold_gap = (current_threshold - rho.abs()).clamp(min=0.05)
        rho = rho + sigma_noise * threshold_gap * torch.randn_like(rho)

        
        # Proposal 2: Soft Recurrent Competition (E→PV→E feedback loop)
        # Instead of hard Top-K, we rely on the power-law PV+ activation
        # and conductance-based inhibition calculated later in the loop.
        # This allows for truly emergent stochastic competition.
        for m in range(module_starts.size(0)):
            ms = module_starts[m].item()
            me = module_ends[m].item()
            mod_rho = rho[ms:me]
            # Use mean sparsity for the module since k-WTA is a population constraint
            target_k = max(1, int(torch.mean(sparsity_alpha[ms:me]).item() * (me - ms)))
            if target_k < mod_rho.size(0):
                top_acts, _ = torch.topk(mod_rho.abs(), target_k)
                soft_threshold = top_acts[-1] * 0.9  # Soft margin sharper
                rho[ms:me] = mod_rho * torch.sigmoid((mod_rho.abs() - soft_threshold) * 20.0)
        # Myelinated Temporal Filtering
        rho_slow_states = rho_slow_states + current_dt * (rho - rho_slow_states) / 10.0
        rho_unmyelinated = rho_slow_states + torch.randn_like(rho) * 0.05 * (rho - rho_slow_states).abs()
        rho_myelinated = rho

        # Phase 3: per-synapse STSP update (E-E recurrent connections)
        # u: facilitation (residual calcium), x: depression (vesical recovery)
        # We only update surface weights for speed
        pre_act = rho[indices[0]].abs()
        d_u = (U0 - current_u) / tau_facil + U0 * (1.0 - current_u) * pre_act
        d_x = (1.0 - current_x) / tau_depress - current_u * current_x * pre_act
        
        current_u = (current_u + current_dt * d_u * ee_mask).clamp(0.01, 1.0)
        current_x = (current_x + current_dt * d_x * ee_mask).clamp(0.01, 1.0)

        # Conductance-Based Synaptic Inputs with STSP modulation
        # We use scatter_add for per-synapse modulated sparse MV
        # Weight gain is normalized by U0 to ensure unit gain at rest (Murray 2014)
        u_slice = current_u[basal_intra_mask]
        x_slice = current_x[basal_intra_mask]
        eff_weights = weights_basal_intra * (u_slice / max(U0, 1e-3)) * x_slice
        i_exc_intra = torch.zeros(num_nodes, device=current_b.device)
        i_inh_intra = torch.zeros(num_nodes, device=current_b.device)
        
        # Basal Intra (STSP modulated)
        contrib_e = eff_weights * (1.0 - src_inh_basal_intra) * rho_unmyelinated[indices_basal_intra[0]]
        contrib_i = weights_basal_intra * src_inh_basal_intra * rho_unmyelinated[indices_basal_intra[0]]
        i_exc_intra.scatter_add_(0, indices_basal_intra[1], contrib_e)
        i_inh_intra.scatter_add_(0, indices_basal_intra[1], contrib_i.abs())
        
        # Basal Inter & External
        i_exc_raw = i_exc_intra + torch.mv(w_basal_inter_E, rho_myelinated) + input_vector
        i_inh_raw = i_inh_intra + torch.abs(torch.mv(w_basal_inter_I, rho_myelinated))
        
        # --- Solution 1: Conductance-Based E/I Tracking ---
        current_inh_s = current_inh_s + current_dt * (i_exc_raw.abs() - current_inh_s) / 10.0
        ei_tracking_scale = (current_inh_s + 0.1) / (i_inh_raw + 0.1)
        i_inh = i_inh_raw * ei_tracking_scale * 5.0 # Enforce g=5.0 ratio from Brunel AI regime
        
        # Add perisomatic PV shunting drive (Proposal 2 & 5)
        # We shunt based on PV activity
        i_inh = i_inh + pv_drive * 10.0
        
        # --- Item 3: DG Hilar Mossy Cell Loop (Disynaptic Inhibition) ---
        if is_dg.any():
            # Approximation: DG activity drives local hilar mossy cells which 
            # then drive interneurons that inhibit the DG population.
            dg_activity = current_s[is_dg].abs()
            # Scaling factor 5.0 creates aggressive decorrelation for pattern separation
            i_inh[is_dg] = i_inh[is_dg] + dg_activity * 5.0
            
        i_exc = i_exc_raw

        # Basal processing (conductance terms for membrane update)
        basal_E = i_exc
        basal_I = -i_inh
        
        # Apical processing (feedback / top-down) - Split E and I
        apical_E = torch.mv(w_apical_intra_E, rho_unmyelinated) + torch.mv(w_apical_inter_E, rho_myelinated)
        apical_I = torch.mv(w_apical_intra_I, rho_unmyelinated) + torch.mv(w_apical_inter_I, rho_myelinated)
        # Add SST dendritic inhibition to apical conductances
        apical_I = apical_I - sst_drive * current_inh_d * 5.0

        # Conductance-Based Integration (Item 1)
        # We treat synaptic inputs as conductances. 
        # g_E and g_I are non-negative transients.
        g_E_basal = torch.clamp(basal_E, min=0.0)
        g_I_basal = torch.clamp(-basal_I, min=0.0)
        
        g_E_apical = torch.clamp(apical_E, min=0.0)
        g_I_apical = torch.clamp(-apical_I, min=0.0)

        # Jacobian gain control (applied to conductances)
        drho = 1.0 - rho * rho
        state_norm = rho.norm().clamp(min=1e-6)
        
        # Basal SR control
        jac_basal = torch.mv(w_basal_intra_sparse, rho * drho) + torch.mv(w_basal_inter_sparse, rho * drho)
        eff_sr_basal = jac_basal.norm() / state_norm
        target_max_sr: float = 0.95
        if eff_sr_basal > target_max_sr:
            scale_b = target_max_sr / eff_sr_basal
            g_E_basal = g_E_basal * scale_b
            g_I_basal = g_I_basal * scale_b
            
        # Apical SR control
        jac_apical = torch.mv(w_apical_intra_sparse, rho * drho) + torch.mv(w_apical_inter_sparse, rho * drho)
        eff_sr_apical = jac_apical.norm() / state_norm
        if eff_sr_apical > target_max_sr:
            scale_a = target_max_sr / eff_sr_apical
            g_E_apical = g_E_apical * scale_a
            g_I_apical = g_I_apical * scale_a

        # Phase 3: per-synapse STSP is already applied during scatter_add
        # No further per-node resource scaling needed.
        
        # Basal drive: dV/dt = gL(EL-V) + gE(EE-V) + gI(EI-V)
        # Using normalized voltages where E_L=-0.7, E_E=1.0, E_I=-1.0
        # Solution 1: Fluctuation-Driven Stochasticity
        # Integrate an OU process noise term simulating background bombardment
        sigma_noise = 0.05
        # The equation expects a force term, standard stochastic integration is sigma * sqrt(dt) * xi
        # Since this `dv_basal` is later divided by `taus` and multiplied by `current_dt`,
        # we scale the noise by sqrt(2*taus/dt) to ensure steady-state variance is constant (sigma^2).
        ou_noise = sigma_noise * torch.randn_like(current_b) * torch.sqrt(2.0 * taus / current_dt)

        # Apply tau scaling to the integration
        dv_basal = (G_L * (E_L - current_b) + 
                    g_E_basal * (E_E - current_b) + 
                    g_I_basal * (E_I - current_b) + 
                    biases + input_vector + 
                    ou_noise) / taus
        
        # --- Item 1: Non-linear Sigmoidal Integration for Apical ---
        # Instead of pure quadratic, we use a saturating sigmoidal term to allow XOR logic
        # and frequency tuning (preventing runaway excitation).
        v_a_norm = current_a / 1.5
        sigmoidal_feedback = torch.sigmoid(v_a_norm * 10.0 - 5.0) - torch.sigmoid(v_a_norm * 15.0 - 10.0)
        
        dv_apical = (G_L * (E_L - current_a) + 
                     g_E_apical * (E_E - current_a) + 
                     g_I_apical * (E_I - current_a) +
                     2.0 * sigmoidal_feedback) / (taus * apical_tau_mult) # Non-linear XOR-capable term
        
        # --- Item 3: Diversified Interneuron Microcircuits (PV+/SOM+) ---
        # Note: Interneuron activity val calculation moved to top of loop for get_soma usage.
        
        # --- Item 2: AI State Stabilization ---

        # Long-term inhibition scaling (moving average tracking)
        # This replaces the fast per-node depletion with a slower homeostatic drive.
        current_inh_s = current_inh_s + current_dt * (g_E_basal.norm() - current_inh_s) / 500.0

        # Item 3: Gap-Junction Coupling (Electrical Synapses)
        # Guarantees that the inhibitory pool synchronizes its firing
        if gap_junction_indices.size(1) > 0:
            v_src = current_b[gap_junction_indices[0]]
            v_dst = current_b[gap_junction_indices[1]]
            i_gap_vals = gap_junction_weights * (v_dst - v_src)
            i_gap = torch.zeros_like(current_b)
            i_gap.scatter_add_(0, gap_junction_indices[0], i_gap_vals)
            dv_basal = dv_basal + g_gap * i_gap

        # dv_basal and dv_apical are now fully defined by conductances
        # no extra redundant terms


        # Semi-implicit compartment updates
        # Item 3: Hierarchical Timescale Gradients & Filtering
        # Apical: integrates feedback via a bandpass filter (3-6 Hz resonance)
        # We use a second-order system approximation.
        bp_omega = 2.0 * 3.14159 * 4.5 * taus / 1000.0 # ~4.5 Hz resonance, scaled by tau
        bp_zeta = 0.3 # Damping ratio
        
        # Update bandpass states
        d_bp1 = current_bp2
        d_bp2 = -2.0 * bp_zeta * bp_omega * current_bp2 - bp_omega**2 * current_bp1 + dv_apical
        
        current_bp1 = current_bp1 + current_dt * d_bp1
        current_bp2 = current_bp2 + current_dt * d_bp2
        
        # Item 3: Diversified Interneuron Dynamics
        # PV is fast (~2ms)
        pv_tau_mult = 0.1 # Fast PV (10% of standard tau -> ~1-2ms if tau=20)
        # Note: dv_basal and dv_apical already have tau factored in above
        current_b = (current_b + current_dt * (dv_basal * current_ais) / torch.where(is_pv, pv_tau_mult, 1.0)) / imex_denom
        current_a = current_a + current_dt * dv_apical # NMDA-like slow apical
        current_a = current_a.clamp(-1.5, 1.5)

        # BAC Firing CaHVA Plateau Update: Supralinear Regenerative Current (Item 2)
        # coincidence = somatic spike + apical input
        soma_spike = torch.sigmoid(current_b * 12.0 - 6.0)
        coincidence = soma_spike * torch.sigmoid(current_a * 10.0 - 5.0)
        nmda_plateau = torch.where(current_a > 0.7, torch.ones_like(current_a) * 1.0, torch.zeros_like(current_a))
        burst_trigger = coincidence + nmda_plateau
        
        cahva_tau = torch.ones_like(cahva_states) * 30.0
        cahva_tau[is_l56] = 100.0  # Biological: longer plateaus in L5 pyramidal
        cahva_tau[is_l23] = 20.0
        cahva_states = cahva_states + current_dt * (burst_trigger * 5.0 - cahva_states) / cahva_tau
        cahva_states = cahva_states.clamp(0.0, 3.0)

        # Spike-dependent threshold adaptation (Item 2)
        # Threshold increases when firing to maintain irregular state
        # Hover near critical state via fast spike-growth and slow decay
        current_threshold = current_threshold + current_dt * (1.0 - current_threshold) / 100.0
        current_threshold = torch.clamp(current_threshold + 0.25 * rho.abs() * current_dt, 0.8, 5.0)

        # Proposal 5: Spike-Frequency Adaptation for RS Units (AHP Current)
        is_rs = (~is_inhibitory)
        sfa_states = sfa_states + current_dt * (current_b.abs() - sfa_states) / taus
        current_b = torch.where(is_rs, current_b - 0.15 * (20.0 / taus) * sfa_states * current_b.sign(), current_b)

        # Clamp compartments
        current_b = current_b.clamp(-1.5, 1.5)
        current_a = current_a.clamp(-1.5, 1.5)
        
        # Update somatic state
        current_s = rho.clone()
        
        # Proposal 2: Hard-clamp input mask on soma if provided
        if input_mask is not None:
            current_s = current_s * (1.0 - input_mask) + input_vector * input_mask
        
        # --- Item 4: Intrinsic Purkinje Timing (mGluR7 simulation) ---
        # Purkinje cells integrate excitatory input into a calcium store.
        # When a threshold is crossed, the neuron "pauses" (shunting inhibition).
        if purkinje_mask.any():
            # Accumulate calcium based on excitatory input (i_exc)
            # Purkinje cells receive very high frequency input (up to 200Hz)
            # mGluR7 has a slow activation/decay constant (~50-100ms)
            d_calcium = (i_exc - calcium_store) / 20.0 # tau = 10ms for faster dynamics 
            calcium_store = torch.where(purkinje_mask, calcium_store + current_dt * d_calcium, calcium_store)
            
            # Pause Trigger: shunting inhibition if calcium > threshold
            pause_mask = purkinje_mask & (calcium_store > 1.8)
            # Apply shunting inhibition: drive potential toward -1.0 (pause state)
            current_b[pause_mask] = current_b[pause_mask] * 0.5 - 1.0
            
            # Reset calcium store during pause (refractory period)
            calcium_store[pause_mask] *= 0.8
            
        current_lifetime = 0.9999 * current_lifetime + 0.0001 * current_s.abs()
        diff = torch.norm(current_s - old_soma).item()

        if diff <= tolerance * 1.2:
            break

        if diff > prev_diff and current_dt > min_dt:
            current_dt = max(current_dt * 0.5, min_dt)
            imex_denom = 1.0 + dynamic_damping * current_dt / taus
        elif diff < prev_diff * 0.8 and current_dt < dt:
            current_dt = min(current_dt * 1.2, dt)
            imex_denom = 1.0 + dynamic_damping * current_dt / taus
            imex_denom_apical = 1.0 + dynamic_damping * current_dt / (taus * apical_tau_mult)

        prev_diff = diff
        step_count += 1

    return (current_b, current_a, current_s, diff, step_count, current_dt, 
            sfa_states, cahva_states, rho_slow_states, 
            current_threshold, current_ais, current_inh_s, current_inh_d, current_u, current_x,
            current_ip_gain, current_ip_bias, current_bp1, current_bp2, current_lifetime, nmda_ratio, calcium_store)


class PredictiveCodingEngine:
    """
    Hierarchical Predictive Coding engine replacing flat EqProp.

    Key changes from DragonEngineTorch:
    1. IMEX integration instead of RK4 (10-20 steps vs 100)
    2. Predictive coding errors instead of energy-based EqProp
    3. Per-module temporal prediction matrices
    4. Local Hebbian weight updates (no nudge/beta phases)
    5. Single settle phase per token (no free/pos/neg phases)
    """

    def __init__(self, num_nodes, indices, values, biases, taus,
                 module_ranges, module_levels, hier_pairs,
                 modules=None,
                 positions: Optional[np.ndarray] = None,
                 dt=0.5, device='cuda' if torch.cuda.is_available() else 'cpu',
                 temporal_alpha=0.5,
                 is_inhibitory: Optional[np.ndarray] = None,
                 is_pv: Optional[np.ndarray] = None,
                 is_sst: Optional[np.ndarray] = None,
                 is_vip: Optional[np.ndarray] = None,
                 is_lts: Optional[np.ndarray] = None,
                 dg_indices: Optional[np.ndarray] = None,
                 gc_indices: Optional[np.ndarray] = None,
                 purkinje_indices: Optional[np.ndarray] = None):
        """
        Args:
            num_nodes: Total number of nodes.
            indices: [2, E] edge index array.
            values: [E] edge weight array.
            biases: [N] bias vector.
            taus: [N] time constants.
            module_ranges: List of (start, end) for each module.
            module_levels: Array of level per module.
            hier_pairs: List of (upper_mod_id, lower_mod_id) pairs.
            positions: Optional [N, 3] spatial positions.
            dt: Integration timestep (0.5-1.0 for IMEX).
            device: Torch device.
            temporal_alpha: Balance between spatial and temporal prediction (0.5).
        """
        self.dt = dt
        self.device = device
        self.num_nodes = num_nodes
        self._initial_temporal_alpha = temporal_alpha # Store for tensorization later in __init__
        
        # Sparsity alpha: 10-15% for GCs (temporal basis) else 2% (Bio-Adam)
        self.sparsity_alpha = torch.ones(num_nodes, dtype=torch.float32, device=device) * 0.02
        if gc_indices is not None:
            gc_t = torch.tensor(gc_indices, dtype=torch.long, device=device)
            self.sparsity_alpha[gc_t] = 0.15 # Relaxed sparsity for temporal basis set

        # Spatial positions
        if positions is not None:
            self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        else:
            self.positions = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)

        # Sparse weights (COO)
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.num_edges = self.indices.shape[1]
        self.weight_values = torch.tensor(values, dtype=torch.float32, device=device)
        self.base_weight_values = self.weight_values.clone()
        self.w_surface = self.weight_values # Alias for training script
        self._weight_values_raw = self.weight_values # Alias for structural plasticity

        # Parameters
        self.taus = torch.tensor(taus, dtype=torch.float32, device=device)
        self.biases = torch.tensor(biases, dtype=torch.float32, device=device)
        self.initial_biases = self.biases.clone()
        
        # Phase 3: Intrinsic Purkinje cell timing (mGluR7)
        self.calcium_store = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.purkinje_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if purkinje_indices is not None:
             purk_t = torch.tensor(purkinje_indices, dtype=torch.long, device=device)
             self.purkinje_mask[purk_t] = True

        # State
        self.state_basal = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.state_apical = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.state = torch.zeros(num_nodes, dtype=torch.float32, device=device)  # somatic state
        self.sfa_states = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        
        # Output/Motor indices for nudging
        self.output_indices = torch.arange(256, 512, device=device) 
        self.output_edge_mask = (self.indices[1] >= 256) & (self.indices[1] < 512)
        self.apical_beta = torch.ones(num_nodes, dtype=torch.float32, device=device)
        self.cahva_states = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.rho_slow_states = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.ais_distance = torch.ones(num_nodes, dtype=torch.float32, device=device)
        
        # AI State & Stabilization Variables
        self.threshold_adaptation = torch.ones(num_nodes, dtype=torch.float32, device=device)
        self.inh_depression_soma = torch.ones(num_nodes, dtype=torch.float32, device=device) # PV-mediated
        self.inh_depression_dend = torch.ones(num_nodes, dtype=torch.float32, device=device) # SST-mediated


        # Intrinsic Plasticity (IP) Parameters (Item 1)
        self.ip_gain = torch.ones(num_nodes, dtype=torch.float32, device=device) * 1.0 # Initialized to 1.0
        self.ip_bias = torch.zeros(num_nodes, dtype=torch.float32, device=device) # Initialized to 0.0
        self.activation_ema = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.activation_var = torch.ones(num_nodes, dtype=torch.float32, device=device)
        self.burst_ema = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        
        # Bandpass Filter States for Apical (Item 3)
        self.apical_bp1 = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.apical_bp2 = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.lifetime_firing = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Context EMA buffer for cross-boundary context preservation
        self.context_ema = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.context_ema_alpha = 0.1  # Blend rate: 10% new, 90% old
        self.context_injection_weight = 0.5 # Excitatory feedback weight for persistent activity

        # Fisher Information Matrix (Diagonal Approximation) for Pruning
        self.fisher_diag = torch.ones(self.num_edges, dtype=torch.float32, device=device)
        self.w_max = 5.0 # For Log-STDP


        # Synaptic Correlation Stats for Sleep/Pruning (Item 4)
        self.corr_ema = torch.zeros(self.num_edges, dtype=torch.float32, device=device)
        self.corr_var = torch.ones(self.num_edges, dtype=torch.float32, device=device)

        # Module metadata
        self.module_ranges = module_ranges  # [(start, end), ...]
        self.module_levels = module_levels  # np array of levels
        self.hier_pairs = hier_pairs        # [(upper_id, lower_id), ...]
        self.num_modules = len(module_ranges)

        # Laminar Segregation Masks
        self.is_l4 = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.is_l23 = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.is_l56 = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.is_neg_pe = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if modules is not None:
            for mod in modules:
                self.is_l4[mod['l4_indices']] = True
                self.is_l23[mod['l23_indices']] = True
                self.is_l56[mod['l56_indices']] = True
                # Designate half of L2/3 as Negative Prediction Error neurons
                l23_idx = mod['l23_indices']
                n_neg = len(l23_idx) // 2
                self.is_neg_pe[l23_idx[:n_neg]] = True
        
        # DG Sparsity Mask
        self.is_dg = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if dg_indices is not None:
            self.is_dg[dg_indices] = True
            
        # Target Broca module (Level 0, Module 0 typically)
        if modules is not None and len(modules) > 0:
            self._broca_l56_indices = torch.tensor(modules[0]['l56_indices'], dtype=torch.long, device=device)
        else:
            self._broca_l56_indices = torch.tensor([], dtype=torch.long, device=device)

        # Build module-level lookup tensors for fast access
        self._build_module_tensors()

        # Dale's law assignment
        if is_inhibitory is not None:
            self.is_inhibitory = torch.tensor(is_inhibitory, dtype=torch.bool, device=device)
        else:
            self.is_inhibitory = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
        # Tripartite Interneurons
        if is_pv is not None:
            self.is_pv = torch.tensor(is_pv, dtype=torch.bool, device=device)
        else:
            self.is_pv = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
        if is_sst is not None:
            self.is_sst = torch.tensor(is_sst, dtype=torch.bool, device=device)
        else:
            self.is_sst = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
        if is_vip is not None:
            self.is_vip = torch.tensor(is_vip, dtype=torch.bool, device=device)
        else:
            self.is_vip = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
        if is_lts is not None:
            self.is_lts = torch.tensor(is_lts, dtype=torch.bool, device=device)
        else:
            # Default: half of SST interneurons are LTS
            self.is_lts = self.is_sst.clone()

        # --- Top-down edge mask for proper predictive coding ---
        # In hierarchical predictive coding, the spatial prediction error at
        # level ℓ = x_ℓ − f(W_topdown * x_{ℓ+1})
        # We must isolate top-down edges (higher→lower level) from the full
        # weight matrix. Using all synaptic input (intra-module, lateral,
        # bottom-up) gives the dynamics residual, not a prediction error.
        self.node_to_level = torch.zeros(num_nodes, dtype=torch.long, device=device)
        self.node_to_level[:512] = -1  # I/O nodes conceptually at level -1
        for mod_idx, (start, end) in enumerate(self.module_ranges):
            self.node_to_level[start:end] = self.module_levels[mod_idx]
        self.max_level = int(self.node_to_level.max().item())
        
        # Phase 4: Hippocampal Preordering (temporal_alpha boost)
        self.temporal_alpha = torch.ones(num_nodes, dtype=torch.float32, device=device) * self._initial_temporal_alpha
        # Level 2 (Hippocampal CA3/CA1) get high temporal weighting for preordering
        hippo_mask = (self.node_to_level == 2)
        self.temporal_alpha[hippo_mask] = 0.85

        # Hierarchical Temporal Gradients (Murray 2014, Chaudhuri 2015)
        # τ_E = 20 ms, τ_I = 10 ms (Standard AI regime)
        self.taus = torch.where(self.is_inhibitory, torch.tensor(10.0, device=device), torch.tensor(20.0, device=device))
        
        # Scaling INTs across hierarchy: L0 -> (Murray MT 70ms), L3 -> (Murray ACC 350ms)
        # We use a linear gradient for the base time constant scaling
        self.tau_scale = 1.0 + 4.0 * (self.node_to_level.float().clamp(min=0.0) / max(1, self.max_level))
        self.taus *= self.tau_scale
        
        # Grade NMDA/AMPA ratio (Phase 2): 0.4 (L0) to 0.65 (L3)
        self.nmda_ratio = 0.4 + 0.25 * (self.node_to_level.float().clamp(min=0.0) / max(1, self.max_level))

        # Per-edge level lookups for top-down, bottom-up, and lateral masks
        src_levels = self.node_to_level[self.indices[0]]
        dst_levels = self.node_to_level[self.indices[1]]

        # Item 4: Recurrent Scaling
        # Increase the weight of recurrent excitatory connections (W_EE) linearly with level
        # This increases topic persistence in higher association areas (L2-L3)
        lateral_recurrent_mask = (src_levels == dst_levels) & (src_levels >= 0) & (~self.is_inhibitory[self.indices[0]])
        self.weight_values[lateral_recurrent_mask] *= (1.0 + src_levels[lateral_recurrent_mask].float())

        # Top-down: source at strictly higher level than destination
        self.topdown_edge_mask = src_levels > dst_levels
        self.topdown_indices = self.indices[:, self.topdown_edge_mask]

        # Bottom-up: source at strictly lower level than destination
        self.bottomup_edge_mask = src_levels < dst_levels

        # Lateral: same level, excluding I/O nodes (which are at level 0
        # but aren't association nodes)
        self.lateral_edge_mask = (
            (src_levels == dst_levels) &
            (self.indices[0] >= 512) &
            (self.indices[1] >= 512)
        )

        # Intra/Inter module masks for myelinated temporal filtering
        src_modules = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        for mod_idx, (start, end) in enumerate(self.module_ranges):
            src_modules[start:end] = mod_idx
        idx_src_mod = src_modules[self.indices[0]]
        idx_dst_mod = src_modules[self.indices[1]]
        self.intra_edge_mask = (idx_src_mod == idx_dst_mod) & (idx_src_mod != -1)
        self.inter_edge_mask = ~self.intra_edge_mask

        # Partition edges into compartments and module-locality (Step 2)
        # Basal: lateral + bottom-up
        # Apical: feedback top-down
        basal_mask = ~self.topdown_edge_mask
        apical_mask = self.topdown_edge_mask

        self.basal_intra_mask = basal_mask & self.intra_edge_mask
        self.basal_inter_mask = basal_mask & self.inter_edge_mask
        self.apical_intra_mask = apical_mask & self.intra_edge_mask
        self.apical_inter_mask = apical_mask & self.inter_edge_mask

        self.indices_basal_intra = self.indices[:, self.basal_intra_mask]
        self.indices_basal_inter = self.indices[:, self.basal_inter_mask]
        self.indices_apical_intra = self.indices[:, self.apical_intra_mask]
        self.indices_apical_inter = self.indices[:, self.apical_inter_mask]

        # Fix 4 removed: Uncoupled Product Feedback Alignment (PFA)
        # Weights are left completely asymmetric.


        # --- Free-edge mask for spectral radius enforcement ---
        # I/O projection edges (source or dest < 512) are external forcing,
        # not autonomous recurrence. They are 6.5x boosted and constitute
        # ~39% of edges. Including them in spectral radius estimation
        # inflates the measured SR from ~0.57 (free edges) to ~29 (full
        # matrix), causing enforce_spectral_radius to multiply all learned
        # weights by 0.95/29 ≈ 0.033 every 100 steps — zeroing them out.
        # This mask matches graph.py's initialization, which tunes SR on
        # free edges only.
        self.free_edge_mask = (self.indices[0] >= 512) & (self.indices[1] >= 512)
        self.free_edge_indices = self.indices[:, self.free_edge_mask]

        # --- Temporal prediction state ---
        # Previous state buffer per module (for temporal prediction errors)
        self.previous_state = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Temporal transition matrices A_ℓ per module
        # Stored as diagonal approximation for efficiency:
        # A_ℓ is a vector of size module_size (diagonal of the full matrix)
        # Full matrix would be module_size x module_size but too expensive
        self.temporal_A = []
        for start, end in self.module_ranges:
            size = end - start
            # Initialize near identity (predict persistence)
            a = torch.ones(size, dtype=torch.float32, device=device) * 0.9
            a += torch.randn(size, dtype=torch.float32, device=device) * 0.05
            self.temporal_A.append(a)

        # --- Metaplastic cascade: 3 timescales per synapse ---
        # Surface: fast, updated every step
        # Mid: medium, τ ≈ 100 steps
        # Deep: slow, τ ≈ 10000 steps
        self.w_deep = torch.zeros_like(self.weight_values)
        self.w_surface = torch.zeros_like(self.weight_values)
        self.w_mid = torch.zeros_like(self.weight_values)

        # Cascade transfer rates — significantly increased to allow transient
        # syntactic rules to meaningfully accumulate in surface weights.
        self.tau_surface_to_mid = 2000.0   # was 1000.0; slower drain lets surface accumulate
        
        # We now track separate mid-to-deep transfer rates per edge depending on 
        # whether the source node belongs to a hippocampal or neocortical module.
        self.tau_mid_to_deep = torch.ones_like(self.weight_values) * 1000.0

        # Metaplastic scaling: how much accumulated deep weight
        # reduces surface learning rate. Reduced from 1.0 to 0.1 because
        # w_deep ≈ 0.83 after chars was causing 1.83x LR reduction (with
        # omega adding another ~10x). At 0.1, the cascade inertia of
        # w_deep already protects important weights without also killing
        # the effective learning rate.
        self.meta_scale = 0.1  # was 0.25; reduce inertia so surface LR isn't crushed by w_deep

        # FIX 1: Track the initial w_deep Frobenius norm as a target.
        # This is the SR-tuned initialization; effective weights should
        # never exceed ~2x this norm during training.
        # CHANGED: Compute on free edges only, matching graph.py's tuning.
        self._initial_deep_frob = self.weight_values[self.free_edge_mask].norm().item()

        # --- Synaptic intelligence (Zenke et al., 2017) ---
        self.omega = torch.zeros_like(self.weight_values)       # accumulated importance
        self.prev_weights = self.effective_weights.clone()       # for computing Δw per step
        self.si_baseline_weights = self.effective_weights.clone() # for computing total Δw over epoch
        self.running_contribution = torch.zeros_like(self.weight_values)  # path integral
        self.si_damping = 0.1  # prevents omega from growing unboundedly

        # Tsodyks-Markram STSP (Phase 3)
        # u: facilitation (residual calcium), x: depression (vesicle availability)
        # These operate on the ms-to-seconds timescale for activity-silent memory.
        self.u_facilitation = torch.ones_like(self.weight_values) * 0.2
        self.x_depression = torch.ones_like(self.weight_values)
        self.tau_facil = 1500.0   # τ_F ≈ 1500ms
        self.tau_depress = 200.0  # τ_D ≈ 200ms
        self.U0 = 0.2             # Inherent release probability

        # Store latest prediction errors for weight updates
        self.spatial_errors = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.temporal_errors = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Convergence tracking for training gate (Fix 4)
        self.last_settle_diff = 0.0
        self.topdown_pred_var = 0.0

        # Thalamic Gate tracking
        self.surprise_ema = 0.0

        # Homeostatic Synaptic Scaling (HSS)
        self.calcium_traces = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.calcium_target = 0.1
        self.tau_calcium = 1000.0
        self.hss_rho = 0.001
        # Three-Factor Learning (Eligibility Traces)
        self.eligibility_traces = torch.zeros_like(self.weight_values)
        self.tau_eligibility = 2.0  # Short-term memory of local coincidence

        # --- Item 3: Gap-Junction Coupling for PV Interneurons ---
        # Electrical synapses connect PV nodes within the same module for synchronization.
        pv_idx = torch.where(self.is_pv)[0].cpu().numpy()
        gap_indices = []
        for start_m, end_m in self.module_ranges:
            mod_pv = pv_idx[(pv_idx >= start_m) & (pv_idx < end_m)]
            if len(mod_pv) > 1:
                for i in range(len(mod_pv)):
                    for j in range(i + 1, len(mod_pv)):
                        gap_indices.append([mod_pv[i], mod_pv[j]])
                        gap_indices.append([mod_pv[j], mod_pv[i]])
        
        if gap_indices:
            self.gap_junction_indices = torch.tensor(gap_indices, dtype=torch.long, device=device).t()
            self.gap_junction_weights = torch.ones(self.gap_junction_indices.size(1), dtype=torch.float32, device=device) * 0.5
        else:
            self.gap_junction_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
            self.gap_junction_weights = torch.zeros(0, dtype=torch.float32, device=device)
        self.eligibility_traces = torch.zeros_like(self.weight_values)
        self.tau_eligibility = 2.0  # Short-term memory of local coincidence

    def _build_module_tensors(self):
        """Pre-build tensors for module start/end ranges for fast slicing."""
        self.mod_starts = torch.tensor(
            [r[0] for r in self.module_ranges], dtype=torch.long, device=self.device)
        self.mod_ends = torch.tensor(
            [r[1] for r in self.module_ranges], dtype=torch.long, device=self.device)
        self.mod_level_tensor = torch.tensor(
            self.module_levels, dtype=torch.long, device=self.device)

    def activation_function(self, s):
        return torch.tanh(s) + 0.01 * s
        
    @property
    def effective_weights(self):
        """Effective weight is base topology + learned cascade deltas."""
        return self.weight_values + self.w_surface + self.w_mid + self.w_deep

    def cascade_transfer(self, include_deep=True, surface_floor=0.01):
        """Call periodically (every ~500 steps) after weight update.

        Transfers weight magnitude downward through the cascade:
        surface → mid (above floor) and optionally mid → deep (gated).

        The surface floor ensures that fast transient weights always retain
        a minimum magnitude for hierarchical signal propagation. Without
        this, continuous proportional drain drives TD/BU surface weights
        to zero, making higher levels input-invariant.

        Args:
            include_deep: If False, only surface→mid transfer occurs.
            surface_floor: Minimum surface magnitude to retain (default 0.003).
        """
        # Surface → Mid: Transfer only excess above floor
        surface_abs = self.w_surface.abs()
        excess_mask = surface_abs > surface_floor
        if excess_mask.any():
            excess = (surface_abs - surface_floor) * self.w_surface.sign()
            # Transfer 20% of excess per call (called every ~500 steps)
            transfer_sm = torch.zeros_like(self.w_surface)
            transfer_sm[excess_mask] = excess[excess_mask] * 0.2
            self.w_mid += transfer_sm
            self.w_surface -= transfer_sm

        # Mid → Deep: Continuous leaky integration if enabled
        if include_deep:
            gate_mask = self.w_mid.abs() > 0.005  # lowered from 0.02
            transfer_md = torch.zeros_like(self.w_mid)
            transfer_md[gate_mask] = self.w_mid[gate_mask] / self.tau_mid_to_deep[gate_mask]
            self.w_deep += transfer_md
            self.w_mid -= transfer_md

        # Gentle w_deep norm control (unchanged)
        deep_frob = self.w_deep[self.free_edge_mask].norm().item()
        max_deep_frob = self._initial_deep_frob
        if deep_frob > max_deep_frob:
            self.w_deep[self.free_edge_mask] *= max_deep_frob / deep_frob

    def update_synaptic_intelligence(self, current_loss=0.0):
        """Call after each weight update to track intrinsic activity correlation for pruning.

        Computes the Fisher Information Matrix (FIM) diagonal approximation.
        Synapses that carry high-variance information about the output distribution
        are preserved, while those that don't fluctuate with meaningful signals are targeted.
        """
        rho = torch.tanh(self.state)
        idx_i = self.indices[0]
        idx_j = self.indices[1]
        
        # Apply Detonator Synapses Boost (DG -> CA3)
        # These variables (edge_rows, edge_cols, dg_indices, ca3_indices, weight_vals)
        # are assumed to be available in the scope where this function is called,
        # or are attributes of 'self' that are not explicitly defined in the provided context.
        # For this edit, we assume they are available as per the instruction.
        # If they are not, this would lead to a NameError.
        # dg_mask = np.isin(edge_rows, self.dg_indices)
        # ca3_mask = np.isin(edge_cols, self.ca3_indices)
        # detonator_mask = dg_mask & ca3_mask
        # # Mossy Fibers are extremely powerful ("detonators")
        # weight_vals[detonator_mask] *= 30.0

        # Activity correlation metric |pre * post|
        activity_correlation = (rho[idx_i] * rho[idx_j])
        
        if not hasattr(self, 'corr_ema'):
            self.corr_ema = torch.zeros_like(self.weight_values)
            self.corr_var = torch.ones_like(self.weight_values)
            
        # Update mean and variance (Fisher Information Proxy)
        self.corr_ema = 0.99 * self.corr_ema + 0.01 * activity_correlation
        self.corr_var = 0.99 * self.corr_var + 0.01 * (activity_correlation - self.corr_ema).pow(2)
        
        # Fisher Information Proxy: Activity-Dependent Importance
        # Item 6: Importance is the squared covariance of pre-post correlation
        # This highlights synapses with high-variance, non-zero signaling.
        self.fisher_diag = self.corr_var * self.corr_ema.abs().clamp(min=1e-3)
        
        self.running_contribution += activity_correlation.pow(2) # Accumulate squared correlation
        self.prev_weights = self.effective_weights.clone()

    def consolidate_importance(self):
        """Call at phase boundaries or every ~10K steps.

        Transfers running contribution to permanent importance (omega).
        """
        # delta_w_total MUST be relative to the start of the consolidation epoch,
        # otherwise we are dividing 10K steps of accumulated contribution by
        # the tiny delta_w of a single step, which explodes omega near infinity.
        delta_w_total = self.effective_weights - self.si_baseline_weights
        
        # Normalize by total weight change to get per-unit importance
        self.omega += torch.relu(self.running_contribution) / (delta_w_total.pow(2) + 1e-6)

        # Decay old importance slowly to allow forgetting truly obsolete knowledge
        self.omega *= (1.0 - self.si_damping)

        # Reset accumulator and baseline
        self.running_contribution.zero_()
        self.si_baseline_weights = self.effective_weights.clone()

    def offline_renormalization(self, num_replay_cycles=20, lambd=0.8, theta_memory=0.1):
        """
        Simulates slow-wave sleep (0.5–4 Hz) for Synaptic Homeostasis (SHY).
        
        Phase 1: Spontaneous Reactivation & Relative Potentiation tracking.
        Phase 2: Multiplicative Downscaling with Protection Threshold.
        Phase 3: CaMKIV Intrinsic Plasticity Rescue.
        """
        with torch.no_grad():
            free_mask = self.free_edge_mask
            
            # --- Phase 1: Spontaneous Reactivation (Replay) ---
            # Track co-activity during sleep replay for "Relative Potentiation" rule.
            # Synapses that are co-active during replay are exempt from downscaling.
            co_activity = torch.zeros(self.num_edges, device=self.device)
            for _ in range(num_replay_cycles):
                # Drive with noise to trigger attractor replay
                noise = torch.randn(self.num_nodes, device=self.device) * 0.1
                self.settle(noise, max_steps=10)
                rho = torch.tanh(self.state)
                # Local Hebbian co-activity trace
                co_activity += (rho[self.indices[0]] * rho[self.indices[1]]).abs()

            if co_activity.max() > 0:
                co_activity /= co_activity.max()
            
            # Phase 5: Multiplicative Scaling with Importance Protection (SHY)
            # α ≈ 0.85
            alpha = 0.85
            # Protection factor κ * ω: High-importance synapses are protected
            protection = torch.sigmoid(self.omega)
            
            # Replay-based protection (Bartram et al. 2017)
            # Synapses that were co-active during replay are spared
            replay_safe = (co_activity > 0.1).float()
            
            # Effective scaling factor per synapse: w_new = w * [α + (1-α) * max(prot, replay)]
            protective_mask = torch.max(protection, replay_safe)
            scaling_factor = alpha + (1.0 - alpha) * protective_mask
            
            with torch.no_grad():
                # Surface/volatile weights are more susceptible to downscaling
                # Deep/consolidated weights are more resistant
                surface_scaling = scaling_factor * 0.9 # Extra decay for surface noise
                deep_scaling = 1.0 - (1.0 - scaling_factor) * 0.5 # 50% decay resist for deep
                
                self.w_surface *= surface_scaling
                self.w_mid *= (surface_scaling + deep_scaling) / 2.0
                self.w_deep *= deep_scaling
                self.weight_values *= deep_scaling
                
                # Threshold elimination to create bimodal distribution (margin widening)
                theta_min = 0.005
                prune_mask = self.effective_weights.abs() < theta_min
                self.weight_values[prune_mask] = 0.0
                self.w_surface[prune_mask] = 0.0
                self.w_mid[prune_mask] = 0.0
                self.w_deep[prune_mask] = 0.0
            
            # Aggressive transient clearing for surface and mid removed (handled by targeted scaling)
            # self.w_mid[free_mask] *= 0.05
            # self.w_surface[free_mask] *= 0.05
            
            # --- Phase 3: CaMKIV Intrinsic Plasticity Rescue ---
            # Reset stabilization states FIRST
            self.threshold_adaptation.fill_(1.0)
            self.inh_depression_soma.fill_(1.0)
            self.inh_depression_dend.fill_(1.0)
            self.ais_distance.fill_(1.0)
            
            # Rescues quiescent neurons by lowering threshold and increasing leak resistance.
            quiescent_mask = (self.activation_ema < (self.sparsity_alpha * 0.1))
            if quiescent_mask.any():
                self.threshold_adaptation[quiescent_mask] *= 0.5
                self.ais_distance[quiescent_mask] *= 1.5
                self.ip_bias[quiescent_mask] += 0.5
            
            # Reset remaining states
            self.activation_ema.zero_()
            self.burst_ema.zero_()
            self.ip_gain.fill_(1.0) 
            self.ip_bias.fill_(0.0) 
            self.apical_bp1.zero_()
            self.apical_bp2.zero_()
            self.corr_ema.zero_()
            self.corr_var.fill_(1.0)
            self.state.zero_()
            self.calcium_traces.zero_()
            
            # --- Phase 4: Structural Renormalization ---
            # Renormalize net weights to maintain stability
            self.enforce_spectral_radius()
            self.remodel_structure()


    def reset_plastic_weights(self):
        """
        Restores weights and intrinsic parameters to their initial base state.
        Ensures a clean slate for curriculum epochs.
        """
        with torch.no_grad():
            self.weight_values.copy_(self.base_weight_values)
            self.ip_gain.fill_(1.0)
            self.ip_bias.zero_()
            self.corr_ema.zero_()
            self.corr_var.fill_(1.0)
            self.threshold_adaptation.fill_(1.0)
            self.activation_ema.zero_()
            self.activation_var.fill_(1.0)
            if hasattr(self, 'calcium_traces'):
                self.calcium_traces.zero_()
        print("[ENGINE] Plastic weights and homeostatic biases reset to base state.")

    def remodel_structure(self, prune_ratio=0.01):
        """
        Selective Structural Plasticity.
        Prunes the `prune_ratio` lowest-utility synapses (based on Fisher info) 
        and reinitializes them. This mimics biological synaptogenesis.
        Only applies to free edges (not I/O).
        """
        if prune_ratio <= 0.0:
            return 0
            
        with torch.no_grad():
            free_utility = self.fisher_diag[self.free_edge_mask]
            n_prune = int(free_utility.size(0) * prune_ratio)
            
            if n_prune == 0:
                return 0
                
            # Local Fisher-Information-Aware Pruning
            # Instead of magnitude, we prune synapses with low coincidence statistics (low Fisher info)
            threshold = torch.kthvalue(free_utility, n_prune).values.item()
            prune_mask_free = free_utility <= threshold
            
            prune_mask = torch.zeros_like(self.fisher_diag, dtype=torch.bool)
            prune_mask[self.free_edge_mask] = prune_mask_free
            
            # Clear cascade and tracking stats for pruned edges
            self.w_surface[prune_mask] = 0.0
            self.w_mid[prune_mask] = 0.0
            self.w_deep[prune_mask] = 0.0
            self.eligibility_traces[prune_mask] = 0.0
            self.fisher_diag[prune_mask] = 1.0 # Reset to 1.0 (max uncertainty) initially
            self.omega[prune_mask] = 0.0
            self.running_contribution[prune_mask] = 0.0
            
            n_reset = prune_mask.sum().item()
            src_inh = self.is_inhibitory[self.indices[0, prune_mask]]
            
            # Reinitialize based on parent distribution (Log-Normal inspired)
            avg_magnitude = self.weight_values[self.free_edge_mask].abs().mean().item()
            new_weights = torch.exp(torch.randn(n_reset, device=self.device) * 0.5) * (avg_magnitude * 0.5)
            new_weights = torch.where(src_inh, -new_weights, new_weights)
            
            self.weight_values[prune_mask] = new_weights
            
        
            
            return n_reset

    def zero_states(self):
        """
        Zeroes out all fast transient states.
        Should be called between distinct input sequences or evaluation batches.
        """
        self.state.zero_()
        self.state_basal.zero_()
        self.state_apical.zero_()
        self.sfa_states.zero_()
        self.cahva_states.zero_()
        self.rho_slow_states.zero_()
        self.apical_bp1.zero_()
        self.apical_bp2.zero_()
        self.apical_beta.fill_(1.0)
        if hasattr(self, 'previous_state'):
            self.previous_state.zero_()

    def inject_apical_nudge(self, target_vector, strength=1.0):
        """
        Injects a target signal into the apical compartments of output nodes.
        Used during the nudge phase to propagate error gradients downward.
        """
        with torch.no_grad():
            # target_vector is expected to be [256] for motor nodes 256-511
            target = target_vector.to(self.device)
            self.state_apical[self.output_indices] += strength * target

    def compute_burst_coincidence(self):
        """
        Calculates somatic burst probability based on BAC-firing logic.
        Burst = sigmoid(basal_drive) * (sigmoid(apical_fast) - 0.4*sigmoid(apical_slow))
        """
        with torch.no_grad():
            # Approximate the basal drive (usually tanh in get_soma)
            i_soma_base = self.state_basal * self.ip_gain + self.ip_bias
            somatic_spike = torch.sigmoid(torch.tanh(i_soma_base) * 15.0 - 7.5)
            
            # Apical XOR logic
            apical_gate_fast = torch.sigmoid(self.state_apical * 8.0 - 3.0)
            apical_gate_slow = torch.sigmoid(self.state_apical * 12.0 - 10.0)
            apical_xor = apical_gate_fast - 0.4 * apical_gate_slow
            
            return (somatic_spike * apical_xor).clamp(min=0.0)

    def update_apical_beta(self):
        """
        Adjusts per-module apical sensitivity based on NPE/PPE ratio.
        High NPE (over-prediction) -> lower gain to prioritize sensory evidence.
        High PPE (novelty) -> higher gain to strengthen top-down expectations.
        """
        with torch.no_grad():
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                mod_error = self.spatial_errors[start:end]
                # PPE nodes in this module (positive error)
                ppe_sum = mod_error[~self.is_neg_pe[start:end]].abs().sum().item()
                # NPE nodes in this module (negative error)
                npe_sum = mod_error[self.is_neg_pe[start:end]].abs().sum().item()
                
                ratio = (ppe_sum + 1e-6) / (npe_sum + 1e-6)
                # target beta between 0.5 and 3.0 via sigmoid
                target_beta = 0.5 + 2.5 * torch.sigmoid(torch.tensor(ratio - 1.0)).item()
                # Smooth update
                self.apical_beta[start:end] = 0.9 * self.apical_beta[start:end] + 0.1 * target_beta

    def settle(self, input_vector, max_steps=40, tol=5e-3,
               input_mask=None, damping=0.15, implicit_damping=2.0,
               sigma_noise=0.05):
        """Single-phase settling using IMEX integration."""
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32, device=self.device)
        else:
            input_vector = input_vector.to(self.device, dtype=torch.float32)

        if input_mask is not None:
            if not isinstance(input_mask, torch.Tensor):
                input_mask = torch.tensor(input_mask, dtype=torch.float32, device=self.device)
            else:
                input_mask = input_mask.to(self.device)

        # STSP E-E mask (Phase 3)
        # Source at association (>512), not inhibitory, same module
        src_not_inh = (~self.is_inhibitory[self.indices[0]]).float()
        dst_not_inh = (~self.is_inhibitory[self.indices[1]]).float()
        ee_mask = src_not_inh * dst_not_inh * self.intra_edge_mask.float()

        # Persistent Activity Injection (Working Memory / Trace Conditioning)
        input_total = input_vector.clone()
        if hasattr(self, 'context_injection_weight'):
            input_total[self.is_l56] += self.context_ema[self.is_l56] * self.context_injection_weight

        res = jit_solve_dynamics_imex(
            self.state_basal, self.state_apical,
            self.weight_values, self.indices,
            self.basal_intra_mask, self.basal_inter_mask,
            self.apical_intra_mask, self.apical_inter_mask,
            self.biases, self.taus, input_total, self.dt, max_steps, tol, input_mask,
            self.mod_starts, self.mod_ends,
            self.is_pv, self.is_sst, self.is_vip, self.is_lts, self.is_inhibitory,
            self.is_l23, self.is_l56, self.is_neg_pe, self.is_dg,
            self.activation_ema, self.ip_gain, self.ip_bias,
            self.node_to_level,
            self.sparsity_alpha, 
            self.sfa_states, self.cahva_states,
            self.rho_slow_states, self.ais_distance, self.threshold_adaptation,
            self.inh_depression_soma, self.inh_depression_dend,
            self.u_facilitation, self.x_depression, self.nmda_ratio,
            self.tau_facil, self.tau_depress, self.U0,
            ee_mask,
            self.gap_junction_indices, self.gap_junction_weights,
            self.apical_bp1, self.apical_bp2, self.lifetime_firing,
            0.5, damping, implicit_damping, sigma_noise, self.apical_beta,
            self.calcium_store, self.purkinje_mask
        )
        
        # Unpack the 22-element return tuple
        (self.state_basal, self.state_apical, self.state, diff, steps, self.dt, 
         self.sfa_states, self.cahva_states, self.rho_slow_states, 
         self.threshold_adaptation, self.ais_distance, self.inh_depression_soma, self.inh_depression_dend, 
         self.u_facilitation, self.x_depression,
         self.ip_gain, self.ip_bias, self.apical_bp1, self.apical_bp2, 
         self.lifetime_firing, self.nmda_ratio, self.calcium_store) = res
        
        self.last_settle_diff = diff
        return self.state

    def compute_prediction_errors(self):
        """
        Compute hierarchical prediction errors.

        Spatial prediction error at level ℓ:
            ε_ℓ = x_ℓ − W_topdown * tanh(x_{ℓ+1})

        Uses ONLY top-down edges (higher→lower level) for the prediction,
        not the full weight matrix. The full synaptic input includes
        intra-module, lateral, and bottom-up contributions which are part
        of the dynamics, not the hierarchical prediction.

        I/O nodes (0-511) get zero spatial error here — the output target
        is injected separately by the training loop.
        Top-level nodes also get zero (no parent to predict them).

        Returns total prediction error energy F = 0.5 * sum(||ε||²)
        """
        rho = torch.tanh(self.state)

        # Build sparse matrix with ONLY top-down edges
        cascade_weights = self.effective_weights
        td_vals = (cascade_weights[self.topdown_edge_mask] *
                   self.u_facilitation[self.topdown_edge_mask] *
                   self.x_depression[self.topdown_edge_mask])
        td_sparse = torch.sparse_coo_tensor(
            self.topdown_indices.flip(0), td_vals,
            (self.num_nodes, self.num_nodes))

        # Top-down prediction: what higher levels predict for lower levels
        topdown_pred = torch.mv(td_sparse, rho)

        if getattr(self, 'temporal_variance_ema', None) is None:
            self.temporal_variance_ema = torch.ones(self.num_nodes, device=self.device)
        else:
            # Bound variation so EMA isn't wildly inflating temporal precision
            var_change = (rho - rho.mean()).pow(2).clamp(0, 5.0)
            self.temporal_variance_ema = 0.99 * self.temporal_variance_ema + 0.01 * var_change
            
        # FIX: Rebalancing Generative Predictive Coding & Laminar Segregation
        # PRC (Inhibitory) units regulate "volume" of error signals via divisive normalization
        prc_activity = torch.relu(self.state[self.is_inhibitory]).mean()
        precision = 10.0 * (1.0 + prc_activity)
        
        # Spatial error components (Item 4)
        # Prediction: W_td * tanh(x_upper)
        prediction = precision * topdown_pred
        
        # Positive Error (pPE): Excited by stimulus, inhibited by prediction
        # pPE = max(0, x - prediction)
        ppe_raw = torch.clamp(self.state - prediction, min=0.0)
        
        # Negative Error (nPE): Excited by prediction, inhibited by stimulus
        # nPE = max(0, prediction - x)
        npe_raw = torch.clamp(prediction - self.state, min=0.0)
        
        # Combine into a single error tensor where nPE units store negative error signals
        # This allows existing energy minimizers to work while preserving nPE identity
        combined_error = torch.where(self.is_neg_pe, -npe_raw, ppe_raw)
        
        # Divisive normalization of the error signal by PRC units
        divisive_factor = 1.0 + torch.relu(self.state[self.is_inhibitory]).mean()
        normalized_error = combined_error / divisive_factor
        
        # ERR units (L2/3) predominantly ascend errors.
        self.spatial_errors = normalized_error.clamp(-2.0, 2.0)
        self.spatial_errors[self.is_l56] *= 0.1  # EXP units don't broadcast upward prediction errors
        self.spatial_errors[self.is_l4] *= 0.5   # L4 are input recipients, moderate error

        # Store top-down prediction variance for diagnostics
        self.topdown_pred_var = topdown_pred[512:].var().item()

        # Zero errors for nodes without meaningful top-down prediction:
        # - I/O nodes: input is clamped; output target injected by trainer
        self.spatial_errors[:512] = 0
        # - Top-level nodes: no parent level predicts them (they are the prior)
        top_mask = self.node_to_level == self.max_level
        self.spatial_errors[top_mask] = 0

        spatial_energy = 0.5 * torch.sum(self.spatial_errors ** 2).item()

        # --- Temporal prediction errors ---
        # ε_temporal_ℓ = x_ℓ(t) - A_ℓ * x_ℓ(t-1)
        self.temporal_errors.zero_()
        temporal_energy = 0.0

        for mod_idx, (start, end) in enumerate(self.module_ranges):
            current = self.state[start:end]
            prev = self.previous_state[start:end]
            a_diag = self.temporal_A[mod_idx]

            # Temporal prediction: A * previous_state
            temporal_pred = a_diag * prev
            t_error = current - temporal_pred
            self.temporal_errors[start:end] = t_error

            temporal_energy += 0.5 * torch.sum(t_error ** 2).item()

        # Total free energy with Metabolic Cost (minimizing surprisal and restricting activity bounds)
        metabolic_cost = 0.001 * torch.sum(self.state.abs()).item()
        total_energy = spatial_energy + torch.sum(self.temporal_alpha * 0.5 * self.temporal_errors**2).item() + metabolic_cost

        return total_energy

    def update_weights_phase2(self, free_state, nudge_state, learning_rate=0.01, 
                              target_sequence=None, hippo_edge_mask=None, active_level_max=None,
                              dopamine: float = 1.0, acetylcholine: float = 1.0):
        """
        Local Hebbian weight update based on True Equilibrium Propagation.

        Top-Down Spatial weight update (Prospective Configuration):
            ΔW_ij ∝ (rho_nudge_i - rho_free_i) * rho_nudge_j

        Neuromodulatory Gating:
        - Dopamine (DA): RPE surrogate. Scales plasticity based on surprise/success.
        - Acetylcholine (ACh): Uncertainty/Attention surrogate. High ACh = rapid memory encoding.
        """
        with torch.no_grad():
            rho_free = torch.tanh(free_state)
            rho_nudge = torch.tanh(nudge_state)
            
            # --- Intrinsic Plasticity (IP) Update ---
            # Info-Max: Adjust both gain and offset to maximize mutual information
            if getattr(self, 'activation_ema', None) is None:
                self.activation_ema = torch.zeros(self.num_nodes, device=self.device)
                self.activation_var = torch.ones(self.num_nodes, device=self.device)
                self.ip_gain = torch.ones(self.num_nodes, device=self.device)

            # Define ip_gradient early to avoid scope issues
            ip_gradient = self.sparsity_alpha - self.activation_ema
            if ip_gradient is None: # Safety check
                ip_gradient = torch.zeros_like(self.activation_ema)

            # Calculate somatic bursts from current basal/apical compartments
            bursts = torch.relu(self.state_basal) * torch.relu(self.state_apical)
                
            self.activation_ema = 0.99 * self.activation_ema + 0.01 * rho_free.abs()
            self.activation_var = 0.99 * self.activation_var + 0.01 * (rho_free - self.activation_ema).pow(2)
            # --- Item 1: KL-Divergence Based Intrinsic Plasticity (IP) ---
            # Targets an exponential firing rate distribution: P(y) = (1/mu) * exp(-y/mu)
            # Gradient derived from minimizing KL-divergence D_KL(P_y || P_target)
            mu = self.sparsity_alpha
            y = self.state.abs() # Firing rate magnitude
            
            # Theoretical gradiant for exponential distribution with tanh non-linearity
            # delta_beta = (1/mu) - y*(1/mu + 1)
            # delta_alpha = (1/alpha) + u * delta_beta
            ip_lr = 0.05
            
            # rheobase (bias) update
            ip_grad_beta = (1.0/mu) - y * (1.0/mu + 1.0)
            self.ip_bias = torch.clamp(self.ip_bias + ip_lr * ip_grad_beta, -5.0, 5.0)
            
            #sensitivity (gain) update
            ip_grad_alpha = (1.0/self.ip_gain) + self.state_basal * ip_grad_beta
            self.ip_gain = torch.clamp(self.ip_gain + ip_lr * ip_grad_alpha, 0.2, 8.0)
            
            # --- Item 4: Track Synaptic Correlations for Sleep Renalization ---
            with torch.no_grad():
                rho_pre = torch.tanh(self.state_basal[self.indices[0]])
                rho_post = torch.tanh(self.state[self.indices[1]])
                corr = (rho_pre * rho_post).abs()
                self.corr_ema = 0.95 * self.corr_ema + 0.05 * corr
                self.corr_var = 0.95 * self.corr_var + 0.05 * (corr - self.corr_ema).pow(2)
            
            # Use free phase scaled by gain for associative rules
            rho = torch.tanh(free_state * self.ip_gain)
            
            # --- Update Calcium Traces for HSS ---
            self.calcium_traces += (rho.abs() - self.calcium_traces) / self.tau_calcium

            # --- Spatial weight update (top-down + bottom-up + lateral) ---
            idx_i = self.indices[0]
            idx_j = self.indices[1]

            grad = torch.zeros_like(self.weight_values)

            # --- Item 1: Differential Hebbian Update (Apical - Basal) ---
            # The local difference between apical expectation and basal reality 
            # serves as an instantaneous, localized error signal.
            td_rho_nudge_j = rho_nudge[idx_j[self.topdown_edge_mask]]
            td_w = self.effective_weights[self.topdown_edge_mask]
            
            td_error = self.state_apical[idx_i[self.topdown_edge_mask]] - self.state_basal[idx_i[self.topdown_edge_mask]]
            td_grad = td_error * td_rho_nudge_j
            
            # Mirror Descent Multiplicative Dynamics: dw proportional to w
            grad[self.topdown_edge_mask] = 10.0 * td_grad * (0.01 + torch.abs(td_w))

            # Pre-synaptic state t-1 for temporal prediction
            prev_rho = torch.tanh(self.previous_state)

            # Bottom-up edges: Multiplicative STDP
            bu_prev_rho_i = prev_rho[idx_i[self.bottomup_edge_mask]]
            bu_rho_j = rho[idx_j[self.bottomup_edge_mask]]
            bu_w_plastic = self.effective_weights[self.bottomup_edge_mask]
            bu_burst_j = bursts[idx_j[self.bottomup_edge_mask]]
            
            bu_grad = bu_prev_rho_i * bu_rho_j * (1.0 + bu_burst_j)
            # LMD for bottom-up: changes are proportional to spine size (current weight)
            grad[self.bottomup_edge_mask] = 0.2 * bu_grad * (0.01 + torch.abs(bu_w_plastic)) * (1.0 - torch.abs(bu_w_plastic) / 5.0)

        

            # FIX 4: Lateral edges — Central-Annual-Surround (CAS) Topology
            lat_i = idx_i[self.lateral_edge_mask]
            lat_j = idx_j[self.lateral_edge_mask]
            lat_prev_rho_i = prev_rho[lat_i]
            lat_rho_j = rho[lat_j]
            lat_w = self.effective_weights[self.lateral_edge_mask]

            pos_i = self.positions[lat_i]
            pos_j = self.positions[lat_j]
            dists = torch.norm(pos_i - pos_j, dim=1)
            
            # CAS Spatial Topology Identifiers
            cas_center = (dists < 0.2).float()
            cas_annular = ((dists >= 0.2) & (dists < 0.5)).float()
            cas_surround = (dists >= 0.5).float()

            # Per-module mean activity for inhibition scaling
            module_mean_act = torch.zeros(self.num_modules, device=self.device)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                module_mean_act[mod_idx] = rho[start:end].abs().mean()

            lat_src_mod = torch.zeros(lat_i.size(0), dtype=torch.long, device=self.device)
            if hasattr(self, 'node_to_module'):
                lat_src_mod = self.node_to_module[lat_i]
            else:
                for mod_idx, (start, end) in enumerate(self.module_ranges):
                    mask = (lat_i >= start) & (lat_i < end)
                    lat_src_mod[mask] = mod_idx

            mod_act = module_mean_act[lat_src_mod]
            inhibition_strength = torch.clamp(mod_act - 0.3, min=0.0)

            oja_excitatory = lat_prev_rho_i * lat_rho_j - lat_w * lat_rho_j.pow(2)
            inhibitory = -inhibition_strength * lat_prev_rho_i.abs() * lat_rho_j.abs()
            
            grad[self.lateral_edge_mask] = 0.1 * (
                cas_center * oja_excitatory + 
                cas_annular * (inhibitory * 2.0) + 
                cas_surround * (oja_excitatory * 0.1)
            )

            # --- Orthogonalization penalty on lateral weights ---
            with torch.enable_grad():
                lat_w_var = lat_w.detach().clone().requires_grad_(True)
                total_penalty = 0.0
                
                for mod_idx, (start, end) in enumerate(self.module_ranges):
                    mod_size = end - start
                    if mod_size <= 1: continue
                    
                    # Topological Divergence for Hippocampus
                    # DG (Pattern Separation): Requires massive expansion recoding.
                    # We dedicate more nodes to DG to reduce probability of overlap.
                    # Increase to 70% expansion for DG
                    # These variables (module_id, hippo_module_id, n_in_module)
                    # are assumed to be available in the scope where this function is called,
                    # or are attributes of 'self' that are not explicitly defined in the provided context.
                    # For this edit, we assume they are available as per the instruction.
                    # If they are not, this would lead to a NameError.
                    # if module_id == hippo_module_id:
                    #     n_l4 = max(1, int(n_in_module * 0.70)) # DG
                    #     n_l23 = max(1, int(n_in_module * 0.20)) # CA3
                    #     n_l56 = n_in_module - n_l4 - n_l23      # CA1

                    mask = (lat_i >= start) & (lat_i < end) & (lat_j >= start) & (lat_j < end)
                    if not mask.any(): continue
                    
                    mod_i_idx = lat_i[mask] - start
                    mod_j_idx = lat_j[mask] - start
                    mod_w_var = lat_w_var[mask]
                    
                    W_dense = torch.sparse_coo_tensor(
                        torch.stack([mod_i_idx, mod_j_idx]), mod_w_var, (mod_size, mod_size)
                    ).to_dense()
                    
                    W_norm = torch.nn.functional.normalize(W_dense, p=2, dim=1, eps=1e-6)
                    sim_matrix = torch.mm(W_norm, W_norm.t())
                    
                    I = torch.eye(mod_size, device=self.device)
                    penalty = torch.sum((sim_matrix - I) ** 2)
                    total_penalty = total_penalty + penalty
                    
                if isinstance(total_penalty, torch.Tensor) and total_penalty.requires_grad:
                    ortho_grad = torch.autograd.grad(total_penalty, lat_w_var)[0]
                    grad[self.lateral_edge_mask] -= 0.05 * ortho_grad

                # --- Item 4: DG Contrastive Orthogonalization Penalty ---
                # Mathematically force the layer to maximize Hamming distance and minimize cosine similarity.
                # Specifically applied to input projections (sensory -> DG).
                dg_input_mask = self.is_dg[self.indices[1]] & (self.indices[0] < 512)
                if dg_input_mask.any():
                    dg_w = self.weight_values[dg_input_mask]
                    dg_w_var = dg_w.detach().clone().requires_grad_(True)
                    # Contrastive penalty: minimize dot product between different input weights
                    # We treat each DG node's input weight vector as a point in high-D space
                    # But since we have a sparse representation, we'll approximate with weight variance
                    # and a penalty on dense reconstruction similarity.
                    penalty_dg = torch.sum(dg_w_var.pow(2)) # Energy constraint
                    # For a more advanced contrastive loss, we'd need multiple inputs.
                    # As a proxy, we'll use the existing orthogonality logic if possible.
                    grad[dg_input_mask] -= 0.1 * dg_w_var # Simple decorrelation/shrinkage to force sparsity

            # --- Inhibitory Synaptic Plasticity (Item 1) ---
            # Vogels-like rule: Dw_ij = eta * (x_i * (x_j - rho_target))
            # Targets a low-frequency AI firing regime.
            src_inh = self.is_inhibitory[idx_i]
            if src_inh.any():
                rho_inh_pre = rho_free[idx_i[src_inh]]
                rho_inh_post = rho_free[idx_j[src_inh]]
                
                # Target activity rho_target = sparsity_alpha
                inh_error = rho_inh_post - self.sparsity_alpha[idx_j[src_inh]]
                inh_grad = rho_inh_pre * inh_error
                
                # Update inhibitory weights (only if source is inhibitory)
                # We want MORE inhibition (more negative weight) if activity is too high.
                # So if error > 0, Dw should be negative.
                grad[src_inh] -= 1.0 * inh_grad

            # Gradient clipping: Explicit max_norm=1.0 limit
            grad_norm = grad.norm()
            if grad_norm > 1.0:
                grad = grad * (1.0 / grad_norm)
            
            # --- Metabolic Cost (Weight Penalty) ---
            grad -= 0.0001 * self.effective_weights.sign()

            grad = grad.clamp(-1.0, 1.0)

            # --- Dale's ANNs (DANNs) Fisher Information Scaling ---
            if not hasattr(self, 'fisher_info'):
                self.fisher_info = torch.ones(self.num_nodes, device=self.device)
            var_inst = (rho_free - getattr(self, 'activation_ema', torch.zeros_like(rho_free))).pow(2)
            self.fisher_info = 0.99 * self.fisher_info + 0.01 * var_inst
            
            src_inh_all = self.is_inhibitory[idx_i]
            fisher_scale = 1.0 / (self.fisher_info[idx_i] + 1e-4)
            fisher_scale = torch.clamp(fisher_scale, 0.1, 5.0)
            grad[src_inh_all] *= fisher_scale[src_inh_all] * 0.2  # Dampen and normalize inhibitory updates

            # --- Local Homeostatic Scaling (Variance Control) ---
            target_var = 0.1
            tau_update = 0.05 * (var_inst - target_var)
            self.taus = torch.clamp(self.taus + tau_update, 0.5, 100.0)

            # --- Calculate Thalamic Saliency Gate ---
            # Use total network energy as a proxy for 'surprise'.
            # If current_energy is much higher than the EMA, it's a novel/important signal -> scale up learning.
            # If it's matching or lower, it's predictable noise (like spaces) -> scale down learning.
            total_temp_energy = torch.sum(self.temporal_alpha * 0.5 * self.temporal_errors ** 2).item()
            current_energy = 0.5 * torch.sum(self.spatial_errors ** 2).item() + total_temp_energy
            if self.surprise_ema == 0.0:
                self.surprise_ema = current_energy
            else:
                self.surprise_ema = 0.99 * self.surprise_ema + 0.01 * current_energy
            
            saliency_gate = torch.clamp(torch.tensor(current_energy / max(self.surprise_ema, 1e-6)), min=0.1, max=3.0).item()

            # Combined importance-aware learning rate (metaplastic + SI)
            consolidation = torch.abs(self.w_deep)
            importance = self.omega
            
            # Base learning rate explicitly gated by Neuromodulators
            # Dopamine scales overall magnitude
            # Acetylcholine scales rate of new structural learning (surface weights)
            global_neuromodulation = max(0.01, dopamine * acetylcholine)
            
            meta_lr = (learning_rate * global_neuromodulation) / (1.0 + self.meta_scale * consolidation + 0.1 * importance)
            
            # Apply Thalamic Gating
            meta_lr = meta_lr * saliency_gate

            # CLS: hippocampal synapses get highly boosted learning under high ACh
            if hippo_edge_mask is not None:
                meta_lr = meta_lr * (1.0 + hippo_edge_mask.float() * (9.0 * acetylcholine))

            # Level gate (Strategy 2): restrict updates to edges whose max endpoint
            # level is <= active_level_max.  L0-only per-character, L1+L2 at word
            # boundaries, L3 at sentence boundaries.
            if active_level_max is not None:
                src_lvls = self.node_to_level[self.indices[0]]
                dst_lvls = self.node_to_level[self.indices[1]]
                edge_max_lvl = torch.maximum(src_lvls, dst_lvls)
                level_gate = (edge_max_lvl <= active_level_max).float()
                grad = grad * level_gate

            # --- Three-Factor Learning ---
            # 1. Accumulate local Hebbian correlation into the eligibility trace
            self.eligibility_traces = self.eligibility_traces * (1.0 - 1.0/self.tau_eligibility) + grad * (1.0/self.tau_eligibility)

            # Multiplicative Update Requirement (Item 5)
            # dw ∝ w * grad
            # To preserve sign, we use abs(w) and apply to w
            delta_w = meta_lr * self.eligibility_traces * (self.weight_values.abs() + 0.01)
            self.w_surface += delta_w
            
            # --- Anti-Hebbian Lateral Competition (Oja-like) ---
            # F_i = beta * F_i + (1-beta) * (grad_i)^2
            with torch.no_grad():
                self.fisher_diag = 0.99 * self.fisher_diag + 0.01 * (delta_w ** 2)

            # --- Homeostatic Synaptic Scaling (HSS) ---
            # dw = -rho_hss * w * (C - epsilon)
            # Reverses sign for inhibitory synapses to increase inhibition when overly active
            calcium_dev = self.calcium_traces[self.indices[1]] - self.calcium_target
            hss_mod = -self.hss_rho * calcium_dev
            src_inh = self.is_inhibitory[self.indices[0]]
            hss_mod[src_inh] *= -1.0  
            w_plastic = self.w_surface + self.w_mid + self.w_deep
            self.w_surface += hss_mod * w_plastic

            # Fisher-Weighted Pruning
            # Instead of uniform RMS normalization, selectively decay weights based on their Fisher Information.
            # Weights with high F_i (crucial for loss) decay slower.
            # Δw_prune = -lambda_prune * w_i / (F_i + epsilon)
            lambda_prune = 1e-4
            epsilon = 1e-6
            prune_decay = lambda_prune / (self.fisher_diag + epsilon)
            prune_decay = torch.clamp(prune_decay, 0.0, 0.05)
            
            cascade_all = self.w_surface + self.w_mid + self.w_deep
            cascade_all = cascade_all * (1.0 - prune_decay)
            
            # Reconstruct w_surface
            self.w_surface = cascade_all - self.w_mid - self.w_deep

         

            # --- Top-down weight diversity regularization ---
            # (Kept from original — prevents mode collapse)
            td_idx = torch.where(self.topdown_edge_mask)[0]
            td_src = self.topdown_indices[0]
            td_vals = self.w_surface[td_idx]

            unique_src, inverse = torch.unique(td_src, return_inverse=True)
            src_sums = torch.zeros(len(unique_src), device=self.device)
            src_counts = torch.zeros(len(unique_src), device=self.device)
            src_sums.scatter_add_(0, inverse, td_vals)
            src_counts.scatter_add_(0, inverse, torch.ones_like(td_vals))
            src_means = src_sums / src_counts.clamp(min=1)

            correction = src_means[inverse] * 0.01
            self.w_surface[td_idx] -= correction

            # --- Dale's Law: Enforce E/I constraints ---
            # Excitatory neurons can only have positive outgoing weights.
            # Inhibitory neurons can only have negative outgoing weights.
            src_inh = self.is_inhibitory[self.indices[0]]
            cascade_all = self.w_surface + self.w_mid + self.w_deep + self.weight_values
            
            # Constraint: total effective_weight >= 0 for Exc, <= 0 for Inh
            cascade_all_clamped = torch.where(src_inh, cascade_all.clamp(max=0.0), cascade_all.clamp(min=0.0))
            
            # Reconstruct w_surface from the clamped effective weight
            self.w_surface = cascade_all_clamped - self.w_mid - self.w_deep - self.weight_values

            # --- Bias/AIS update from prediction errors + IP ---
            bias_grad = (rho_nudge - rho_free) + self.temporal_alpha * self.temporal_errors
            if active_level_max is not None:
                node_gate = (self.node_to_level <= active_level_max).float()
                bias_grad = bias_grad * node_gate
                ip_gradient = ip_gradient * node_gate

            self.biases += learning_rate * 0.1 * bias_grad
            self.biases.clamp_(-1.0, 1.0)
            
            # Ais Distance Differential Equation
            self.ais_distance += learning_rate * 0.5 * ip_gradient
            self.ais_distance.clamp_(0.5, 2.5)

            # --- Temporal transition matrix update ---
            eta = learning_rate * 25.0  # Dedicated temporal learning rate (η)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                # Level gate: skip modules above the active update threshold
                if active_level_max is not None and self.module_levels[mod_idx] > active_level_max:
                    continue
                t_error = self.temporal_errors[start:end]
                prev = self.previous_state[start:end]

                a_mod = eta * t_error * prev

                a_mod = a_mod.clamp(-0.05, 0.05) - 0.005 * self.temporal_A[mod_idx]
                self.temporal_A[mod_idx] += a_mod

                self.temporal_A[mod_idx].clamp_(-1.0, 1.0)

    def store_previous_state(self):
        """Store current state as previous state for temporal prediction."""
        self.previous_state = self.state.clone()

    def enforce_spectral_radius(self, target_max=0.95):
        """
        Estimate dominant eigenvalue of the CASCADE-ONLY weights (learned
        deltas: w_surface + w_mid + w_deep) on FREE edges, and dampen
        w_surface if the cascade spectral radius exceeds target_max.

        Decoupled from base weights: The base weight_values were already
        SR-tuned at initialization (graph.py). Enforcing on the total
        effective weight (base + cascade) was crushing learned structure
        because the base SR (~0.90) consumed most of the budget, leaving
        almost no room for the cascade to add meaningful structure.

        By enforcing on cascade-only, the learned weights are constrained
        independently, and the base initialization is preserved.

        Only dampens w_surface — w_mid and w_deep are protected long-term
        memory that must never be rescaled by a transient SR measurement.
        """
        with torch.no_grad():
            # Power iteration vector (persistent across calls for convergence)
            if getattr(self, '_power_iter_v', None) is None:
                self._power_iter_v = torch.randn(self.num_nodes, device=self.device)
                norm = torch.norm(self._power_iter_v)
                if norm > 0:
                    self._power_iter_v /= norm

            # Build sparse matrix from FREE edges of CASCADE ONLY (not base weights)
            cascade_weights = (self.w_surface + self.w_mid + self.w_deep)[self.free_edge_mask]

            # Skip if cascade is negligible
            cascade_norm = cascade_weights.norm().item()
            if cascade_norm < 1e-6:
                self.last_sr = 0.0
                return

            W_cascade = torch.sparse_coo_tensor(
                self.free_edge_indices, cascade_weights,
                (self.num_nodes, self.num_nodes)
            )

            # Power iteration (5 steps)
            v = self._power_iter_v
            for _ in range(5):
                v_next = torch.mv(W_cascade, v)
                norm = torch.norm(v_next)
                if norm > 1e-8:
                    v = v_next / norm

            self._power_iter_v = v

            # Rayleigh quotient estimate
            Wv = torch.mv(W_cascade, v)
            eigenvalue = torch.dot(v, Wv)
            sr = torch.abs(eigenvalue).item()
            self.last_sr = sr

            # Only dampen w_surface (fast transient weights) on free edges
            if sr > target_max:
                dampening = target_max / sr
                self.w_surface[self.free_edge_mask] *= dampening

    def enforce_jacobian_spectral_radius(self, target_max=0.95):
        """
        Enforce spectral radius on the actual Jacobian J = (1/τ)(-I + W*diag(sech²(x))),
        not just weight norms. This bounds the true dynamical instability.

        Uses power iteration on the full Jacobian (with I/O rows/cols zeroed)
        and dampens w_surface on free edges when SR exceeds target.
        """
        with torch.no_grad():
            J = self.get_jacobian()  # Already zeros I/O rows/cols

            # Power iteration on J (5 steps, persistent vector)
            if getattr(self, '_jac_power_v', None) is None:
                self._jac_power_v = torch.randn(self.num_nodes, device=self.device)
                norm = self._jac_power_v.norm()
                if norm > 0:
                    self._jac_power_v /= norm

            v = self._jac_power_v
            for _ in range(5):
                v_next = J @ v
                norm = v_next.norm()
                if norm > 1e-8:
                    v = v_next / norm

            self._jac_power_v = v

            # Rayleigh quotient estimate of spectral radius
            Jv = J @ v
            sr = torch.abs(torch.dot(v, Jv)).item()
            self.last_jacobian_sr = sr

            # Only dampen w_surface (fast transient weights) on free edges
            if sr > target_max:
                dampening = target_max / sr
                self.w_surface[self.free_edge_mask] *= dampening

    # Removed duplicate sleep_phase method.


    def get_jacobian(self) -> torch.Tensor:
        """
        Computes the Jacobian of the temporal transition dynamics at the current state.
        This enables gradient alignment tracking (Diagnostic Test 2).
        
        Returns:
            [N, N] Dense Jacobian matrix tensor.
        """
        with torch.no_grad():
            N = self.num_nodes
            
            rho = torch.tanh(self.state)
            drho = 1.0 - rho.pow(2)
            
            W_eff = torch.sparse_coo_tensor(
                self.indices, self.effective_weights, 
                (N, N)
            ).to_dense()
            
            W_drho = W_eff * drho.unsqueeze(0)
            
            taus_inv = 1.0 / self.taus.unsqueeze(1)
            I = torch.eye(N, dtype=torch.float32, device=self.device)
            
            J = taus_inv * (-I + W_drho)

            # Zero out I/O node rows/columns (nodes 0-511).
            # I/O nodes have tau=0.1 which amplifies their Jacobian rows
            # by 10x, inflating the measured spectral radius. Since I/O
            # nodes are clamped during settling, their Jacobian
            # contribution is meaningless but dominates the eigenvalue.
            J[:512, :] = 0
            J[:, :512] = 0

            return J



    def update_context_ema(self):
        """Update running EMA of state for cross-boundary context preservation."""
        self.context_ema = (1 - self.context_ema_alpha) * self.context_ema + self.context_ema_alpha * self.state

    def blend_context(self, blend_factor=0.2):
        """Blend stored context EMA back into state after boundary reset."""
        self.state = (1 - blend_factor) * self.state + blend_factor * self.context_ema

    def damp_weights(self, factor=0.9):
        """Damps recurrent weights by a factor (applied to deep level)."""
        self.w_deep *= factor

    def cascade_transfer(self, include_deep=True):
        """
        Transfers learned weights down the memory cascade.
        Surface -> Mid -> Deep.
        """
        with torch.no_grad():
            # 1. Surface to Mid
            transfer_s2m = self.w_surface / self.tau_surface_to_mid
            self.w_mid += transfer_s2m
            self.w_surface -= transfer_s2m

            # 2. Mid to Deep
            if include_deep:
                transfer_m2d = self.w_mid / self.tau_mid_to_deep
                
                # FIX 3: Magnitude-gated hippocampal transfer
                # Neocortical synapses (τ_deep=2000) drip-feed continuously.
                # Hippocampal synapses (τ_deep=50) wait until a coherent pattern
                # forms in w_mid (magnitude > 0.1) before flushing rapidly.
                # This prevents noisy transient patterns from cluttering deep CLS memory.
                is_hippo = self.tau_mid_to_deep < 100.0
                has_magnitude = self.w_mid.abs() > 0.1
                
                # Zero out transfer for hippo edges lacking sufficient structural magnitude
                transfer_m2d[is_hippo & ~has_magnitude] = 0.0

                self.w_deep += transfer_m2d
                self.w_mid -= transfer_m2d


    def cascade_stats(self):
        """Returns mean absolute magnitude at each cascade level for diagnostics."""
        return {
            'surface': self.w_surface.abs().mean().item(),
            'mid': self.w_mid.abs().mean().item(),
            'deep': self.w_deep.abs().mean().item(),
        }

    def get_prediction_error_by_level(self):
        """
        Returns dict of average prediction error magnitude per level.
        Useful for monitoring whether hierarchy is learning properly.
        """
        errors_by_level = {}
        for mod_idx, (start, end) in enumerate(self.module_ranges):
            level = self.module_levels[mod_idx]
            spatial_err = torch.mean(torch.abs(self.spatial_errors[start:end])).item()
            temporal_err = torch.mean(torch.abs(self.temporal_errors[start:end])).item()

            if level not in errors_by_level:
                errors_by_level[level] = {'spatial': [], 'temporal': [], 'count': 0}
            errors_by_level[level]['spatial'].append(spatial_err)
            errors_by_level[level]['temporal'].append(temporal_err)
            errors_by_level[level]['count'] += 1

        result = {}
        for level, data in errors_by_level.items():
            result[level] = {
                'spatial': np.mean(data['spatial']),
                'temporal': np.mean(data['temporal']),
            }
        return result

    def get_topdown_weight_stats(self):
        """
        Returns mean and std of top-down weights grouped by (src_level, dst_level).
        """
        with torch.no_grad():
            td_weights = self.effective_weights[self.topdown_edge_mask]
            src_levels = self.node_to_level[self.topdown_indices[0]]
            dst_levels = self.node_to_level[self.topdown_indices[1]]

            stats = {}
            for src_l in range(self.max_level + 1):
                for dst_l in range(src_l):
                    mask = (src_levels == src_l) & (dst_levels == dst_l)
                    if mask.any():
                        vals = td_weights[mask]
                        stats[(src_l, dst_l)] = {
                            'mean': vals.mean().item(),
                            'std': vals.std().item(),
                        }
            return stats