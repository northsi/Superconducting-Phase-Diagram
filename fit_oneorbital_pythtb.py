#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PythTB-based Fitting Script for 2H-Bilayer TMDs (Fractional Input)
Input file format: k_frac_1  k_frac_2  k_frac_3  E1  E2 ...
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg") # No GUI
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from pythtb import *
import sys
import time

# --- Helper Utilities ---
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_band_file(path: str):
    """
    Load DFT bands from file.
    Assumes columns: k_frac_1 k_frac_2 k_frac_3 E1 E2 ...
    """
    log(f"Loading data from {path}...")
    try:
        data = np.loadtxt(path)
    except Exception as e:
        log(f"Error loading file: {e}")
        sys.exit(1)
        
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("Input file must have at least 4 columns!")
    
    # 假设前两列已经是分数坐标
    k_frac = data[:, 0:2] 
    E = data[:, 3:]       # Energies starts from column 4 (index 3)
    return k_frac, E

def cumulative_path_distance(k_coords):
    """Simple Euclidean distance in the input coordinate space for plotting x-axis"""
    dk = np.diff(k_coords, axis=0)
    step = np.linalg.norm(dk, axis=1)
    return np.concatenate(([0.0], np.cumsum(step)))

# --- PythTB Model Builder (Embeds the Correct Bilayer Logic) ---
def solve_pythtb_model(params_array, k_frac_input, a_val):
    """
    Constructs the model and solves eigenvalues.
    """
    # Unpack parameters
    # Order: mu, t1-t5, lam1, lam2, t00-t03
    (mu, t1, t2, t3, t4, t5, 
     lam1, lam2, 
     t00, t01, t02, t03) = params_array

    # 1. Define Lattice (Real Space)
    # 虽然输入是分数坐标，但 PythTB 内部定义 Hopping 仍需实空间几何
    lat = [[a_val, 0.0], [-0.5*a_val, 0.5*np.sqrt(3)*a_val]]
    orb = [[0.0, 0.0], [0.0, 0.0]] # Layer 1, Layer 2
    
    my_model = tb_model(2, 2, lat, orb, nspin=2)

    # 2. Pauli Matrices
    sigma_0 = np.eye(2)
    sigma_z = np.array([[1, 0], [0, -1]])

    # 3. On-site Energy
    my_model.set_onsite([mu * sigma_0, mu * sigma_0])

    # 4. Helper for Intra-layer Hopping (Spin-Layer Locking)
    def add_intra(amp_t, amp_lam, vec):
        # Layer 1: +lambda
        h1 = amp_t * sigma_0 + 1j * amp_lam * sigma_z
        my_model.set_hop(h1, 0, 0, vec)
        # Layer 2: -lambda (Inversion)
        h2 = amp_t * sigma_0 - 1j * amp_lam * sigma_z
        my_model.set_hop(h2, 1, 1, vec)

    # NN (d^2 = 1): [1,0], [0,1], [-1,-1]
    # [-1,-1] is correct here. My previous [1,1] was WRONG (it's also NN).
    vecs_NN  = [[1, 0], [0, 1], [-1, -1]]

    # NNN (d^2 = 3): [2,1], [1,2], [-1,1]
    # Check: 4+1-2=3. Correct.
    vecs_NNN = [[2, 1], [1, 2], [-1, 1]]

    # 3NN (d^2 = 4): [2,0], [0,2], [-2,-2]
    # Check: 4. Correct.
    vecs_3NN = [[2, 0], [0, 2], [-2, -2]]

    # 4NN (d^2 = 7): [3,1], [1,3], [3,2], [2,3], [-1,2], [-2,1]
    # Check [3,1]: 9+1-3=7. Correct.
    vecs_4NN = [[3, 1], [1, 3], [3, 2], [2, 3], [-1, 2], [-2, 1]]

    # 5NN (d^2 = 9): [3,0], [0,3], [-3,-3]
    vecs_5NN = [[3, 0], [0, 3], [-3, -3]]

    # 6. Add Intra-layer Hoppings
    for v in vecs_NN:  add_intra(t1, lam1, v)
    for v in vecs_NNN: add_intra(t2, 0.0, v)   # No SOC for NNN
    for v in vecs_3NN: add_intra(t3, lam2, v)  # lam2 is 3NN
    for v in vecs_4NN: add_intra(t4, 0.0, v)
    for v in vecs_5NN: add_intra(t5, 0.0, v)

    # 7. Add Inter-layer Hoppings
    # t00 (Vertical) -> vec [0,0] (FIXED)
    my_model.set_hop(t00 * sigma_0, 0, 1, [0, 0])
    
    # t01 (Slanted NN)
    for v in vecs_NN: my_model.set_hop(t01 * sigma_0, 0, 1, v)
    # t02 (Slanted NNN)
    for v in vecs_NNN: my_model.set_hop(t02 * sigma_0, 0, 1, v)
    # t03 (Slanted 3NN)
    for v in vecs_3NN: my_model.set_hop(t03 * sigma_0, 0, 1, v)

    # 8. Solve
    # PythTB solve_all takes fractional coordinates directly
    evals = my_model.solve_all(k_frac_input)
    
    return evals.T 

def match_bands_hungarian(E_tb, E_ref):
    """
    Match TB bands (E_tb) to reference bands (E_ref) at each k-point using the Hungarian algorithm.
    E_tb: (Nk, Nb_tb)
    E_ref: (Nk, Nb_ref)  -- assume Nb_tb >= Nb_ref, we'll return only Nb_ref columns matched
    Returns: E_tb_matched (Nk, Nb_ref)
    """
    Nk, Nb_tb = E_tb.shape
    _, Nb_ref = E_ref.shape
    if Nb_tb < Nb_ref:
        raise ValueError("TB has fewer bands than reference!")

    E_out = np.zeros((Nk, Nb_ref))
    for ik in range(Nk):
        # cost matrix: squared diff between each pair
        cost = (E_tb[ik, :][:, None] - E_ref[ik, :][None, :])**2
        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind indices in 0..Nb_tb-1 map to col_ind in 0..Nb_ref-1
        # pick the matched TB energies corresponding to the reference band order
        # create a mapping from ref-band idx -> TB energy
        tb_for_ref = np.full(Nb_ref, np.nan)
        for r, c in zip(row_ind, col_ind):
            tb_for_ref[c] = E_tb[ik, r]
        # if any unmatched (should not happen), fill with large number
        nanmask = np.isnan(tb_for_ref)
        if np.any(nanmask):
            tb_for_ref[nanmask] = 1e3
        E_out[ik, :] = tb_for_ref
    return E_out



# --- Main Logic ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="DFT bands file (fractional k)")
    ap.add_argument("--a_tb_ang", type=float, default=3.44, help="Lattice constant a (Angstrom)")
    ap.add_argument("--iter", type=int, default=50, help="Basin Hopping iterations")
    ap.add_argument("--out_png", default="fit_result_frac.png")
    ap.add_argument("--out_params", default="fit_params_frac.txt")
    
    args = ap.parse_args()

    # 1. Load Data
    k_frac, E_dft_all = load_band_file(args.infile)
    Nk, Nb_total = E_dft_all.shape
    
    # Fit lowest 4 bands
    Nb_fit = min(Nb_total, 4)
    E_dft_target = E_dft_all[:, :Nb_fit]
    
    # Simple path distance for plotting (Fractional Path Length)
    k_dist_vals = cumulative_path_distance(k_frac)
    
    log(f"Data Loaded: {Nk} points. Fitting {Nb_fit} bands.")

    # -------------- 3) 改进后的 objective ----------------

    E_cut = 0.6      # eV, 超导能量窗口（50 meV）
    E_soft = 0.01     # eV, 平滑参数，避免发散
    reg_lambda = 1e-5 # 很小的正则


    def objective(p):
        try:
            # ---- 1. Solve TB model ----
            E_tb = solve_pythtb_model(p, k_frac, args.a_tb_ang)

            # ---- 2. Shape normalization: (Nk, Nb) ----
            if E_tb.ndim != 2:
                return 1e6

            if E_tb.shape[0] == Nk:
                E_tb_full = E_tb
            elif E_tb.shape[1] == Nk:
                E_tb_full = E_tb.T
            else:
                return 1e6

            if E_tb_full.shape[1] < Nb_fit:
                return 1e6

            # ---- 3. Take lowest Nb_fit bands & sort (degeneracy-safe) ----
            E_tb_use = np.sort(E_tb_full[:, :Nb_fit], axis=1)
            E_dft_use = np.sort(E_dft_target, axis=1)

            # ---- 4. Construct superconductivity-focused weights ----
            # Only energies near EF matter
            mask = np.abs(E_dft_use) < E_cut
            if not np.any(mask):
                # No FS crossing in this k-range → neutral penalty
                return 0.0

            # Strongly weight states closer to EF
            weights = 1.0 / (E_soft + np.abs(E_dft_use))
            weights *= mask
            weights /= np.mean(weights[mask])

            # ---- 5. Loss: windowed, weighted MSE ----
            diff = (E_tb_use - E_dft_use) * weights
            mse = np.mean(diff[mask]**2)

            # ---- 6. Mild regularization ----
            reg = reg_lambda * np.mean(p**2)

            return mse + reg

        except Exception:
            return 1e6


    # 3. Setup Optimization
    # Initial Guess (mu, t1...t5, lam1, lam2, t00...t03)
    x0 = [0.05,  # mu
          0.03, 0.10, 0.006, -0.01, -0.007, # t1-t5
          0.02, 0.001, # lam1, lam2
          0.06, 0.02, 0.004, 0.015] # t00-t03
    
    # Bounds: +/- 5 eV for most params
    bounds = [(-1.0, 1.0)] * 12

    log("--- Starting Optimization (Basin Hopping) ---")
    log("Assuming input k-points are FRACTIONAL.")

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "tol": 1e-5
    }

    step_count = 0
    def print_fun(x, f, accepted):
        nonlocal step_count
        step_count += 1
        acc_str = "Acc" if accepted else "Rej"
        print(f"  Iter {step_count}: RMSE = {np.sqrt(f):.6f} eV [{acc_str}]", flush=True)

    # Run
    res = basinhopping(
        objective, 
        x0, 
        niter=args.iter, 
        minimizer_kwargs=minimizer_kwargs,
        callback=print_fun,
        disp=False,
        stepsize=0.02, 
        T=0.01
    )

    best_params = res.x
    final_rmse = np.sqrt(res.fun)
    
    log("-" * 40)
    log(f"Optimization Done. Final RMSE: {final_rmse:.6f} eV")
    log("-" * 40)

    # 4. Save Parameters
    p_names = ["mu", "t1", "t2", "t3", "t4", "t5", "lam1", "lam2", "t00", "t01", "t02", "t03"]
    with open(args.out_params, "w") as f:
        f.write("# Fitted PythTB 2H-Bilayer Parameters (Fractional K input)\n")
        for i, name in enumerate(p_names):
            f.write(f"{name:10s} = {best_params[i]:.12f}\n")
        f.write(f"RMSE_eV    = {final_rmse:.12f}\n")

    # 5. Final Plot
    E_tb_final = solve_pythtb_model(best_params, k_frac, args.a_tb_ang)

    plt.figure(figsize=(8, 6), dpi=150)
    # Plot DFT
    for ib in range(E_dft_target.shape[1]):
        plt.plot(k_dist_vals, E_dft_target[:, ib], 'k-', lw=2, alpha=0.3, label='DFT' if ib==0 else "")
    
    # Plot TB
    for ib in range(E_tb_final.shape[1]):
        plt.plot(k_dist_vals, E_tb_final[:, ib], 'r--', lw=1.5, alpha=0.9, label='TB Fit' if ib==0 else "")

    plt.axhline(0, color='gray', ls=':')
    plt.xlabel("Path Distance (Fractional Units)")
    plt.ylabel("Energy (eV)")
    plt.title(f"PythTB Fit Result (RMSE: {final_rmse:.4f} eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png)
    log(f"Plot saved to {args.out_png}")

if __name__ == "__main__":
    main()