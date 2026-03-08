#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analytical Hamiltonian Fitting Script for 2H-Bilayer TMDs
Includes K-point transformation and trainable mu parameter.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import sys

# ==========================================
#               --- 配置区域 ---
# ==========================================

# 1. 文件与路径
INPUT_FILE  = "bands_data.dat"        # 输入文件：k1, k2, k3, E1, E2... (分数坐标)
OUTPUT_PNG  = "fit_result_simple.png"   
OUTPUT_TXT  = "fit_params_simple.txt"   

# 2. 物理参数
A_TB_ANG    = 3.445               # TB 模型对应的晶格常数 a (Å)
ALAT_ANG    = 3.445              # QE 计算中使用的 alat 参数 (Å)
NB_FIT      = 4                  # 拟合能带数量
E_SOFT      = 0.1                # 权重平滑因子

# 3. 初始猜测 (Initial Guess)
# params: [mu, t1, t2, t3, t4, t5, lam1, lam2, t00, t01, t02, t03] (共12个)
X0 = [0.0, 0.03, 0.10, 0.006, -0.01, -0.007, 0.02, 0.001, 0.06, 0.02, 0.004, 0.015]

# 4. 边界范围 (Bounds)
BOUNDS_L = [-0.3] + [-0.2] * 11  # mu 的范围稍大，hopping 范围保持 -2 到 2
BOUNDS_U = [ 0.3] + [ 0.2] * 11

# ==========================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_band_file(path):
    log(f"Loading data from {path}...")
    try:
        data = np.loadtxt(path)
        return data[:, 0:3], data[:, 3:]
    except Exception as e:
        log(f"Error loading file: {e}")
        sys.exit(1)

def convert_k_frac_to_qe_cart(k_frac, a_ang, alat_ang):
    """
    Kx_qe = (alat/a) * k1
    Ky_qe = (alat/a) * ( (1/√3)*k1 + (2/√3)*k2 )
    """
    k1 = k_frac[:, 0]
    k2 = k_frac[:, 1]
    factor = alat_ang / a_ang
    kx_qe = factor * k1
    ky_qe = factor * ( (1.0/np.sqrt(3.0)) * k1 + (2.0/np.sqrt(3.0)) * k2 )
    return np.column_stack((kx_qe, ky_qe))

def cumulative_path_distance(k_coords):
    dk = np.diff(k_coords, axis=0)
    step = np.linalg.norm(dk, axis=1)
    return np.concatenate(([0.0], np.cumsum(step)))

def get_hamiltonian_eigs_qe_k(kx_qe, ky_qe, params, a_eff):
    """
    params: [mu, t1, t2, t3, t4, t5, lam1, lam2, t00, t01, t02, t03]
    """
    mu, t1, t2, t3, t4, t5, lam1, lam2, t00, t01, t02, t03 = params
    a = a_eff
    alpha = 0.5 * a * kx_qe
    beta = 0.5 * np.sqrt(3.0) * a * ky_qe

    # Intra-layer terms
    xi = ( mu
        + 2*t1*(np.cos(2*alpha) + 2*np.cos(alpha)*np.cos(beta))
        + 2*t2*(np.cos(2*beta) + 2*np.cos(3*alpha)*np.cos(beta))
        + 2*t3*(np.cos(4*alpha) + 2*np.cos(2*alpha)*np.cos(2*beta))
        + 4*t4*(np.cos(alpha)*np.cos(3*beta) + np.cos(4*alpha)*np.cos(2*beta) + np.cos(5*alpha)*np.cos(beta))
        + 2*t5*(np.cos(6*alpha) + 2*np.cos(3*alpha)*np.cos(3*beta)) )

    Lambda = ( 2*lam1*(np.sin(2*alpha) - 2*np.sin(alpha)*np.cos(beta))
        + 2*lam2*(np.sin(4*alpha) - 2*np.sin(2*alpha)*np.cos(2*beta)) )

    # Inter-layer terms
    T = ( t00
        + 2*t01*(2*np.cos(alpha)*np.cos(beta) + np.cos(2*alpha))
        + 2*t02*(np.cos(2*beta) + 2*np.cos(3*alpha)*np.cos(beta))
        + 2*t03*(np.cos(4*alpha) + 2*np.cos(2*alpha)*np.cos(2*beta)) )

    H = np.array([
        [xi + Lambda, 0,           T,           0          ],
        [0,           xi - Lambda, 0,           T          ],
        [T,           0,           xi - Lambda, 0          ],
        [0,           T,           0,           xi + Lambda]
    ], dtype=float)

    return np.linalg.eigvalsh(H)

def calculate_residuals(p, kxy_qe, E_dft_target, a_eff, weights):
    Nk = kxy_qe.shape[0]
    E_tb_list = [get_hamiltonian_eigs_qe_k(kxy_qe[i,0], kxy_qe[i,1], p, a_eff) for i in range(Nk)]
    E_tb_all = np.array(E_tb_list)
    E_tb_fit = np.sort(E_tb_all[:, :NB_FIT], axis=1)
    E_dft_sorted = np.sort(E_dft_target, axis=1)
    return ((E_tb_fit - E_dft_sorted) * weights).flatten()

def main():
    k_frac_all, E_dft_all = load_band_file(INPUT_FILE)
    E_dft_target = E_dft_all[:, :NB_FIT]
    
    kxy_qe = convert_k_frac_to_qe_cart(k_frac_all, A_TB_ANG, ALAT_ANG)
    a_eff = A_TB_ANG * (2.0 * np.pi / ALAT_ANG)
    
    weights = 1.0 / (np.abs(E_dft_target) + E_SOFT)
    weights /= np.mean(weights)
    
    log(f"Starting Analytical Fit with mu inclusion...")
    start_t = time.time()
    
    res = least_squares(
        calculate_residuals, 
        X0, 
        bounds=(BOUNDS_L, BOUNDS_U),
        args=(kxy_qe, E_dft_target, a_eff, weights),
        method='trf',
        verbose=1
    )
    
    log(f"Optimization finished. Cost: {res.cost:.6f}")
    best_p = res.x

    p_names = ["mu", "t1", "t2", "t3", "t4", "t5", "lam1", "lam2", "t00", "t01", "t02", "t03"]
    final_rmse = np.sqrt(2.0 * res.cost / E_dft_target.size)

    with open(OUTPUT_TXT, "w") as f:
        f.write("# Analytical Fit (mu is now a trainable parameter)\n")
        f.write(f"# RMSE_eV = {final_rmse:.12f}\n")
        for i, name in enumerate(p_names):
            f.write(f"{name:10s} = {best_p[i]:.12f}\n")
    
    E_tb_final = np.array([get_hamiltonian_eigs_qe_k(kxy_qe[i,0], kxy_qe[i,1], best_p, a_eff) for i in range(kxy_qe.shape[0])])
    k_dist = cumulative_path_distance(k_frac_all[:, 0:2])
    
    plt.figure(figsize=(7, 5), dpi=150)
    for i in range(NB_FIT):
        plt.plot(k_dist, E_dft_target[:, i], 'k-', alpha=0.3, lw=2, label='DFT' if i==0 else "")
        plt.plot(k_dist, np.sort(E_tb_final[:, :NB_FIT], axis=1)[:, i], 'r--', alpha=0.8, lw=1.5, label='TB Fit' if i==0 else "")
    
    plt.axhline(0, c='gray', ls=':', lw=1)
    plt.xlabel("K-path (Fractional Distance)")
    plt.ylabel("Energy (eV)")
    plt.title("Analytical Fit (Trainable mu)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    log(f"Results saved to {OUTPUT_TXT} and {OUTPUT_PNG}")

if __name__ == "__main__":
    main()