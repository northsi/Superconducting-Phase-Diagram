from pythtb import *
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数定义 ---
a = 3.44
params = {
    'mu': 0.047117804179,
    't1': 0.027241510475,   'lam1': 0.019771580282,  
    't2': 0.097048128757,   'lam2': 0.001381298593,  
    't3': 0.006456144806,                       
    't4': -0.010013520585,                      
    't5': -0.010229267961,                      
    't00': 0.049168763556,                      
    't01': 0.019955827013,                      
    't02': 0.004103384902,                      
    't03': 0.013522176355                       
}

omega = np.exp(1j * 2 * np.pi / 3)

sigma_0 = np.eye(2, dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# --- 2. 扩展定义晶格与 5层 BdG 模型 ---
lat = [[a, 0.0], [-a/2, a*np.sqrt(3)/2]]

# 10 个轨道: 5层电子 + 5层空穴
orb = [[0.0, 0.0]] * 10

# 定义直观的索引
T1_e, T2_e, S1_e, S2_e, S3_e = 0, 1, 2, 3, 4   # 电子轨道
T1_h, T2_h, S1_h, S2_h, S3_h = 5, 6, 7, 8, 9   # 空穴轨道

my_model = tb_model(2, 2, lat, orb, nspin=2)

# ==========================================================
# --- 3. 设置 On-site Energy ---
# ==========================================================
# 为了物理严谨，TMD 和 超导衬底 可以有不同的化学势
# 如果它们完全欧姆接触且费米面对齐，可以设为一样，但变量最好独立
mu_T = params['mu']
mu_S = params['mu'] # 这里假设衬底和TMD化学势对齐

my_model.set_onsite([
    mu_T * sigma_0,   # T1_e
    mu_T * sigma_0,   # T2_e
    mu_S * sigma_0,   # S1_e
    mu_S * sigma_0,   # S2_e
    mu_S * sigma_0,   # S3_e
    -mu_T * sigma_0,  # T1_h
    -mu_T * sigma_0,  # T2_h  <-- 【修正】：补齐了所有逗号
    -mu_S * sigma_0,  # S1_h
    -mu_S * sigma_0,  # S2_h
    -mu_S * sigma_0   # S3_h
])

# ==========================================================
# --- 4. 仅仅为 TMD 层 (T1, T2) 添加跃迁 ---
# ==========================================================
def add_TMD_intra_hopping(amp_t, amp_lam, vec):
    h1_e = amp_t * sigma_0 + 1j * amp_lam * sigma_z
    my_model.set_hop(h1_e, T1_e, T1_e, vec)                  
    my_model.set_hop(-np.conj(h1_e), T1_h, T1_h, vec)
    
    h2_e = amp_t * sigma_0 - 1j * amp_lam * sigma_z
    my_model.set_hop(h2_e, T2_e, T2_e, vec)                  
    my_model.set_hop(-np.conj(h2_e), T2_h, T2_h, vec)

vecs_NN = [[1, 0], [0, 1], [-1, -1]] 
vecs_NNN = [[2, 1], [1, 2], [-1, 1]] 
vecs_3NN = [[2, 0], [0, 2], [-2, -2]]
vecs_4NN = [[3, 1], [3, 2], [2, 3], [1, 3], [-1, 2], [-2, 1]] 
vecs_5NN = [[3, 0], [0, 3], [-3, -3]]

for v in vecs_NN: add_TMD_intra_hopping(params['t1'], params['lam1'], v)
for v in vecs_NNN: add_TMD_intra_hopping(params['t2'], 0, v)
for v in vecs_3NN: add_TMD_intra_hopping(params['t3'], params['lam2'], v)
for v in vecs_4NN: add_TMD_intra_hopping(params['t4'], 0, v)
for v in vecs_5NN: add_TMD_intra_hopping(params['t5'], 0, v)

def add_TMD_inter_hopping(amp_t, vec):
    h_e = amp_t * sigma_0
    my_model.set_hop(h_e, T1_e, T2_e, vec)            
    my_model.set_hop(-np.conj(h_e), T1_h, T2_h, vec)  

add_TMD_inter_hopping(params['t00'], [0, 0])
for v in vecs_NN: add_TMD_inter_hopping(params['t01'], v)
for v in vecs_NNN: add_TMD_inter_hopping(params['t02'], v)
for v in vecs_3NN: add_TMD_inter_hopping(params['t03'], v)

# ==========================================================
# --- 5. 构建底层 3 层 s-wave 超导体及其近邻效应 ---
# ==========================================================
t_sc = 1.0       # 1 eV，典型的宽带金属跃迁强度
t_z_sc = 0.5     # 0.5 eV，超导层内部的垂直耦合，让它成为真正的 3D 金属
# 3. 超导参数
Delta_s = 0.1    # 衬底本征 s-wave 能隙
# 5.1 超导层内的普通金属跃迁 (没有任何奇异的 SOC)
for v in vecs_NN:
    for e_idx, h_idx in [(S1_e, S1_h), (S2_e, S2_h), (S3_e, S3_h)]:
        my_model.set_hop(t_sc * sigma_0, e_idx, e_idx, v)
        my_model.set_hop(-np.conj(t_sc * sigma_0), h_idx, h_idx, v)

# 5.2 超导层内部的垂直跃迁 (S1-S2, S2-S3)
for (e1, h1), (e2, h2) in [ ((S1_e, S1_h), (S2_e, S2_h)), ((S2_e, S2_h), (S3_e, S3_h)) ]:
    my_model.set_hop(t_z_sc * sigma_0, e1, e2, [0, 0])
    my_model.set_hop(-np.conj(t_z_sc * sigma_0), h1, h2, [0, 0])

# 5.3 【新增关键步骤】：TMD 与 超导衬底的 范德华近邻耦合 (T2 <-> S1)
t_prox = 0.03  # 你需要自定义这个近邻跳跃强度参数
my_model.set_hop(t_prox * sigma_0, T2_e, S1_e, [0, 0])
my_model.set_hop(-np.conj(t_prox * sigma_0), T2_h, S1_h, [0, 0])

# S-wave 奇宇称自旋配对矩阵 (自旋单态配对，电子-空穴空间)
# 对应于 Delta * i * sigma_y = [[0, Delta], [-Delta, 0]]
s_wave_mat = np.array([
    [0, 1],
    [-1, 0]
], dtype=complex)

# 5.4 超导层的 s-wave 配对项 (On-site 局域配对)
# 假设 s_wave_mat = np.array([[0, 1], [-1, 0]]) * 1j (即 i*sigma_y)
for e_idx, h_idx in [(S1_e, S1_h), (S2_e, S2_h), (S3_e, S3_h)]:
    my_model.set_hop(Delta_s * s_wave_mat, e_idx, h_idx, [0, 0])

# ==========================================================
# --- 添加 Rashba SOC (仅限于 Top Layer 第一层) ---
# ==========================================================

# 设置 Rashba 耦合强度参数 (可根据需要调整大小，若按原样则设为 1.0)
alpha_R = 0.01  

# 沿 [1, 0] 方向的 Rashba 矩阵
R1 = np.array([
    [0, -0.5],
    [0.5, 0]
], dtype=complex)

# 沿 [0, 1] 方向的 Rashba 矩阵
R2 = np.array([
    [0, 0.25 + (np.sqrt(3)/4)*1j],
    [-0.25 + (np.sqrt(3)/4)*1j, 0]
], dtype=complex)

# 沿 [-1, -1] 方向的 Rashba 矩阵
R3 = np.array([
    [0, 0.25 - (np.sqrt(3)/4)*1j],
    [-0.25 - (np.sqrt(3)/4)*1j, 0]
], dtype=complex)

def add_rashba(mat, vec):
    """
    为第一层添加 Rashba 跃迁
    注意：因为同一方向上已经有了正常的 t 和 SOC 跃迁，必须使用 mode="add" 进行叠加
    """
    # 电子部分的 Rashba 跃迁
    my_model.set_hop(alpha_R * mat, T1_e, T1_e, vec, mode="add")
    
    # 空穴部分的 Rashba 跃迁 (BdG 要求对正常跃迁项取 -H*)
    my_model.set_hop(-np.conj(alpha_R * mat), T1_h, T1_h, vec, mode="add")

# 添加三个给定方向的 Rashba SOC
add_rashba(R1, [1, 0])
add_rashba(R2, [0, 1])
add_rashba(R3, [-1, -1])

# --- 5. 添加超导配对项 ---
# 超导配对参数 (适当放大Delta以便在图中清晰看到能隙和边缘态)
Delta0 = 0.1

# 你提供的超导配对矩阵
S_1 = np.eye(2, dtype=complex)

S_2 = np.array([
    [omega**2, 0],
    [0, omega]
], dtype=complex)

S_3 = np.array([
    [omega, 0],
    [0, omega**2]
], dtype=complex)

def add_pairing(mat, vec):
    # 1. 正向跃迁 (+vec)
    my_model.set_hop(Delta0 * mat, T1_e, T1_h, vec)
    my_model.set_hop(Delta0 * mat, T2_e, T2_h, vec)
    
    # 2. 【核心修复】：反向跃迁 (-vec)，保证费米子反对称性 Delta_{ij} = -Delta_{ji}^T
    # 这一步是为了让动量空间满足 Delta(-k) = -Delta^T(k)，恢复真实的 BdG 粒子空穴对称性
    neg_vec = [-vec[0], -vec[1]]
    my_model.set_hop(-Delta0 * mat.T, T1_e, T1_h, neg_vec)
    my_model.set_hop(-Delta0 * mat.T, T2_e, T2_h, neg_vec)

add_pairing(S_1, [1, 0])
add_pairing(S_2, [0, 1])
add_pairing(S_3, [-1, -1])

import numpy as np
import matplotlib.pyplot as plt

def plot_k0_edge_states(ribbon_model, Ny):
    """
    计算并画出一维纳米带在 k=0 处的能谱，并将零能态的空间波函数分布分别独立画出。
    接着输出边缘态的【层/轨道分辨贡献图】(支持异质结)。
    最后输出单一边缘态的【局域化长度 (Localization Length) 分析图】。
    """
    # 1. 求解 k=0 处的能级和波函数
    evals, evecs = ribbon_model.solve_one([0.0], eig_vectors=True)
    
    # 2. 找到最靠近 E=0 的能级
    abs_evals = np.abs(evals)
    sorted_indices = np.argsort(abs_evals)
    edge_indices = sorted_indices[:4]
    
    # 3. 创建动态画布：1个能谱图 + 4个独立的波函数图
    num_edge_states = len(edge_indices)
    fig, axes = plt.subplots(1, num_edge_states + 1, figsize=(4 * (num_edge_states + 1), 4.5))
    
    # ================= 第 1 张图：k=0 能谱 =================
    ax_spec = axes[0]
    ax_spec.plot(evals, 'ko', markersize=3, alpha=0.5, label='Bulk States')
    ax_spec.plot(edge_indices, evals[edge_indices], 'ro', markersize=6, label='Edge States')
    ax_spec.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax_spec.set_ylabel("Energy (eV)")
    ax_spec.set_xlabel("State Index")
    ax_spec.set_title("Energy Spectrum at $k_x = 0$")
    ax_spec.legend()
    
    # ================= 第 2~5 张图：每个边缘态单独展示 =================
    y_positions = np.arange(Ny)          
    all_prob_y = []       
    
    # 动态推断每个原胞的状态数 (-1 会让 numpy 自动算出是 8 还是 20)
    # 并初始化一个全零矩阵来完美累加 4 个态的轨道分布
    first_wf = evecs[edge_indices[0], :]
    states_per_cell = np.abs(first_wf).reshape(Ny, -1).shape[1]
    total_prob_mat = np.zeros((Ny, states_per_cell)) 
    
    for i, idx in enumerate(edge_indices):
        wf = evecs[idx, :]            
        prob_density = np.abs(wf)**2  
        
        # 将一维或多维数组 reshape 为 (Ny, states_per_cell)
        prob_mat_single = prob_density.reshape(Ny, states_per_cell)
        
        # 累加到总矩阵中 (完美消除简并态的层极化/不对称误差)
        total_prob_mat += prob_mat_single
        
        # 提取单个态在 y 方向上的总分布
        prob_y = np.sum(prob_mat_single, axis=1)
        all_prob_y.append(prob_y)
        
        ax_wf = axes[i + 1]  
        ax_wf.plot(y_positions, prob_y, ls='-', lw=2, color='C1', alpha=0.8)
        ax_wf.set_xlabel("Width Index (y)")
        if i == 0:
            ax_wf.set_ylabel("Probability Density $|\psi|^2$") 
            
        ax_wf.set_title(f"Edge State {i + 1}\nE={evals[idx]:.6f} eV", fontsize=11)
        ax_wf.set_ylim(bottom=0)

    plt.tight_layout()  
    plt.savefig('edge_state.png', dpi=300)  
    plt.show()  

    # =================================================================
    # 科学提取：使用求和后的矩阵进行 层/轨道分辨 (自适应 4 轨道 或 10 轨道)
    # =================================================================
    fig_layer, ax_layer = plt.subplots(figsize=(8, 5))
    
    if states_per_cell == 20: # 10 轨道异质结模型 (每项含2个自旋)
        # 根据我们在构建异质结时的定义：
        # T1_e(0,1), T2_e(2,3), S123_e(4~9)
        # T1_h(10,11), T2_h(12,13), S123_h(14~19)
        T1_contrib = np.sum(total_prob_mat[:, 0:2], axis=1) + np.sum(total_prob_mat[:, 10:12], axis=1)
        T2_contrib = np.sum(total_prob_mat[:, 2:4], axis=1) + np.sum(total_prob_mat[:, 12:14], axis=1)
        Substrate_contrib = np.sum(total_prob_mat[:, 4:10], axis=1) + np.sum(total_prob_mat[:, 14:20], axis=1)
        
        ax_layer.plot(y_positions, T1_contrib, 'b-o', markersize=4, lw=1.5, label='Top TMD (T1)')
        ax_layer.plot(y_positions, T2_contrib, 'r-s', markersize=4, lw=1.5, label='Bottom TMD (T2)')
        ax_layer.plot(y_positions, Substrate_contrib, 'g-^', markersize=4, lw=1.5, label='s-wave Substrate (S1+S2+S3)')
        ax_layer.set_title('Heterostructure Layer Resolution (Sum of 4 Edge States)')
        
    elif states_per_cell == 8: # 原来的 4 轨道 TMD 模型
        L1_contrib = np.sum(total_prob_mat[:, 0:2], axis=1) + np.sum(total_prob_mat[:, 4:6], axis=1)
        L2_contrib = np.sum(total_prob_mat[:, 2:4], axis=1) + np.sum(total_prob_mat[:, 6:8], axis=1)
        
        ax_layer.plot(y_positions, L1_contrib, 'b-o', markersize=4, lw=2, label='Layer 1 (L1)')
        ax_layer.plot(y_positions, L2_contrib, 'r-s', markersize=4, lw=2, label='Layer 2 (L2)')
        ax_layer.set_title('Bilayer TMD Layer Resolution (Sum of 4 Edge States)')
        
    ax_layer.plot(y_positions, np.sum(total_prob_mat, axis=1), 'k--', alpha=0.5, label='Total Probability')
    ax_layer.set_xlabel("Ribbon Width Index (y)")
    ax_layer.set_ylabel("Total Probability Density $\sum|\psi|^2$")
    ax_layer.legend()
    ax_layer.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('layer_resolved.png', dpi=300)
    plt.show()

    # =================================================================
    # 科学拟合：使用简并子空间的总概率密度求 Localization Length
    # =================================================================
    total_prob_y = np.sum(all_prob_y, axis=0)
    
    half_Ny = Ny // 2
    left_prob_y = total_prob_y[:half_Ny]
    peak_y = np.argmax(left_prob_y)
    
    fit_length = min(20, half_Ny - peak_y - 1) 
    fit_y = np.arange(peak_y, peak_y + fit_length)
    fit_prob = total_prob_y[fit_y]
    
    valid_mask = fit_prob > 1e-12
    fit_y = fit_y[valid_mask]
    fit_prob = fit_prob[valid_mask]
    
    fig2, ax_loc = plt.subplots(figsize=(6, 5))
    if len(fit_y) >= 2:
        slope, intercept = np.polyfit(fit_y, np.log(fit_prob), 1)
        xi = np.abs(2.0 / slope) 
        
        plot_fit_y = np.arange(peak_y, peak_y + fit_length + 10)
        fit_line = np.exp(slope * plot_fit_y + intercept)
        ax_loc.plot(plot_fit_y, fit_line, 'r--', lw=2.5, 
                    label=f'Total Density Fit\n$\\xi \\approx {xi:.2f}$ cells')
    else:
        xi = np.nan
        
    ax_loc.plot(y_positions, total_prob_y, marker='o', ls='-', color='navy', 
                label='Total Edge Probability $\\sum |\\psi_i|^2$')
    ax_loc.set_yscale('log')
    ax_loc.set_ylim(bottom=1e-8, top=np.max(total_prob_y) * 2) 
    ax_loc.set_xlabel("Ribbon Width Index (y)")
    ax_loc.set_ylabel("Total Probability Density (Log Scale)")
    ax_loc.set_title("Localization Length of Degenerate Edge Subspace")
    ax_loc.legend()
    ax_loc.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('localization_length_total.png')
    plt.show()
    
    print(f"通过总概率密度拟合，边缘态局域化长度为: {xi:.4f} 个原胞")

# ==========================================================
# --- 6. 【核心】开边界：建立一维纳米带模型 ---
# ==========================================================

# 沿第二个晶格方向切割，宽度为 40 个原胞
Ny = 100 
# cut_piece(宽度, 沿哪个基矢切断, 是否相连)
# 1 表示沿 lat 的第二个基矢方向切断，保留第一个基矢 [a, 0.0] 的周期性
ribbon_model = my_model.cut_piece(Ny, 1, glue_edgs=False)

# # --- 7. 计算一维能带 ---
# # 一维布里渊区路径：从 -pi 到 pi (在约化坐标下是 -0.5 到 0.5)
# path_1D = [[-0.5], [0.0], [0.5]]
# k_path, k_dist, k_node = ribbon_model.k_path(path_1D, 201, report=False)
# evals_1D = ribbon_model.solve_all(k_path)

# # --- 8. 绘图 ---
# fig, ax = plt.subplots(figsize=(8, 6))

# # 画出所有能带（体态会密集排列，边缘态通常穿过能隙）
# for i in range(evals_1D.shape[0]):
#     ax.plot(k_dist, evals_1D[i, :], 'k-', lw=0.5, alpha=0.6)

# # 设置图像范围，重点观察费米面(E=0)附近的超导能隙和边缘态
# ax.set_ylim(-0.05, 0.05) 
# ax.set_xlim(k_dist[0], k_dist[-1])
# ax.set_xticks(k_node)
# ax.set_xticklabels([r'$-\pi/a$', r'$0$', r'$\pi/a$'])
# ax.set_ylabel("Energy (eV)", fontsize=12)
# ax.set_title(f"1D BdG Nanoribbon (Width = {Ny} cells)", fontsize=14)
# ax.axhline(0, color='red', ls='--', lw=1.0, alpha=0.8)

# plt.tight_layout()  # 1. 先调整布局
# #plt.savefig('dz2_helical.png')  # 2. 再保存图片
# plt.show()  # 3. 最后显示

plot_k0_edge_states(ribbon_model, Ny)