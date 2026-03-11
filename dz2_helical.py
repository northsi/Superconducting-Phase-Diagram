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



# --- 2. 定义晶格与 2D BdG 模型 ---
lat = [[a, 0.0], [-a/2, a*np.sqrt(3)/2]]

# 4个轨道: L1电子, L2电子, L1空穴, L2空穴
orb = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
L1_e, L2_e = 0, 1 
L1_h, L2_h = 2, 3 

# 【已修正】：dim_k=2, dim_r=2，靠 orb 长度定义 4 轨道
my_model = tb_model(2, 2, lat, orb, nspin=2)

# --- 3. 设置 On-site Energy ---
my_model.set_onsite([
    params['mu'] * sigma_0,   # L1 e
    params['mu'] * sigma_0,   # L2 e
    -params['mu'] * sigma_0,  # L1 h
    -params['mu'] * sigma_0   # L2 h
])

# --- 3. 设置 On-site Energy (加入垂直位移电场) ---
# V_disp = 0.05  # 层间电势差 (20 meV)

# # L1 加正电势，L2 加负电势
# # BdG 矩阵中，空穴 (h) 的电势是电子 (e) 的相反数
# my_model.set_onsite([
#     (params['mu'] + V_disp) * sigma_0,   # L1 e
#     (params['mu'] - V_disp) * sigma_0,   # L2 e
#     -(params['mu'] + V_disp) * sigma_0,  # L1 h
#     -(params['mu'] - V_disp) * sigma_0,   # L2 h
    
# ], mode="add")


# # --- 尝试面内磁场 (Bx) ---
# Bx = 0.02  # 20 meV 的面内磁场

# # sigma_x = [[0, 1], [1, 0]]
# sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

# my_model.set_onsite([
#     params['mu'] * sigma_0 + Bx * sigma_x,   # L1 e
#     params['mu'] * sigma_0 + Bx * sigma_x,   # L2 e
#     -params['mu'] * sigma_0 - Bx * sigma_x,  # L1 h (注意空穴部分也对应翻转)
#     -params['mu'] * sigma_0 - Bx * sigma_x
# ])



# --- 4. 添加跃迁 ---
def add_intra_hopping(amp_t, amp_lam, vec):
    h1_e = amp_t * sigma_0 + 1j * amp_lam * sigma_z
    my_model.set_hop(h1_e, L1_e, L1_e, vec)                  
    my_model.set_hop(-np.conj(h1_e), L1_h, L1_h, vec)        # BdG 空穴跃迁为 -H*
    
    h2_e = amp_t * sigma_0 - 1j * amp_lam * sigma_z
    my_model.set_hop(h2_e, L2_e, L2_e, vec)                  
    my_model.set_hop(-np.conj(h2_e), L2_h, L2_h, vec)        

vecs_NN = [[1, 0], [0, 1], [-1, -1]] 
vecs_NNN = [[2, 1], [1, 2], [-1, 1]] 
vecs_3NN = [[2, 0], [0, 2], [-2, -2]]
vecs_4NN = [[3, 1], [3, 2], [2, 3], [1, 3], [-1, 2], [-2, 1]] 
vecs_5NN = [[3, 0], [0, 3], [-3, -3]]

for v in vecs_NN: add_intra_hopping(params['t1'], params['lam1'], v)
for v in vecs_NNN: add_intra_hopping(params['t2'], 0, v)
for v in vecs_3NN: add_intra_hopping(params['t3'], params['lam2'], v)
for v in vecs_4NN: add_intra_hopping(params['t4'], 0, v)
for v in vecs_5NN: add_intra_hopping(params['t5'], 0, v)

def add_inter_hopping(amp_t, vec):
    h_e = amp_t * sigma_0
    my_model.set_hop(h_e, L1_e, L2_e, vec)            
    my_model.set_hop(-np.conj(h_e), L1_h, L2_h, vec)  

add_inter_hopping(params['t00'], [0, 0])
for v in vecs_NN: add_inter_hopping(params['t01'], v)
for v in vecs_NNN: add_inter_hopping(params['t02'], v)
for v in vecs_3NN: add_inter_hopping(params['t03'], v)

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

# R1 = np.array([
#     [0, -0.5 * 1j],
#     [0.5 * 1j, 0]
# ], dtype=complex)

# # 沿 [0, 1] 方向的 Rashba 矩阵
# R2 = np.array([
#     [0, 0.25 * 1j + (np.sqrt(3)/4)],
#     [-0.25 * 1j - (np.sqrt(3)/4), 0]
# ], dtype=complex)

# # 沿 [-1, -1] 方向的 Rashba 矩阵
# R3 = np.array([
#     [0, 0.25 * 1j - (np.sqrt(3)/4)],
#     [-0.25 * 1j + (np.sqrt(3)/4), 0]
# ], dtype=complex)

def add_rashba(mat, vec):
    """
    为第一层添加 Rashba 跃迁
    注意：因为同一方向上已经有了正常的 t 和 SOC 跃迁，必须使用 mode="add" 进行叠加
    """
    # 电子部分的 Rashba 跃迁
    my_model.set_hop(alpha_R * mat, L1_e, L1_e, vec, mode="add")
    
    # 空穴部分的 Rashba 跃迁 (BdG 要求对正常跃迁项取 -H*)
    my_model.set_hop(-np.conj(alpha_R * mat), L1_h, L1_h, vec, mode="add")

# 添加三个给定方向的 Rashba SOC
add_rashba(R1, [1, 0])
add_rashba(R2, [0, 1])
add_rashba(R3, [-1, -1])

# ==========================================================
# --- 终极测试：TRS 保持的层间自旋翻转微扰 (打破高阶晶体保护) ---
# ==========================================================
# V_dirty_soc = 0.05  # 微扰强度

# sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

# # 电子部分：引入 -i * sigma_x 的层间跃迁
# # sigma_x = [[0, 1], [1, 0]]
# my_model.set_hop(-1j * V_dirty_soc * sigma_x, L1_e, L2_e, [0, 0], mode="add")

# # 空穴部分：必须是 -H_e^*
# # (-1j * sigma_x) 的复共轭是 (+1j * sigma_x)，前面再加个负号，还是 (-1j * sigma_x)！
# my_model.set_hop(-1j * V_dirty_soc * sigma_x, L1_h, L2_h, [0, 0], mode="add")

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
    my_model.set_hop(Delta0 * mat, L1_e, L1_h, vec)
    my_model.set_hop(Delta0 * mat, L2_e, L2_h, vec)
    
    # 2. 【核心修复】：反向跃迁 (-vec)，保证费米子反对称性 Delta_{ij} = -Delta_{ji}^T
    # 这一步是为了让动量空间满足 Delta(-k) = -Delta^T(k)，恢复真实的 BdG 粒子空穴对称性
    neg_vec = [-vec[0], -vec[1]]
    my_model.set_hop(-Delta0 * mat.T, L1_e, L1_h, neg_vec)
    my_model.set_hop(-Delta0 * mat.T, L2_e, L2_h, neg_vec)

add_pairing(S_1, [1, 0])
add_pairing(S_2, [0, 1])
add_pairing(S_3, [-1, -1])

import numpy as np
import matplotlib.pyplot as plt

def plot_k0_edge_states(ribbon_model, Ny, original_norb=4):
    """
    计算并画出一维纳米带在 k=0 处的能谱，并将零能态的空间波函数分布分别独立画出。
    最后再输出一张单一边缘态的局域化长度（Localization Length）分析图。
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
    states_per_cell = original_norb * 2 
    y_positions = np.arange(Ny) # 纳米带宽度方向的坐标
    
    # 保存所有边缘态的 prob_y 数据以便后续使用
    all_prob_y = []
    
    for i, idx in enumerate(edge_indices):
        ax_wf = axes[i + 1]  
        
        wf = evecs[idx, :]            
        prob_density = np.abs(wf)**2  
        prob_y = np.sum(prob_density.reshape(Ny, states_per_cell), axis=1)
        all_prob_y.append(prob_y)
        
        ax_wf.plot(y_positions, prob_y, 
                ls='-', 
                lw=2,                      # 线宽加粗
                color='C1',
                alpha=0.8)                  # 添加透明度

        ax_wf.set_xlabel("Width Index (y)")
        if i == 0:
            ax_wf.set_ylabel("Probability Density $|\psi|^2$") 
            
        ax_wf.set_title(f"Edge State {i + 1}\nE={evals[idx]:.6f} eV", fontsize=11)
        ax_wf.set_ylim(bottom=0)

    plt.tight_layout()  
    plt.savefig('edge_state.png', dpi=300)  
    plt.show()  


# =================================================================
    # 科学拟合：使用简并子空间的总概率密度求 Localization Length
    # =================================================================
    
    # 1. 将 4 个边缘态的概率密度求和 (消除求解器的随机线性组合效应)
    total_prob_y = np.sum(all_prob_y, axis=0)
    
    # 2. 我们只关注左半边纳米带 (y < Ny/2) 进行拟合
    half_Ny = Ny // 2
    left_prob_y = total_prob_y[:half_Ny]
    
    # 3. 找到左半边的峰值位置
    peak_y = np.argmax(left_prob_y)
    
    # 4. 选择拟合区间 (从峰值向体内延伸，避开体态底噪)
    fit_length = min(20, half_Ny - peak_y - 1) 
    fit_y = np.arange(peak_y, peak_y + fit_length)
    fit_prob = total_prob_y[fit_y]
    
    # 过滤掉数值底噪
    valid_mask = fit_prob > 1e-12
    fit_y = fit_y[valid_mask]
    fit_prob = fit_prob[valid_mask]
    
    fig2, ax_loc = plt.subplots(figsize=(6, 5))
    
    if len(fit_y) >= 2:
        # 指数拟合: ln(|psi|^2) = slope * y + intercept
        slope, intercept = np.polyfit(fit_y, np.log(fit_prob), 1)
        xi = np.abs(2.0 / slope) # 计算局域化长度
        
        # 画出拟合线 (延伸显示得长一点以便观察)
        plot_fit_y = np.arange(peak_y, peak_y + fit_length + 10)
        fit_line = np.exp(slope * plot_fit_y + intercept)
        ax_loc.plot(plot_fit_y, fit_line, 'r--', lw=2.5, 
                    label=f'Total Density Fit\n$\\xi \\approx {xi:.2f}$ cells')
    else:
        xi = np.nan
        
    # 绘制总概率密度的对数图
    ax_loc.plot(y_positions, total_prob_y, marker='o', ls='-', color='navy', 
                label='Total Edge Probability $\\sum |\\psi_i|^2$')
    
    # 设置对数 Y 轴
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

# ==========================================================
# --- 添加边缘势阱 (Edge Potential Well / Barrier) ---
# ==========================================================

# 1. 设置势阱的参数
V_edge = 0   # 边缘最大势能大小 (eV)
well_width = 5   # 势场影响的总深度（原胞层数）
decay_type = "linear"  # 衰减类型

# 2. 获取一维纳米带的总轨道数
norb_1d = ribbon_model.get_num_orbitals()

# 3. 构造势场数组
edge_potential = [np.zeros((2, 2), dtype=complex) for _ in range(norb_1d)]

# 4. 遍历并添加逐渐减小的势场
for y in range(Ny):
    # 计算左边缘的衰减因子
    if y < well_width:
        # 线性衰减: 从边缘 V_edge 到内部 0
        factor = (well_width - y) / well_width
        V_eff = V_edge * factor
        
        # 电子轨道
        edge_potential[y*4 + 0] = V_eff * sigma_0
        edge_potential[y*4 + 1] = V_eff * sigma_0
        # 空穴轨道
        edge_potential[y*4 + 2] = -V_eff * sigma_0
        edge_potential[y*4 + 3] = -V_eff * sigma_0
        
    # 计算右边缘的衰减因子
    elif y >= Ny - well_width:
        factor = (y - (Ny - well_width) + 1) / well_width
        V_eff = V_edge * factor
        
        # 电子轨道
        edge_potential[y*4 + 0] = V_eff * sigma_0
        edge_potential[y*4 + 1] = V_eff * sigma_0
        # 空穴轨道
        edge_potential[y*4 + 2] = -V_eff * sigma_0
        edge_potential[y*4 + 3] = -V_eff * sigma_0

# 5. 叠加到模型
ribbon_model.set_onsite(edge_potential, mode="add")

print(f"已在左右边缘各 {well_width} 层添加线性衰减的边缘势，最大 {V_edge} eV")



# --- 7. 计算一维能带 ---
# 一维布里渊区路径：从 -pi 到 pi (在约化坐标下是 -0.5 到 0.5)
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