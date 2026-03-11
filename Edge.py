import numpy as np
import matplotlib.pyplot as plt

def get_hn_matrices(beta, p):
    """生成 S16-S20 定义的 3x3 轨道跳跃矩阵"""
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_2b, sin_2b = np.cos(2*beta), np.sin(2*beta)
    sq3 = np.sqrt(3)
    
    h1 = np.zeros((3, 3), dtype=complex)
    h1[0,0] = p['e1'] + 2*p['r0']*cos_2b
    h1[0,2] = 2*(p['r1']+p['r2'])/sq3 * cos_2b + 1j * 2*(p['r1']-p['r2'])/sq3 * sin_2b
    h1[1,1] = p['e2'] + 2*(p['r11']+sq3*p['r12'])*cos_2b
    h1[2,0] = np.conj(h1[0,2])
    h1[2,2] = p['e2'] + (2*p['r11'] - 2/sq3 * p['r12'])*cos_2b
    
    h2 = np.zeros((3, 3), dtype=complex)
    h2[0,0] = 2*p['t0']*cos_b
    h2[0,1] = -1j*sq3*p['t2']*sin_b - p['t1']*cos_b
    h2[0,2] = -p['t2']*cos_b + 1j*sq3*p['t1']*sin_b
    h2[1,0] = -1j*sq3*p['t2']*sin_b + p['t1']*cos_b
    h2[1,1] = 0.5*(p['t11']+3*p['t22'])*cos_b
    h2[1,2] = 1j*sq3/2*(p['t22']-p['t11'])*sin_b + 2*p['t12']*cos_b
    h2[2,0] = -p['t2']*cos_b - 1j*sq3*p['t1']*sin_b
    h2[2,1] = 1j*sq3/2*(p['t22']-p['t11'])*sin_b - 2*p['t12']*cos_b
    h2[2,2] = 0.5*(3*p['t11']+p['t22'])*cos_b
    
    h3 = np.zeros((3, 3), dtype=complex)
    h3[0,0] = p['t0'] + 2*p['u0']*cos_2b
    h3[0,1] = -p['t1'] - 1j*sq3*p['u2']*sin_2b - p['u1']*cos_2b
    h3[0,2] = p['t2'] - p['u2']*cos_2b + 1j*sq3*p['u1']*sin_2b
    h3[1,0] = p['t1'] - 1j*sq3*p['u2']*sin_2b + p['u1']*cos_2b
    h3[1,1] = p['t11'] + 0.5*(p['u11']+3*p['u22'])*cos_2b
    h3[1,2] = -p['t12'] + 2*p['u12']*cos_2b + 1j*sq3/2*(p['u22']-p['u11'])*sin_2b
    h3[2,0] = p['t2'] - p['u2']*cos_2b - 1j*sq3*p['u1']*sin_2b
    h3[2,1] = p['t12'] - 2*p['u12']*cos_2b + 1j*sq3/2*(p['u22']-p['u11'])*sin_2b
    h3[2,2] = p['t22'] + 0.5*(3*p['u11']+p['u22'])*cos_2b

    h4 = np.zeros((3, 3), dtype=complex)
    h4[0,0] = 2*p['r0']*cos_b
    h4[0,1] = 1j*(p['r1']+p['r2'])*sin_b - (p['r1']-p['r2'])*cos_b
    h4[0,2] = -(p['r1']+p['r2'])/sq3*cos_b + 1j*(p['r1']-p['r2'])/sq3*sin_b
    h4[1,0] = 1j*(p['r1']+p['r2'])*sin_b + (p['r1']-p['r2'])*cos_b
    h4[1,1] = 2*p['r11']*cos_b
    h4[1,2] = 1j*2*p['r12']*sin_b
    h4[2,0] = -(p['r1']+p['r2'])/sq3*cos_b - 1j*(p['r1']-p['r2'])/sq3*sin_b
    h4[2,1] = 1j*2*p['r12']*sin_b
    h4[2,2] = (2*p['r11'] + 4/sq3*p['r12'])*cos_b

    h5 = np.array([[p['u0'], -p['u1'], p['u2']],
        [p['u1'], p['u11'], -p['u12']],
        [p['u2'], p['u12'], p['u22']]
    ], dtype=complex)

    return h1, h2, h3, h4, h5

def get_6x6_h_blocks(ky, p, layer='top'):

    beta = ky * np.sqrt(3) / 2.0  # 假设 beta = ky/2，根据你的具体晶格常数设定
    if layer == 'bottom':
        beta = -beta
        
    # 调用你写好的函数获取 3x3 矩阵
    h1_3x3, h2_3x3, h3_3x3, h4_3x3, h5_3x3 = get_hn_matrices(beta, p)
    
    # 2. 定义 2x2 自旋单位阵
    I_spin = np.eye(2, dtype=complex)
    
    # 将 3x3 扩展为 6x6 (基矢顺序设为: dz2_up, dz2_dn, dxy_up, dxy_dn, dx2y2_up, dx2y2_dn)
    h1 = np.kron(h1_3x3, I_spin)
    h2 = np.kron(h2_3x3, I_spin)
    h3 = np.kron(h3_3x3, I_spin)
    h4 = np.kron(h4_3x3, I_spin)
    h5 = np.kron(h5_3x3, I_spin)
    
    lambda_soc = p['lambda_soc'] # 假设 SOC 强度为 40 meV
    
    Lz = np.zeros((3, 3), dtype=complex)
    Lz[1, 2] = -2j
    Lz[2, 1] =  2j

    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    H_SOC_Ising = np.kron(0.5 * lambda_soc * Lz, sigma_z)
    h1 += H_SOC_Ising

    # ==========================================
    # 2. Rashba SOC (仅限 Top Layer)
    # ==========================================
    if layer == 'top':
        aR = p.get('aR', 0.01)  # 你的 aR 系数
        
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        
        # 对于 h2 (跨越 1 个原胞的向右跳跃 e^{ia})
        spin_mat_h2 = (-0.5j * np.cos(beta)) * sigma_y - (0.5 * np.sqrt(3) * np.sin(beta)) * sigma_x
        R1 = np.kron(aR * np.eye(3, dtype=complex), spin_mat_h2)
        
        # 对于 h3 (跨越 2 个原胞的向右跳跃 e^{i2a})
        spin_mat_h3 = (-0.5j) * sigma_y
        R2 = np.kron(aR * np.eye(3, dtype=complex), spin_mat_h3)
        
        # 将 Rashba 跳跃叠加到原来的紧束缚跳跃上
        h2 += R1
        h3 += R2
    
    return h1, h2, h3, h4, h5

def build_single_layer_ribbon(ky, p, N, layer='top'):
    """构建 N 个原胞宽度的单层纳米带哈密顿量"""
    h1, h2, h3, h4, h5 = get_6x6_h_blocks(ky, p, layer=layer)
    n_orb = 6
    
    H_ribbon = np.zeros((N * n_orb, N * n_orb), dtype=complex)
    
    for i in range(N):
        # 对角块 (On-site)
        H_ribbon[i*n_orb:(i+1)*n_orb, i*n_orb:(i+1)*n_orb] = h1
        
        # 非对角块 (Hopping)
        if layer == 'top':
            if i + 1 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+1)*n_orb:(i+2)*n_orb] = h2
                H_ribbon[(i+1)*n_orb:(i+2)*n_orb, i*n_orb:(i+1)*n_orb] = h2.conj().T
            if i + 2 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+2)*n_orb:(i+3)*n_orb] = h3
                H_ribbon[(i+2)*n_orb:(i+3)*n_orb, i*n_orb:(i+1)*n_orb] = h3.conj().T
            if i + 3 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+3)*n_orb:(i+4)*n_orb] = h4
                H_ribbon[(i+3)*n_orb:(i+4)*n_orb, i*n_orb:(i+1)*n_orb] = h4.conj().T
            if i + 4 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+4)*n_orb:(i+5)*n_orb] = h5
                H_ribbon[(i+4)*n_orb:(i+5)*n_orb, i*n_orb:(i+1)*n_orb] = h5.conj().T
                
        elif layer == 'bottom':
            # 底层的反演等效处理：h_n 的共轭转置
            if i + 1 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+1)*n_orb:(i+2)*n_orb] = h2.conj().T
                H_ribbon[(i+1)*n_orb:(i+2)*n_orb, i*n_orb:(i+1)*n_orb] = h2
            if i + 2 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+2)*n_orb:(i+3)*n_orb] = h3.conj().T
                H_ribbon[(i+2)*n_orb:(i+3)*n_orb, i*n_orb:(i+1)*n_orb] = h3
            if i + 3 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+3)*n_orb:(i+4)*n_orb] = h4.conj().T
                H_ribbon[(i+3)*n_orb:(i+4)*n_orb, i*n_orb:(i+1)*n_orb] = h4
            if i + 4 < N:
                H_ribbon[i*n_orb:(i+1)*n_orb, (i+4)*n_orb:(i+5)*n_orb] = h5.conj().T
                H_ribbon[(i+4)*n_orb:(i+5)*n_orb, i*n_orb:(i+1)*n_orb] = h5

    return H_ribbon

def build_bilayer_ribbon(ky, p, N):
    """
    组合双层 Ribbon 的哈密顿量，并将参数字典 p 传递给单层构建函数。
    """
    # 传入了参数字典 p
    H_top = build_single_layer_ribbon(ky, p, N, layer='top')
    H_bot = build_single_layer_ribbon(ky, p, N, layer='bottom')
    
    # 构建层间耦合矩阵 T_inter
    # 原胞内 6x6 层间跳跃 (这里做了简化，仅考虑原胞内的垂直对齐跳跃)
    T0_3x3 = np.zeros((3, 3), dtype=complex)
    T0_3x3[0, 0] = p['h00']  
    T0_3x3[1, 1] = p['h02']                         # d_xy 
    T0_3x3[2, 2] = p['h03']                         # d_x2y2
    
    # 扩展为 6x6 (带自旋)
    T0_6x6 = np.kron(T0_3x3, np.eye(2))
    
    # 组装 N 个原胞的完整层间耦合大矩阵
    T_inter = np.zeros_like(H_top, dtype=complex)
    n_orb = 6
    for i in range(N):
        T_inter[i*n_orb:(i+1)*n_orb, i*n_orb:(i+1)*n_orb] = T0_6x6
        
    # 组装超级矩阵
    H_bilayer = np.block([
        [H_top, T_inter],
        [T_inter.conj().T, H_bot]
    ])

    I_bilayer = np.eye(H_bilayer.shape[0], dtype=complex)
    
    return H_bilayer + 0.696245 * I_bilayer

def get_delta_blocks(ky, p):
    beta = ky * np.sqrt(3) / 2.0  
        
    omega = np.exp(1j * 2 * np.pi / 3)
    
    # 严格按照傅里叶逆变换提取系数 (跳跃方向必须和正常态 h2, h3 匹配)
    S2 = -np.eye(2, dtype=complex)
    S_minus2 = np.eye(2, dtype=complex)
    
    # 修正：S1 对应向右跳跃 (e^{i k_x'})
    S1 = np.array([
        [omega**2 * np.exp(1j*beta) + omega * np.exp(-1j*beta), 0],
        [0, omega * np.exp(1j*beta) + omega**2 * np.exp(-1j*beta)]
    ], dtype=complex)
    
    # 修正：S_minus1 对应向左跳跃 (e^{-i k_x'})
    S_minus1 = np.array([
        [-omega * np.exp(1j*beta) - omega**2 * np.exp(-1j*beta), 0],
        [0, -omega**2 * np.exp(1j*beta) - omega * np.exp(-1j*beta)]
    ], dtype=complex)
    
    # 提取轨道矩阵块 d_z2
    O_pair = np.zeros((3, 3), dtype=complex)
    O_pair[0, 0] = 1.0  
    
    # 必须把 delta_0 调小到合理区间，例如 0.05
    delta_0 = p.get('delta_0', 0.05)
    delta_1 = 0.05 
    
    # 先正常生成四个配对矩阵
    D2 = delta_0 * np.kron(O_pair, S2)
    D_minus2 = delta_0 * np.kron(O_pair, S_minus2)
    D1 = delta_0 * np.kron(O_pair, S1)
    D_minus1 = delta_0 * np.kron(O_pair, S_minus1)
    # 将所有3~6列（即索引2~5）的对角线赋值为delta_0
    # for D in [D2, D_minus2, D1, D_minus1]:
    #     for j in range(2, 6):
    #         D[j, j] = delta_0

    D0 = np.zeros((6, 6), dtype=complex)
    
    # 给 dxy 轨道 (索引 2 和 3) 加上自旋单态配对 i*sigma_y
    D0[2, 3] = delta_1
    D0[3, 2] = -delta_1
    
    # 给 dx2-y2 轨道 (索引 4 和 5) 加上自旋单态配对 i*sigma_y
    D0[4, 5] = delta_1
    D0[5, 4] = -delta_1
    
    return D1, D_minus1, D2, D_minus2, D0

# ==========================================================
# 步骤 2：组装单层 Ribbon 的配对矩阵 Delta(k_y)
# ==========================================================
def build_single_layer_delta(ky, p, N):
    D1, D_minus1, D2, D_minus2, D0 = get_delta_blocks(ky, p)
    n_orb = 6
    Delta_ribbon = np.zeros((N * n_orb, N * n_orb), dtype=complex)
    
    # D1 和 D2 是向 +x 方向跳跃 (填入上三角)
    # D_minus1 和 D_minus2 是向 -x 方向跳跃 (填入下三角)
    for i in range(N):

        Delta_ribbon[i*n_orb:(i+1)*n_orb, i*n_orb:(i+1)*n_orb] = D0

        if i + 1 < N:
            Delta_ribbon[i*n_orb:(i+1)*n_orb, (i+1)*n_orb:(i+2)*n_orb] = D1
            Delta_ribbon[(i+1)*n_orb:(i+2)*n_orb, i*n_orb:(i+1)*n_orb] = D_minus1
        if i + 2 < N:
            Delta_ribbon[i*n_orb:(i+1)*n_orb, (i+2)*n_orb:(i+3)*n_orb] = D2
            Delta_ribbon[(i+2)*n_orb:(i+3)*n_orb, i*n_orb:(i+1)*n_orb] = D_minus2
            
    return Delta_ribbon

def build_bdg_matrix(ky, p, N):

    H_k = build_bilayer_ribbon(ky, p, N)
    H_minus_k = build_bilayer_ribbon(-ky, p, N)
    
    # 2. 超导配对矩阵 Delta(k)
    Delta_top = build_single_layer_delta(ky, p, N)
    Delta_bot = build_single_layer_delta(ky, p, N)
    
    # 组装双层的配对矩阵
    Delta_bilayer = np.block([
        [Delta_top, np.zeros_like(Delta_top)],
        [np.zeros_like(Delta_bot), Delta_bot]
    ])
    
    H_BdG = np.block([
        [H_k, Delta_bilayer],
        [Delta_bilayer.conj().T, -H_minus_k.conj()]
    ])
    
    return H_BdG

# ==========================================
# 运行与画图部分
# ==========================================
# 把你发给我的字典定义在这里 (包含 SOC 参数)
p = {
    # 单层正常态跳跃参数 & SOC
    'e1': 1.946600,  'e2': 1.349600,
    't0': -0.420573, 't1': 0.221238,  't2': 0.254557,
    't11': 0.779500, 't12': 0.231953, 't22': -0.129562,
    'r0': 0.410334,  'r1': -0.295446, 'r2': -0.274108,
    'r11': 0.175777, 'r12': -0.006219,
    'u0': -0.319703, 'u1': 0.257766,  'u2': -0.001795,
    'u11': 0.157093, 'u12': -0.004888, 'u22': 0.072577,
    'lambda_soc': 0.109868,

    'aR': 0.00,
    
    'delta_0': 0.1,
    # 双层层间耦合参数
    'h00': -0.030930, 
    'h01': 0.052870,
    'h02': 0.099913, 
    'h03': 0.099914,
}

N_width = 100  
ky_values = np.linspace(-np.pi, np.pi, 101)

all_bands = []

print("正在计算 Ribbon 能带图...")

for ky in ky_values:
    # 1. 构建 BdG 矩阵
    H_bdg = build_bdg_matrix(ky, p, N_width) 
    
    # 2. 直接对角化获取本征值 (eigvalsh 专门用于 Hermitian 矩阵，且不计算特征向量)
    evals = np.linalg.eigvalsh(H_bdg)
    
    # 3. 存储结果
    all_bands.append(evals)

# 转换为 numpy 数组方便切片绘图 [len(ky_values) x num_eigenvalues]
all_bands = np.array(all_bands)

print("计算完成！开始绘图。")

# ---------------------------------------------------------
# 绘图：能带图 (Ribbon Band Structure)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# 遍历每一条能带进行绘制
# all_bands.shape[1] 是本征值的数量（即矩阵维度）
for i in range(all_bands.shape[1]):
    plt.plot(ky_values, all_bands[:, i], color='black', linewidth=0.5, alpha=0.6)

# 设置坐标轴标签和范围
plt.xlabel(r'$k_y$', fontsize=14)
plt.ylabel(r'$E(k_y)$', fontsize=14)
plt.title('Ribbon BdG Band Structure', fontsize=14)

plt.xlim(-np.pi, np.pi)
plt.ylim(-0.1, 0.1)  # 根据需要调整能量范围
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('threeband_ribbon.png', dpi=300)
plt.show()

# def plot_single_ky_state(ky_target, p, N_width):
#     """
#     计算并绘制指定 ky 点的能谱和最靠近 E=0 的波函数。
#     """
#     print(f"正在计算 ky = {ky_target:.4f} 处的 BdG 矩阵，请稍候...")
    
#     # 这里的函数名请确保与你代码中定义的 BdG 组装函数一致
#     # 如果你用的是前面 s-wave 的代码，这里可能是 build_bdg_matrix_swave
#     H_bdg = build_bdg_matrix(ky_target, p, N_width) 
    
#     evals, evecs = np.linalg.eigh(H_bdg)
    
#     # 1. 寻找最靠近 E=0 的本征态
#     zero_idx = np.argmin(np.abs(evals))
#     E_zero = evals[zero_idx]
#     psi_zero = evecs[:, zero_idx]
    
#     # 2. 计算该波函数的实空间概率分布
#     prob = np.abs(psi_zero)**2
#     weight_per_cell = np.zeros(N_width)
#     for block in range(4): # 电子/空穴, 顶层/底层 四个区块
#         block_prob = prob[block * 6 * N_width : (block + 1) * 6 * N_width]
#         weight_per_cell += np.sum(block_prob.reshape(N_width, 6), axis=1)
        
#     # ==========================================
#     # 开始绘图 (1行2列的双子图)
#     # ==========================================
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     # --- 左图：该 ky 点的局部能谱 ---
#     # 为了图像清晰，只筛选出带隙附近的能级 (-0.2 到 0.2 eV)
#     close_evals = evals[np.abs(evals) < 0.2]
#     axes[0].plot(np.zeros_like(close_evals), close_evals, 'ko', markersize=3, alpha=0.4, label='Bulk/Other States')
#     # 高亮标出离 E=0 最近的那个态
#     axes[0].plot([0], [E_zero], 'ro', markersize=8, label=f'Closest State\n(E = {E_zero:.6f} eV)')
    
#     axes[0].set_xlim(-1, 1)
#     axes[0].set_xticks([]) # 隐藏不必要的 X 轴刻度
#     axes[0].set_ylabel('Energy (eV)', fontsize=12)
#     axes[0].set_title(f'Energy Levels at $k_y = {ky_target:.4f}$', fontsize=14)
#     axes[0].grid(True, linestyle='--', alpha=0.6)
#     axes[0].legend(loc='best')
    
#     # --- 右图：该态的实空间波函数分布 ---
#     site_indices = np.arange(N_width)
#     axes[1].plot(site_indices, weight_per_cell, marker='o', color='red', linewidth=2, markersize=5)
#     axes[1].fill_between(site_indices, weight_per_cell, color='red', alpha=0.3)
    
#     axes[1].set_xlabel('Site Index (Nanoribbon Width)', fontsize=12)
#     axes[1].set_ylabel(r'Probability Density $|\psi|^2$', fontsize=12)
#     axes[1].set_title(f'Wavefunction profile for $E = {E_zero:.6f}$', fontsize=14)
#     axes[1].set_xlim(0, N_width - 1)
#     axes[1].set_ylim(0, max(weight_per_cell) * 1.1)
#     axes[1].grid(True, linestyle='--', alpha=0.6)
    
#     plt.tight_layout()
#     plt.show()

# ==========================================
# 测试调用示例
# ==========================================
# 假设你想看 ky = 0 处的波函数情况：
# plot_single_ky_state(0.0, p, 30)

# 如果你从你的大能带图上发现边缘态在某个特定的 ky (比如 1.5)，你就可以精确打击：
# plot_single_ky_state(1.5, p, 30)
# ---------------------------------------------------------
# 图 2：边缘态在实空间的波函数概率分布 (Wavefunction Profile)
# ---------------------------------------------------------
# if best_edge_wavefunc is not None:
#     plt.figure(figsize=(8, 4))
    
#     # 画出波函数的分布曲线
#     site_indices = np.arange(N_width)
#     plt.plot(site_indices, best_edge_wavefunc, marker='o', color='red', linewidth=2, markersize=5)
    
#     # 把曲线下方的面积涂色，看起来更直观
#     plt.fill_between(site_indices, best_edge_wavefunc, color='red', alpha=0.3)
    
#     plt.xlabel('Site Index (Width of the Ribbon)', fontsize=14)
#     plt.ylabel(r'Probability Density $|\psi|^2$', fontsize=14)
#     plt.title(f'Wavefunction of Edge Mode (at $k_y = {best_edge_ky:.2f}$, $E = {best_edge_E:.4f}$)', fontsize=14)
#     plt.xlim(0, N_width - 1)
#     plt.ylim(0, max(best_edge_wavefunc) * 1.1)  # y轴稍微留点空隙
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()
# else:
#     print("没有找到符合条件的边缘态，无法绘制波函数分布图！")