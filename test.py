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
    """正常态哈密顿量，加入 Zeeman 和 Rashba"""
    beta = ky * np.sqrt(3) / 2.0  
    if layer == 'bottom':
        beta = -beta
        
    h1_3x3, h2_3x3, h3_3x3, h4_3x3, h5_3x3 = get_hn_matrices(beta, p)
    
    I_spin = np.eye(2, dtype=complex)
    h1 = np.kron(h1_3x3, I_spin)
    h2 = np.kron(h2_3x3, I_spin)
    h3 = np.kron(h3_3x3, I_spin)
    h4 = np.kron(h4_3x3, I_spin)
    h5 = np.kron(h5_3x3, I_spin)
    
    # 1. Ising SOC
    lambda_soc = p.get('lambda_soc', 0.109868) 
    soc_sign = 1.0 if layer == 'top' else -1.0
    Lz = np.zeros((3, 3), dtype=complex)
    Lz[1, 2], Lz[2, 1] = -2j, 2j
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    h1 += np.kron(0.5 * lambda_soc * soc_sign * Lz, sigma_z)
    
    # 2. 面内塞曼磁场 Vz (拓扑相变的关键)
    Vz = p.get('Vz', 0.0)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    h1 += np.kron(np.eye(3), sigma_x) * Vz

    # 3. Rashba SOC
    aR = p.get('aR', 0.0)
    if aR != 0.0 and layer == 'top':
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        spin_mat_h2 = (-0.5j * np.cos(beta)) * sigma_y - (0.5 * np.sqrt(3) * np.sin(beta)) * sigma_x
        spin_mat_h3 = (-0.5j) * sigma_y
        h2 += np.kron(aR * np.eye(3, dtype=complex), spin_mat_h2)
        h3 += np.kron(aR * np.eye(3, dtype=complex), spin_mat_h3)

    return h1, h2, h3, h4, h5

def build_bdg_matrix_swave(ky, p, N):
    """组装 s-wave BdG 矩阵"""
    H_k = build_bilayer_ribbon(ky, p, N)
    H_minus_k = build_bilayer_ribbon(-ky, p, N)
    
    # 显式加入化学势，保证严格的粒子-空穴对称性
    mu = p.get('mu', -0.696245)
    I_12N = np.eye(H_k.shape[0], dtype=complex)
    H_e = H_k - mu * I_12N
    H_h = -H_minus_k.conj() + mu * I_12N
    
    # ==========================================
    # s-wave 核心配对代码：无节点，完美全能隙
    # ==========================================
    delta_0 = p.get('delta_0', 0.05)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    # 原胞内 s-wave 配对矩阵: 1j * sigma_y
    D_onsite = np.kron(np.eye(3), 1j * sigma_y * delta_0)
    
    # 扩展到整条 Ribbon 和双层
    Delta_single = np.kron(np.eye(N), D_onsite)
    Delta_bilayer = np.block([
        [Delta_single, np.zeros_like(Delta_single)],
        [np.zeros_like(Delta_single), Delta_single]
    ])
    
    H_BdG = np.block([
        [H_e, Delta_bilayer],
        [Delta_bilayer.conj().T, H_h]
    ])
    
    return H_BdG

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
        [omega**2 * np.exp(-1j*beta) + omega * np.exp(1j*beta), 0],
        [0, omega * np.exp(-1j*beta) + omega**2 * np.exp(1j*beta)]
    ], dtype=complex)
    
    # 修正：S_minus1 对应向左跳跃 (e^{-i k_x'})
    S_minus1 = np.array([
        [-omega**2 * np.exp(1j*beta) - omega * np.exp(-1j*beta), 0],
        [0, -omega * np.exp(1j*beta) - omega**2 * np.exp(-1j*beta)]
    ], dtype=complex)
    
    # 提取轨道矩阵块 d_z2
    O_pair = np.zeros((3, 3), dtype=complex)
    O_pair[0, 0] = 1.0  
    
    # 必须把 delta_0 调小到合理区间，例如 0.05
    delta_0 = p.get('delta_0', 0.05) 
    
    D2 = delta_0 * np.kron(O_pair, S2)
    D_minus2 = delta_0 * np.kron(O_pair, S_minus2)
    D1 = delta_0 * np.kron(O_pair, S1)
    D_minus1 = delta_0 * np.kron(O_pair, S_minus1)
    
    return D1, D_minus1, D2, D_minus2

# ==========================================================
# 步骤 2：组装单层 Ribbon 的配对矩阵 Delta(k_y)
# ==========================================================
def build_single_layer_delta(ky, p, N):
    D1, D_minus1, D2, D_minus2 = get_delta_blocks(ky, p)
    n_orb = 6
    Delta_ribbon = np.zeros((N * n_orb, N * n_orb), dtype=complex)
    
    # D1 和 D2 是向 +x 方向跳跃 (填入上三角)
    # D_minus1 和 D_minus2 是向 -x 方向跳跃 (填入下三角)
    for i in range(N):
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

    # 'mu': -0.696245,
    # 'delta_0': 0.05,
    # 'Vz': 0.0,       # 关闭磁场
    # 'aR': 0.0,       # 关闭 Rashba

    'mu': -0.696245,
    'delta_0': 0.05,
    'Vz': 0.08,      # 开启塞曼磁场，且 Vz > delta_0
    'aR': 0.05,      # 开启 Rashba SOC
    
    # 双层层间耦合参数
    'h00': -0.030930, 
    'h01': 0.052870,
    'h02': 0.099913, 
    'h03': 0.099914,
}

N_width = 60  
ky_values = np.linspace(-np.pi, np.pi, 100)

bulk_bands = []
edge_bands = []

print("正在对角化 BdG 超大矩阵并寻找边缘态，请稍候...")
for ky in ky_values:
    H_bdg = build_bdg_matrix_swave(ky, p, N_width)
    
    # 使用 eigh 同时获取本征值和波函数 (特征向量)
    evals, evecs = np.linalg.eigh(H_bdg)
    
    ky_bulk = []
    ky_edge = []
    
    for j in range(len(evals)):
        E = evals[j]
        # 提速小技巧：边缘态必然在超导能隙内部或附近，直接跳过高能体态
        if abs(E) > 0.2:
            ky_bulk.append(E)
            continue
            
        psi = evecs[:, j]
        prob = np.abs(psi)**2
        
        # 波函数长度为 24N (4个区块：Top-E, Bot-E, Top-H, Bot-H，每个 6N)
        # 我们把概率密度折叠到 N 个实空间原胞上
        weight_per_cell = np.zeros(N_width)
        for block in range(4):
            block_prob = prob[block * 6 * N_width : (block + 1) * 6 * N_width]
            # 对每个原胞内的 6 个自旋轨道求和
            weight_per_cell += np.sum(block_prob.reshape(N_width, 6), axis=1)
            
        # 判断标准：计算两端各 10% 宽度的波函数权重
        edge_depth = max(1, N_width // 10)
        edge_weight = np.sum(weight_per_cell[:edge_depth]) + np.sum(weight_per_cell[-edge_depth:])
        
        # 如果大于 60% 的概率集中在边缘，则判定为 Edge mode
        if edge_weight > 0.6:
            ky_edge.append(E)
        else:
            ky_bulk.append(E)
            
    bulk_bands.append(ky_bulk)
    edge_bands.append(ky_edge)

print("计算完成！准备绘图。")

plt.figure(figsize=(8, 6))

# 绘制体态 (黑色细点)
for i, ky in enumerate(ky_values):
    if bulk_bands[i]:
        plt.plot([ky]*len(bulk_bands[i]), bulk_bands[i], marker='o', color='black', markersize=1.5, linestyle='None', alpha=0.3)

# 绘制边缘态 (红色粗点，极其醒目)
for i, ky in enumerate(ky_values):
    if edge_bands[i]:
        plt.plot([ky]*len(edge_bands[i]), edge_bands[i], marker='o', color='red', markersize=4, linestyle='None', zorder=5)

plt.xlabel(r'$k_y$', fontsize=14)
plt.ylabel(r'$E(k_y)$', fontsize=14)
plt.title('Bilayer 2H-NbSe2 BdG Band Structure (Edge Modes in Red)', fontsize=14)
plt.xlim(-np.pi, np.pi)
plt.ylim(-1, 1) 
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()