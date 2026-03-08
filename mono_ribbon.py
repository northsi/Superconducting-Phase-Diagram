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


    beta = ky * np.sqrt(3)/ 2.0  # 假设 beta = ky/2，根据你的具体晶格常数设定
        
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
    
    # 3. 添加 Ising SOC (仅存在于原胞内的 h1 矩阵)
    # 形式通常为 lambda * Sz * tau_z (tau_z 作用于 dxy 和 dx2y2 组成的子空间)
    lambda_soc = p['lambda_soc'] # 假设 SOC 强度为 40 meV
    
    # SOC 的符号取决于层：因为 2H 堆叠存在空间反演，上下层的 SOC 极化相反
    soc_sign = 1.0 if layer == 'top' else -1.0
    
    Lz = np.zeros((3, 3), dtype=complex)
    Lz[1, 2] = -2j
    Lz[2, 1] =  2j

    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    H_SOC = np.kron(0.5 * lambda_soc * Lz, sigma_z)
    
    h1 += H_SOC
    
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
    T0_3x3[0, 0] = p['h00'] + p['h01'] * np.cos(ky) # d_z2 的层间耦合 (带一点 ky 色散作为示例)
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
    
    return H_bilayer

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
    
    # 双层层间耦合参数
    'h00': -0.030930, 
    'h01': 0.052870,
    'h02': 0.099913, 
    'h03': 0.099914,
}

N_width = 100  # Ribbon 的宽度 (原胞数量)
ky_values = np.linspace(-np.pi, np.pi, 100)
bands = []

print("正在计算双层能带，请稍候...")
for ky in ky_values:
    # 注意这里多传了一个参数 p
    #H_k = build_bilayer_ribbon(ky, p, N_width)
    H_k = build_single_layer_ribbon(ky, p, N_width)
    evals = np.linalg.eigvalsh(H_k)
    bands.append(evals)

bands = np.array(bands)
print("计算完成！准备绘图。")

plt.figure(figsize=(8, 6))
for i in range(bands.shape[1]):
    plt.plot(ky_values, bands[:, i] + 0.696245, color='black', linewidth=0.5)

plt.xlabel(r'$k_y$', fontsize=14)
plt.ylabel(r'$E(k_y)$', fontsize=14)
plt.title('Bilayer 2H-NbSe2 Nanoribbon Band Structure', fontsize=14)
plt.xlim(-np.pi, np.pi)
plt.ylim(-5, 5) # 根据出图情况调整这里的范围，方便观察费米能级 (E=0) 附近的边缘态
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()