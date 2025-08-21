import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from scipy.optimize import brentq
import warnings
from functools import lru_cache

@dataclass
class Parameters:
    hbar: float = 1.0
    m_star: float = 0.5
    mu: float = 0.0
    beta_so_meV: float = 2.0
    m_alphaR2_over2_meV: float = 8.0

    # k mesh
    Nk: int = 61
    k_max: float = 0.5          

    # Temperature range for Tc calculation
    T_min: float = 0.001
    T_max: float = 0.1

    # ---- 2D coupling grids (NEW) ----
    v0_min: float = 0.01
    v0_max: float = 0.2
    Nv0: int = 20

    v1_min: float = 0.01
    v1_max: float = 0.2
    Nv1: int = 20

    def __post_init__(self):
        # Convert to eV 
        self.beta_so: float = 1e-3 * self.beta_so_meV  
        self.alpha_R: float = float(
            np.sqrt(2.0 * 1e-3 * self.m_alphaR2_over2_meV / self.m_star)
        )
        self.v0_range = np.linspace(self.v0_min, self.v0_max, self.Nv0)
        self.v1_range = np.linspace(self.v1_min, self.v1_max, self.Nv1)


class PauliMatrices:
    def __init__(self):
        # layer 
        self.tau_0 = np.eye(2, dtype=complex)
        self.tau_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.tau_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # spin
        self.sigma_0 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

class IrrepBasisFunctions:
    
    def __init__(self, pauli: PauliMatrices):
        self.pauli = pauli
        self.irrep_list = self._define_irrep_basis_functions()
        self.num_irreps = len(self.irrep_list)
    
    def _get_form_factors(self, k: np.ndarray) -> Dict[str, complex]:
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        
        omega = np.exp(1j * 2 * np.pi / 3)
        
        c_k = np.sum(np.cos(k_dot_R))
        s_k = np.sum(np.sin(k_dot_R))
        
        omega_weights_plus = np.array([1, omega, omega**2])
        omega_weights_minus = np.array([1, omega**2, omega])
        
        c_k_plus = np.sum(omega_weights_plus * np.cos(k_dot_R))
        c_k_minus = np.sum(omega_weights_minus * np.cos(k_dot_R))
        s_k_plus = np.sum(omega_weights_plus * np.sin(k_dot_R))
        s_k_minus = np.sum(omega_weights_minus * np.sin(k_dot_R))
        
        return {
            'const': 1.0,
            'c_k': c_k,
            's_k': s_k,
            'c_k_plus': c_k_plus,
            'c_k_minus': c_k_minus,
            's_k_plus': s_k_plus,
            's_k_minus': s_k_minus
        }
    
    def _define_irrep_basis_functions(self) -> List[Dict]:

        p = self.pauli
        
        irrep_list = []
        
        # 不可约表示 0: A1g (单重态)
        irrep_0 = {
            'label': 'A1_xy',
            'singlet': [
                # psi = 1 * (iσ_y)
                lambda k, ff: 1.0 * 1j * p.sigma_y,
                # psi = e_k * (iσ_y) 
                lambda k, ff: ff['c_k'] * 1j * p.sigma_y
            ],
            'triplet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: (ff['s_k_minus'] * (p.sigma_x + 1j* p.sigma_y) - ff['s_k_plus'] * (p.sigma_x - 1j* p.sigma_y)) * 1j * p.sigma_y
            ]  
        }
        irrep_list.append(irrep_0)
        
        irrep_1 = {
            'label': 'A1_z',
            'singlet': [
                # psi = 1 * (iσ_y)
                lambda k, ff: 1.0 * 1j * p.sigma_y,
                # psi = e_k * (iσ_y) 
                lambda k, ff: ff['c_k'] * 1j * p.sigma_y
            ],
            'triplet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: ff['s_k'] * p.sigma_z * 1j * p.sigma_y
            ]  
        }
        irrep_list.append(irrep_1)

        irrep_2 = {
            'label': 'A2', 
            'singlet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: 0.0 * 1j * p.sigma_y,
            ],
            'triplet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,
                
                lambda k, ff: (ff['s_k_minus'] * (p.sigma_x + 1j* p.sigma_y) + ff['s_k_plus'] * (p.sigma_x - 1j* p.sigma_y)) * 1j * p.sigma_y
            ]
        }
        irrep_list.append(irrep_2)
        
        irrep_3 = {
            'label': 'E_+',
            'singlet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: ff['c_k_plus'] * 1j * p.sigma_y
            ],
            'triplet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: ff['s_k_plus'] * p.sigma_z * 1j * p.sigma_y
            ]
        }
        irrep_list.append(irrep_3)

        irrep_4 = {
            'label': 'E_-',
            'singlet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: ff['c_k_minus'] * 1j * p.sigma_y
            ],
            'triplet': [
                lambda k, ff: 0.0 * 1j * p.sigma_y,

                lambda k, ff: ff['s_k_minus'] * p.sigma_z * 1j * p.sigma_y
            ]
        }
        irrep_list.append(irrep_4)
        
        return irrep_list
    
    def get_basis_functions_for_irrep(self, irrep_idx: int) -> Tuple[List, List]:

        if irrep_idx < 0 or irrep_idx >= self.num_irreps:
            raise ValueError(f"Irrep index {irrep_idx} out of range [0, {self.num_irreps-1}]")
        
        irrep_data = self.irrep_list[irrep_idx]
        
        # 生成实际的函数
        psi_funcs = []
        for func_template in irrep_data['singlet']:
            def make_psi_func(template):
                def psi_func(k):
                    ff = self._get_form_factors(k)
                    return template(k, ff)
                return psi_func
            psi_funcs.append(make_psi_func(func_template))
        
        d_funcs = []
        for func_template in irrep_data['triplet']:
            def make_d_func(template):
                def d_func(k):
                    ff = self._get_form_factors(k)
                    return template(k, ff)
                return d_func
            d_funcs.append(make_d_func(func_template))
        
        return psi_funcs, d_funcs

class SuperconductorSolver:
    
    def __init__(self, params: Parameters):
        self.params = params
        self.pauli = PauliMatrices()
        self.irrep_basis = IrrepBasisFunctions(self.pauli)
        self.k_grid = self._setup_k_grid()
        
    def _setup_k_grid(self) -> np.ndarray:
        kx, ky = np.meshgrid(
            np.linspace(-self.params.k_max, self.params.k_max, self.params.Nk),
            np.linspace(-self.params.k_max, self.params.k_max, self.params.Nk)
        )
        return np.column_stack([kx.ravel(), ky.ravel()])
    
    def _get_hamiltonian(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params
        σ = self.pauli 

        kx, ky = float(k[0]), float(k[1])
        k2 = kx*kx + ky*ky
        xi = k2 / (2.0 * p.m_star) - p.mu

        H0 = xi * σ.sigma_0
        H_R = p.alpha_R * (ky * σ.sigma_x - kx * σ.sigma_y)
        H_I = p.beta_so * σ.sigma_z      

        H_k  = H0 + H_R + H_I
        H_mk = H0 - H_R + H_I    
        H_mk = H_mk.conj()            

        return H_k, H_mk
    
    @staticmethod
    def _eigvals_and_projectors_grouped(H: np.ndarray, tol_group: float = 1e-8) -> Tuple[np.ndarray, List[np.ndarray]]:
        evals, evecs = np.linalg.eigh(H)
        
        # 按本征值分组
        grouped_vals: List[float] = []
        projectors: List[np.ndarray] = []

        n = len(evals)
        i = 0
        while i < n:
            E0 = evals[i]
            j = i + 1
            while j < n and np.abs(evals[j] - E0) <= tol_group:
                j += 1
            
            V = evecs[:, i:j]            
            P = V @ V.conj().T
            
            projectors.append(P)
            grouped_vals.append(np.mean(evals[i:j])) 
            i = j

        return np.asarray(grouped_vals), projectors

    @staticmethod
    def _matsubara_sum(Ej: float, El: float, T: float, tol: float = 1e-10) -> float:
        beta = 1.0 / T
        
        def sech2(x):
            c = np.cosh(x)
            return 1.0 / (c * c)

        if abs(Ej - El) <= tol:
            if abs(Ej) <= tol:
                return beta / 4.0
            else:
                return np.tanh(beta * Ej / 2.0) / (2.0 * Ej)

        if abs(Ej + El) <= tol:
            return 0.25 * beta * sech2(beta * Ej / 2.0)

        return (np.tanh(beta * Ej / 2.0) + np.tanh(beta * El / 2.0)) / (2.0 * (Ej + El))

    def _calculate_susceptibility_for_irrep(self, irrep_idx: int, T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
       
        psi_funcs, d_funcs = self.irrep_basis.get_basis_functions_for_irrep(irrep_idx)
        
        num_psi = len(psi_funcs)
        num_d = len(d_funcs)
        
        I = np.zeros((num_psi, num_psi), dtype=complex)
        J = np.zeros((num_d, num_d), dtype=complex)
        K = np.zeros((num_psi, num_d), dtype=complex)

        for k in self.k_grid:
            eigenvals_e, eigenvals_h, proj_e, proj_h = self._get_spectral_data(k)
            
            # 计算该k点的基函数值
            psi_matrices = [func(k) for func in psi_funcs] if num_psi > 0 else []
            d_matrices = [func(k) for func in d_funcs] if num_d > 0 else []
            
            # 计算所有能带对的贡献
            for j, Ej in enumerate(eigenvals_e):
                Pj = proj_e[j]
                for l, El in enumerate(eigenvals_h):
                    Pl = proj_h[l]
                    
                    S_jl = self._matsubara_sum(float(np.real(Ej)), float(np.real(El)), T)
                    if abs(S_jl) < 1e-12:
                        continue
                    
                    # I: 单重态-单重态
                    if num_psi > 0:
                        for r1 in range(num_psi):
                            A = psi_matrices[r1].conj().T @ Pj
                            for r2 in range(num_psi):
                                M = np.trace(A @ psi_matrices[r2] @ Pl)
                                I[r1, r2] += M * S_jl
                    
                    # J: 三重态-三重态
                    if num_d > 0:
                        for r1 in range(num_d):
                            A = d_matrices[r1].conj().T @ Pj
                            for r2 in range(num_d):
                                M = np.trace(A @ d_matrices[r2] @ Pl)
                                J[r1, r2] += M * S_jl
                    
                    # K: 单重态-三重态
                    if num_psi > 0 and num_d > 0:
                        for r1 in range(num_psi):
                            A = psi_matrices[r1].conj().T @ Pj
                            for r2 in range(num_d):
                                M = np.trace(A @ d_matrices[r2] @ Pl)
                                K[r1, r2] += M * S_jl

        factor = -1.0 / len(self.k_grid)
        return factor * I, factor * J, factor * K
    
    def _get_spectral_data(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        H_k, H_mk = self._get_hamiltonian(k)
        
        eigenvals_e, proj_e = self._eigvals_and_projectors_grouped(H_k)
        eigenvals_h, proj_h = self._eigvals_and_projectors_grouped(H_mk)
        
        return eigenvals_e, eigenvals_h, proj_e, proj_h

    def _get_max_eigenvalue_for_irrep(self, irrep_idx: int, T: float, v_vector: np.ndarray, degeneracy_tol: float = 1e-10) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        # Create diagonal matrix with coupling constants
        v_diag = np.diag(v_vector)
        
        # Create I_{2×2} ⊗ diag{v_r} structure
        I_2x2 = np.eye(2)
        V = np.kron(I_2x2, v_diag)

        I, J, K = self._calculate_susceptibility_for_irrep(irrep_idx, T)

        M = np.block([[I, K], [K.conj().T, J]])

        VM = V @ M
        eigenvals, eigenvectors = eig(VM)
        
        real_eigenvals = np.real(eigenvals)
        
        imag_parts = np.abs(np.imag(eigenvals))
        if np.any(imag_parts > 1e-8):
            warnings.warn(f"Large imaginary parts in eigenvalues: max = {np.max(imag_parts)}")
        
        max_eigenval = np.max(real_eigenvals)
        
        is_max = np.abs(real_eigenvals - max_eigenval) < degeneracy_tol
        max_indices = np.where(is_max)[0]
        
        max_eigenvectors = []
        for idx in max_indices:
            vec = eigenvectors[:, idx]
            vec = vec / np.linalg.norm(vec)
            max_eigenvectors.append(vec)
        
        return max_eigenval, real_eigenvals, max_eigenvectors
    
    
    """ def _find_Tc_for_irrep(self, irrep_idx: int, v_vector: np.ndarray) -> Optional[float]:
        
        # 在温度范围内搜索本征值达到1的点
        T_search = np.linspace(self.params.T_min, self.params.T_max, 100)
        eigenvals = []
        
        for T in T_search:
            max_eig, _, _= self._get_max_eigenvalue_for_irrep(irrep_idx, T, v_vector)
            eigenvals.append(max_eig)
        
        eigenvals = np.array(eigenvals)
        
        # 找到本征值最接近1的点
        diff_from_1 = np.abs(eigenvals - 1.0)
        min_idx = np.argmin(diff_from_1)
        
        # 检查是否找到合理的解
        if eigenvals[min_idx] >= 0.95:  # 允许一些误差
            return T_search[min_idx]
        else:
            return None """
        
    def _find_Tc_for_irrep(self, irrep_idx: int, v_vector: np.ndarray,
                       coarse_N: int = 24,       # 粗搜网格点数
                       xtol: float = 1e-6,       # Brent 绝对容差
                       rtol: float = 1e-6,       # Brent 相对容差
                       maxiter: int = 100,       # Brent 最大迭代
                       near_accept: bool = True, # 若无括号，是否接受“近似零点”
                       near_tol: float = 5e-2    # “近似零点”的 |lambda-1| 阈值
                      ) -> Optional[float]:
        
        T_min = float(self.params.T_min)
        T_max = float(self.params.T_max)

        def f(T: float) -> float:
            lam_max, _, _ = self._get_max_eigenvalue_for_irrep(irrep_idx, float(T), v_vector)
            return float(lam_max - 1.0)

        # 端点快速判别
        f_min = f(T_min)
        if abs(f_min) <= near_tol:
            return T_min
        f_max = f(T_max)
        if abs(f_max) <= near_tol:
            return T_max

        # 若端点已形成括号，直接 Brent
        if f_min * f_max < 0.0:
            return float(brentq(f, T_min, T_max, xtol=xtol, rtol=rtol, maxiter=maxiter))

        # 粗网格括号搜索
        Ts = np.linspace(T_min, T_max, int(coarse_N))
        Fs = np.array([f(T) for T in Ts])

        # 查找相邻点符号翻转（含命中零）
        for a, b, fa, fb in zip(Ts[:-1], Ts[1:], Fs[:-1], Fs[1:]):
            if fa == 0.0:
                return float(a)
            if fa * fb < 0.0:
                return float(brentq(f, float(a), float(b), xtol=xtol, rtol=rtol, maxiter=maxiter))

        # 没有严格括号：可选返回最接近的“近似零点”
        if near_accept:
            idx = int(np.argmin(np.abs(Fs)))
            if abs(Fs[idx]) <= near_tol:
                return float(Ts[idx])

        # 仍未找到
        return None

    def find_dominant_channel_at_coupling(self, v_vector: np.ndarray) -> Dict:
        """
        在给定耦合常数下，找到主导的超导通道（最高Tc的不可约表示）
        """
        results = {
            'v': np.array(v_vector, dtype=float).copy(),
            'irrep_Tcs': [],
            'irrep_labels': [],
            'dominant_irrep': None,
            'dominant_Tc': None
        }
        
        max_Tc = -1.0
        dominant_irrep = None
        
        for irrep_idx in range(self.irrep_basis.num_irreps):
            Tc = self._find_Tc_for_irrep(irrep_idx, v_vector)
            label = self.irrep_basis.irrep_list[irrep_idx]['label']
            
            results['irrep_Tcs'].append(Tc)
            results['irrep_labels'].append(label)
            
            if Tc is not None and Tc > max_Tc:
                max_Tc = Tc
                dominant_irrep = irrep_idx
        
        if dominant_irrep is not None:
            results['dominant_irrep'] = dominant_irrep
            results['dominant_Tc'] = max_Tc
        
        return results

    def compute_phase_diagram(self) -> Dict:
        """
        计算完整的相图：对每个v值找到主导的超导通道
        """
        p = self.params
        Nv0, Nv1 = len(p.v0_range), len(p.v1_range)

        phase_data = {
            'v0_range': p.v0_range,
            'v1_range': p.v1_range,
            # 🔧 修复：正确初始化数组
            'dominant_irreps': np.full((Nv0, Nv1), -1, dtype=int),
            'dominant_Tcs': np.full((Nv0, Nv1), np.nan),
            'all_irrep_Tcs': np.full((Nv0, Nv1, self.irrep_basis.num_irreps), np.nan),
            'irrep_labels': [irrep['label'] for irrep in self.irrep_basis.irrep_list]
        }
        
        print(f"Computing 2D phase diagram on {Nv0}×{Nv1} grid with {self.irrep_basis.num_irreps} irreps...")
        
        for i, v0 in enumerate(p.v0_range):
            for j, v1 in enumerate(p.v1_range):
                v_vector = np.array([v0, v1], dtype=float)
                result = self.find_dominant_channel_at_coupling(v_vector)
                

                if result['dominant_Tc'] is not None:
                    phase_data['dominant_irreps'][i, j] = result['dominant_irrep']
                    phase_data['dominant_Tcs'][i, j] = result['dominant_Tc']
                    
                for k, Tc in enumerate(result['irrep_Tcs']):
                    if Tc is not None:
                        phase_data['all_irrep_Tcs'][i, j, k] = Tc
                        
        return phase_data

    def plot_phase_diagram_2d(self, phase_data: Dict):
        """二维相图绘制：左图最高 Tc 热图，右图主导通道（离散色表）"""
        import matplotlib.pyplot as plt
        v0 = phase_data['v0_range']; v1 = phase_data['v1_range']
        V0, V1 = np.meshgrid(v0, v1, indexing='ij')

        Tc = phase_data['dominant_Tcs']
        dom = phase_data['dominant_irreps']

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        im0 = axes[0].pcolormesh(V0, V1, Tc, shading='nearest')
        fig.colorbar(im0, ax=axes[0], label=r'Highest $T_c$')
        axes[0].set_xlabel(r'$v_0$ (singlet)')
        axes[0].set_ylabel(r'$v_1$ (triplet)')
        axes[0].set_title('Max $T_c$ over irreps')

        # 主导通道图：-1 表示无解
        dom_plot = dom.copy().astype(float)
        dom_plot[dom_plot < 0] = np.nan
        im1 = axes[1].pcolormesh(V0, V1, dom_plot, shading='nearest', cmap='tab10',
                                 vmin=-0.5, vmax=self.irrep_basis.num_irreps - 0.5)
        cbar = fig.colorbar(im1, ax=axes[1])
        cbar.set_label('Dominant irrep index')
        axes[1].set_xlabel(r'$v_0$ (singlet)')
        axes[1].set_ylabel(r'$v_1$ (triplet)')
        axes[1].set_title('Dominant superconducting channel (irrep)')
        plt.tight_layout()
        plt.show()

    """     def analyze_coupling_point(self, v: float):
        
        print(f"\n=== Analysis for v = {v:.3f} ===")
        result = self.find_dominant_channel_at_coupling(v)
        
        print(f"Results for all irreducible representations:")
        for i, (Tc, label) in enumerate(zip(result['irrep_Tcs'], result['irrep_labels'])):
            if Tc is not None:
                status = " <- DOMINANT" if i == result['dominant_irrep'] else ""
                print(f"  Irrep {i:2d} ({label:15s}): Tc = {Tc:.6f}{status}")
            else:
                print(f"  Irrep {i:2d} ({label:15s}): No solution")
        
        if result['dominant_irrep'] is not None:
            dom_idx = result['dominant_irrep']
            dom_label = result['irrep_labels'][dom_idx]
            print(f"\nDominant channel: Irrep {dom_idx} ({dom_label})")
            print(f"Highest Tc: {result['dominant_Tc']:.6f}")
        else:
            print("\nNo superconducting solution found at this coupling!") """

    """ def detailed_temperature_scan(self, irrep_idx: int, v: float, num_T_points: int = 50):
        
        
        T_scan = np.linspace(self.params.T_min, self.params.T_max, num_T_points)
        eigenvals = []
        
        label = self.irrep_basis.irrep_list[irrep_idx]['label']
        print(f"Temperature scan for Irrep {irrep_idx} ({label}) with v = {v_vector}")
        
        for T in tqdm(T_scan, desc="Temperature scan"):
            max_eig, _, _ = self._get_max_eigenvalue_for_irrep(irrep_idx, T, v_vector)
            eigenvals.append(max_eig)
        
        eigenvals = np.array(eigenvals)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(T_scan, eigenvals, 'o-', linewidth=2, markersize=4)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label='Superconducting instability (λ = 1)')
        
        # 找到Tc
        Tc = self._find_Tc_for_irrep(irrep_idx, v)
        if Tc is not None:
            plt.axvline(x=Tc, color='green', linestyle='--', alpha=0.7,
                       label=f'$T_c$ ≈ {Tc:.4f}')
        
        plt.xlabel('Temperature $T$')
        plt.ylabel('Maximum eigenvalue $\lambda_{max}$')
        plt.title(f'Temperature Scan: Irrep {irrep_idx} ({label}), v = {v:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return T_scan, eigenvals """

def main():
    print("Starting 2D (v0, v1) superconductor phase-diagram analysis...")

    params = Parameters(
        Nk=2,
        Nv0=2, v0_min=0.01, v0_max=0.2,
        Nv1=2, v1_min=0.01, v1_max=0.2
    )
    solver = SuperconductorSolver(params)
    
    print(f"System setup:")
    print(f"  Nk = {params.Nk}")
    print(f"  Nv0 × Nv1 = {params.Nv0} × {params.Nv1}")
    print(f"  Number of irreps = {solver.irrep_basis.num_irreps}")
    
    # 显示所有不可约表示的信息
    print(f"\nIrreducible representations:")
    for i, irrep in enumerate(solver.irrep_basis.irrep_list):
        psi_funcs, d_funcs = solver.irrep_basis.get_basis_functions_for_irrep(i)
        print(f"  Irrep {i}: {irrep['label']:15s} - {len(psi_funcs)} psi funcs, {len(d_funcs)} d funcs")
    
    # 方法1：计算完整相图
    print(f"\n{'='*60}")
    print("METHOD 1: Computing complete phase diagram")
    print(f"{'='*60}")
    
    phase_data = solver.compute_phase_diagram()
    
    # 统计结果
    valid_solutions = [tc for tc in phase_data['dominant_Tcs'] if tc is not None]
    print(f"\nPhase diagram results:")
    print(f"  Points with solutions: {len(valid_solutions)}")
    
    if valid_solutions:
        if np.isfinite(valid_solutions).any():
            print(f"  Tc range: {np.nanmin(valid_solutions):.5f} - {np.nanmax(valid_solutions):.5f}")
        
        # 找到全局最优
        max_Tc_idx = np.nanargmax(phase_data['dominant_Tcs'])
        best_v = params.v_range[max_Tc_idx]
        best_Tc = phase_data['dominant_Tcs'][max_Tc_idx]
        best_irrep = phase_data['dominant_irreps'][max_Tc_idx]
        best_label = phase_data['irrep_labels'][best_irrep]
        
        print(f"\nGlobal optimum:")
        print(f"  Best v: {best_v:.3f}")
        print(f"  Best Tc: {best_Tc:.6f}")
        print(f"  Best irrep: {best_irrep} ({best_label})")
    else:
        print("  No superconducting solutions found!")
    
    # 绘制相图
    solver.plot_phase_diagram(phase_data)
    
    """ # 方法2：分析特定耦合常数点
    print(f"\n{'='*60}")
    print("METHOD 2: Analysis at specific coupling points")
    print(f"{'='*60}")
    
    # 选择几个有代表性的点进行分析
    analysis_points = [0.05, 0.1, 0.15]
    for v_point in analysis_points:
        if v_point <= params.v_max and v_point >= params.v_min:
            solver.analyze_coupling_point(v_point)
    
    # 方法3：温度扫描示例
    print(f"\n{'='*60}")
    print("METHOD 3: Detailed temperature scans")
    print(f"{'='*60}")
    
    # 为最优点做温度扫描
    if valid_solutions:
        print(f"\nTemperature scan for global optimum:")
        solver.detailed_temperature_scan(best_irrep, best_v)
    
    # 为不同不可约表示做温度扫描比较
    compare_v = 0.1
    if compare_v <= params.v_max and compare_v >= params.v_min:
        print(f"\nComparing temperature dependence at v = {compare_v:.3f}")
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, solver.irrep_basis.num_irreps))
        
        for irrep_idx in range(solver.irrep_basis.num_irreps):
            Tc = solver._find_Tc_for_irrep(irrep_idx, compare_v)
            if Tc is not None:
                T_scan, eigenvals = solver.detailed_temperature_scan(irrep_idx, compare_v, num_T_points=30)
                label = solver.irrep_basis.irrep_list[irrep_idx]['label']
                plt.plot(T_scan, eigenvals, 'o-', 
                        label=f'Irrep {irrep_idx}: {label} (Tc={Tc:.4f})', 
                        color=colors[irrep_idx], linewidth=2, markersize=3)
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label='Instability threshold')
        plt.xlabel('Temperature $T$')
        plt.ylabel('Maximum eigenvalue $\lambda_{max}$')
        plt.title(f'Eigenvalue Evolution: All Irreps at v = {compare_v:.3f}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 方法4：通道竞争分析
    print(f"\n{'='*60}")
    print("METHOD 4: Channel competition analysis")
    print(f"{'='*60}")
    
    # 统计每个不可约表示在多少个v点上占主导
    dominance_count = {}
    for i in range(solver.irrep_basis.num_irreps):
        dominance_count[i] = np.sum([dom_irrep == i for dom_irrep in phase_data['dominant_irreps'] if dom_irrep is not None])
    
    print(f"\nChannel dominance statistics:")
    for irrep_idx, count in dominance_count.items():
        if count > 0:
            label = phase_data['irrep_labels'][irrep_idx]
            percentage = 100 * count / len(valid_solutions) if valid_solutions else 0
            print(f"  Irrep {irrep_idx:2d} ({label:15s}): dominant at {count:2d} points ({percentage:5.1f}%)")
    
    # 方法5：相变边界分析
    print(f"\n{'='*60}")
    print("METHOD 5: Phase transition boundary analysis")
    print(f"{'='*60}")
    
    # 找到相变边界（主导通道发生变化的点）
    dominant_irreps = np.array(phase_data['dominant_irreps'])
    phase_boundaries = []
    
    for i in range(1, len(dominant_irreps)):
        if dominant_irreps[i] != dominant_irreps[i-1] and dominant_irreps[i] is not None and dominant_irreps[i-1] is not None:
            boundary_v = (params.v_range[i] + params.v_range[i-1]) / 2
            old_irrep = dominant_irreps[i-1]
            new_irrep = dominant_irreps[i]
            old_label = phase_data['irrep_labels'][old_irrep]
            new_label = phase_data['irrep_labels'][new_irrep]
            
            phase_boundaries.append({
                'v': boundary_v,
                'old_irrep': old_irrep,
                'new_irrep': new_irrep,
                'old_label': old_label,
                'new_label': new_label
            })
    
    if phase_boundaries:
        print(f"\nPhase transition boundaries found:")
        for i, boundary in enumerate(phase_boundaries):
            print(f"  Boundary {i+1}: v ≈ {boundary['v']:.3f}")
            print(f"    From: Irrep {boundary['old_irrep']} ({boundary['old_label']})")
            print(f"    To:   Irrep {boundary['new_irrep']} ({boundary['new_label']})")
    else:
        print(f"\nNo phase transition boundaries found (single dominant channel)")
    
    # 方法6：总结报告
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if valid_solutions:
        print(f"\nSuperconductivity found in {len(valid_solutions)} coupling regions")
        print(f"Temperature range: {min(valid_solutions):.5f} - {max(valid_solutions):.5f}")
        
        # 最常见的主导通道
        most_common_irrep = max(dominance_count.items(), key=lambda x: x[1])
        if most_common_irrep[1] > 0:
            most_common_idx = most_common_irrep[0]
            most_common_label = phase_data['irrep_labels'][most_common_idx]
            print(f"Most dominant channel: Irrep {most_common_idx} ({most_common_label})")
        
        print(f"\nGlobal maximum:")
        print(f"  Optimal coupling: v = {best_v:.3f}")
        print(f"  Maximum Tc: {best_Tc:.6f}")
        print(f"  Optimal channel: Irrep {best_irrep} ({best_label})")
        
        # 推荐进一步研究的参数
        print(f"\nRecommended parameters for detailed study:")
        print(f"  v = {best_v:.3f} (global optimum)")
        
        if phase_boundaries:
            print(f"  Phase boundaries at:")
            for boundary in phase_boundaries:
                print(f"    v ≈ {boundary['v']:.3f}")
    else:
        print(f"\nNo superconducting instabilities found in the explored parameter range!")
        print(f"Consider:")
        print(f"  - Increasing coupling constant range")
        print(f"  - Modifying other physical parameters") 
        print(f"  - Checking calculation convergence") """

if __name__ == "__main__":
    main()