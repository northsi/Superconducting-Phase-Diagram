import numpy as np
from scipy.linalg import eig
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm
from scipy.optimize import brentq
import warnings
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os
from scipy.linalg import eigvals

class FileOutputSettings:


    def __init__(self, prefix="output", tag="run"):
        self.prefix: str = "/share/home/yangxiaolong/zhongluyao/NS2/LGE/newparameter/wholeBZ"     # 基础目录
        self.tag: str = "test1"                      # 名称标签，可选


    def ensure_dir(self):
        os.makedirs(self.prefix, exist_ok=True)

    def sub(self, filename):
        return os.path.join(self.prefix, filename)

    def txt(self):
        return os.path.join(self.prefix, f"{self.tag}.txt")

    def png(self):
        return os.path.join(self.prefix, f"{self.tag}.png")


@dataclass
class Parameters: 

    fermi_window: float = 15*1e-3 # eV
    
    k_max: float = 0.55 #AA-1
    k_min: float = 0.25

    Ef = 1.5914

    a     = 3.4696 #AA 

    # deault

    # k mesh
    Nk: int = 61

    # Temperature range for Tc calculation
    eV2K = 11605
    T_max: float = 100 / eV2K  # KtoeV

    # ---- 2D coupling grids (NEW) ----
    v0_min: float = 0.01 #eV
    v0_max: float = 0.2 #eV
    Nv0: int = 20

    v1_min: float = 0.01 #eV
    v1_max: float = 0.2 #eV
    Nv1: int = 20

    def __post_init__(self):
        self.v0_range = np.linspace(self.v0_min, self.v0_max, self.Nv0)
        self.v1_range = np.linspace(self.v1_min, self.v1_max, self.Nv1)


class PauliMatrices:
    def __init__(self):

        self.sigma_0 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.tau_0 = np.eye(2, dtype=complex)
        self.tau_z = np.array([[1, 0], [0, -1]], dtype=complex)

class IrrepBasisFunctions:
    
    def __init__(self, pauli: PauliMatrices, para = Parameters):
        self.a = para.a
        self.pauli = pauli
        self.irrep_list = self._define_irrep_basis_functions()
        self.num_irreps = len(self.irrep_list)
    
    @lru_cache(maxsize=None)
    def _cached_form_factors(self, kx: float, ky: float):
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]]) * self.a
        k = np.array([kx, ky])
        k_dot_R = k @ R_vectors.T

        omega = np.exp(1j * 2 * np.pi / 3)

        c_k = np.sum(np.cos(k_dot_R)) / np.sqrt(3)
        s_k = np.sum(np.sin(k_dot_R)) / np.sqrt(3)

        omega_weights_plus = np.array([1, omega, omega**2])
        omega_weights_minus = np.array([1, omega**2, omega])

        c_k_plus = omega_weights_plus @ np.cos(k_dot_R) / np.sqrt(3)
        s_k_plus = omega_weights_plus @ np.sin(k_dot_R) / np.sqrt(3)
        c_k_minus = omega_weights_minus @ np.cos(k_dot_R) / np.sqrt(3)
        s_k_minus = omega_weights_minus @ np.sin(k_dot_R) / np.sqrt(3)

        return (
            ('const', 1.0),
            ('c_k', c_k),
            ('s_k', s_k),
            ('c_k_plus', c_k_plus),
            ('c_k_minus', c_k_minus),
            ('s_k_plus', s_k_plus),
            ('s_k_minus', s_k_minus),
        )

    def _get_form_factors(self, k: np.ndarray) -> Dict[str, complex]:
        cache = self._cached_form_factors(float(k[0]), float(k[1]))
        return dict(cache)
    
    def _define_irrep_basis_functions(self) -> List[Dict]:

        p = self.pauli
        
        irrep_list = []
        
        irrep_0 = {
            'label': 'A1g',
            'singlet': [
                
                lambda k, ff: np.kron(p.tau_0, 1.0/np.sqrt(2) * 1j * p.sigma_y),
                
                lambda k, ff: np.kron(p.tau_0, ff['c_k'] * 1j * p.sigma_y),
            ],
            'triplet': [
                    lambda k, ff: np.kron(p.tau_z, ff['s_k'] * p.sigma_z @ (1j * p.sigma_y)) ,

                    lambda k, ff: np.kron(p.tau_z, ((ff['s_k_minus'] * (p.sigma_x + 1j * p.sigma_y) - ff['s_k_plus'] * (p.sigma_x - 1j * p.sigma_y))/2) @ (1j * p.sigma_y)),
                ]
            }
        irrep_list.append(irrep_0)

        irrep_1 = {
            'label': 'A2g', 
            'singlet': [
                #lambda k, ff: 0.0 * 1j * np.kron(p.tau_0,p.sigma_y),
            ],
            'triplet': [
                lambda k, ff: np.kron(p.tau_z, ((ff['s_k_minus'] * (p.sigma_x + 1j * p.sigma_y) + ff['s_k_plus'] * (p.sigma_x - 1j * p.sigma_y))/2) @ (1j * p.sigma_y)),
            ]
        }
        irrep_list.append(irrep_1)
        
        irrep_2 = {
            'label': 'Eg',
            'singlet': [
                lambda k, ff: np.kron(p.tau_0, ff['c_k_minus'] * 1j * p.sigma_y),

                lambda k, ff: np.kron(p.tau_0, ff['c_k_plus'] * 1j * p.sigma_y),
            ],
            'triplet': [
                lambda k, ff: np.kron(p.tau_z, ff['s_k_minus'] * p.sigma_z @ (1j * p.sigma_y)),

                lambda k, ff: np.kron(p.tau_z, ff['s_k_plus'] * p.sigma_z @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_z, np.sqrt(2)/2 * ff['s_k'] * (p.sigma_x + 1j * p.sigma_y) @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_z, -np.sqrt(2)/2 * ff['s_k'] * (p.sigma_x - 1j * p.sigma_y) @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_z, np.sqrt(2)/2 * ff['s_k_minus'] * (p.sigma_x - 1j * p.sigma_y) @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_z, -np.sqrt(2)/2 * ff['s_k_plus'] * (p.sigma_x + 1j * p.sigma_y) @ (1j * p.sigma_y)),
            ]
        }
        irrep_list.append(irrep_2)
        
        irrep_3 = {
            'label': 'A1u',
            'singlet': [
                #lambda k, ff: 0.0 * 1j * np.kron(p.tau_z,p.sigma_y),
            ],
            'triplet': [
                lambda k, ff: np.kron(p.tau_0, ((ff['s_k_minus'] * (p.sigma_x + 1j * p.sigma_y) + ff['s_k_plus'] * (p.sigma_x - 1j * p.sigma_y))/2) @ (1j * p.sigma_y)),
            ]
        }
        irrep_list.append(irrep_3)
        
        irrep_4 = {
            'label': 'A2u',
            'singlet': [
                lambda k, ff: np.kron(p.tau_z, 1.0/np.sqrt(2) * 1j * p.sigma_y),

                lambda k, ff: np.kron(p.tau_z, ff['c_k'] * 1j * p.sigma_y),
            ],
            'triplet': [
                lambda k, ff: np.kron(p.tau_0, ff['s_k'] * p.sigma_z @ (1j * p.sigma_y)),

                lambda k, ff: np.kron(p.tau_0, ((ff['s_k_minus'] * (p.sigma_x + 1j* p.sigma_y) - ff['s_k_plus'] * (p.sigma_x - 1j* p.sigma_y))/2) @ (1j * p.sigma_y)),
            ]
        }
        irrep_list.append(irrep_4)

        irrep_5 = {
            'label': 'Eu',
            'singlet': [
                lambda k, ff: np.kron(p.tau_z, ff['c_k_minus'] * 1j * p.sigma_y),

                lambda k, ff: np.kron(p.tau_z, ff['c_k_plus'] * 1j * p.sigma_y),
            ],
            'triplet': [
                lambda k, ff: np.kron(p.tau_0, ff['s_k_minus'] * p.sigma_z @ (1j * p.sigma_y)),

                lambda k, ff: np.kron(p.tau_0, ff['s_k_plus'] * p.sigma_z @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_0, np.sqrt(2)/2 * ff['s_k'] * (p.sigma_x + 1j * p.sigma_y) @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_0, -np.sqrt(2)/2 * ff['s_k'] * (p.sigma_x - 1j * p.sigma_y) @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_0, np.sqrt(2)/2 * ff['s_k_minus'] * (p.sigma_x - 1j * p.sigma_y) @ (1j * p.sigma_y)),
                
                lambda k, ff: np.kron(p.tau_0, -np.sqrt(2)/2 * ff['s_k_plus'] * (p.sigma_x + 1j * p.sigma_y) @ (1j * p.sigma_y)),
            ]
        }
        irrep_list.append(irrep_5)
        return irrep_list
    
    def get_basis_functions_for_irrep(self, irrep_idx: int) -> Tuple[List, List]:

        if irrep_idx < 0 or irrep_idx >= self.num_irreps:
            raise ValueError(f"Irrep index {irrep_idx} out of range [0, {self.num_irreps-1}]")
        
        irrep_data = self.irrep_list[irrep_idx]
        
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

    def __init__(self, params: Parameters, output: FileOutputSettings):
        self.params = params
        self.pauli = PauliMatrices()
        self.irrep_basis = IrrepBasisFunctions(self.pauli)
        self.k_grid = self._setup_k_grid()
        self._eig_cache = self._precompute_eigensystems()
        self._susceptibility_terms = self._precompute_susceptibility_terms()
        self.output = output
        self.output.ensure_dir()
        
    def _setup_k_grid(self) -> np.ndarray:
            # 1. 获取晶格参数
            # 确保 params 中有 a_tb_ang，如果没有请手动填入数值 (如 a = 3.4696)
            a = self.params.a
            
            # 2. 计算 FBZ 边界参数 (Gamma 到 K 的距离)
            # K 点在 x 轴上，距离为 4pi/(3a)
            K_radius = 4.0 * np.pi / (3.0 * a)
            
            # 3. 设置网格范围
            # 稍微大于 FBZ 即可，不用设太大
            grid_limit = K_radius * 1.1
            
            kx, ky = np.meshgrid(
                np.linspace(-grid_limit, grid_limit, self.params.Nk),
                np.linspace(-grid_limit, grid_limit, self.params.Nk)
            )
            
            # 4. === [Layer 1] 第一布里渊区 (FBZ) 掩膜 ===
            # 几何设定: K 点在 0 度 (x轴), 六边形角在 x 轴上
            # 限制 y 坐标 (上下边界)
            cond_bz_y = np.abs(ky) <= K_radius * np.sqrt(3.0) / 2.0
            # 限制斜边 (|kx| + |ky|/sqrt(3) <= R)
            cond_bz_slope = np.abs(kx) + np.abs(ky) / np.sqrt(3.0) <= K_radius
            
            mask_bz = cond_bz_y & cond_bz_slope

            # 5. === [Layer 2] K 模长限制 (Magitude Constraints) ===
            # 计算每个点的模长
            k_magnitude = np.sqrt(kx**2 + ky**2)
            
            # 条件 A: 0.3 <= |K| <= 0.5
            cond_mag_A = (k_magnitude >= 0.3) & (k_magnitude <= 0.55)
            
            # 条件 B: |K| > 0.8
            cond_mag_B = (k_magnitude > 0.85)
            
            # 组合模长条件 (A OR B)
            mask_magnitude = cond_mag_A | cond_mag_B

            # 6. === 合并所有限制条件 ===
            # 点必须同时在 FBZ 内，且满足模长要求
            final_mask = mask_bz & mask_magnitude

            # 7. 应用掩膜提取点
            kx_filtered = kx[final_mask]
            ky_filtered = ky[final_mask]
            k_points_full = np.column_stack([kx_filtered, ky_filtered])
            
            # 8. 能量窗口筛选 (保持原有逻辑)
            k_points_selected = []
            
            # 简单的进度打印
            print(f"Grid setup: BZ & Mag filter kept {len(k_points_full)} points. Checking energy window...")

            for k in k_points_full:
                H_k = self._get_hamiltonian(k)
                eigs = np.linalg.eigvalsh(H_k)
                if np.any(np.abs(eigs - self.params.Ef) <= self.params.fermi_window):
                    k_points_selected.append(k)
    
            k_points_selected = np.array(k_points_selected)

            print('Final selected number =', len(k_points_selected))
            
            if len(k_points_selected) == 0:
                warnings.warn("No k-points found within the Fermi window!")
            
            return k_points_selected
    
    def _precompute_eigensystems(self) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        eig_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for k in self.k_grid:
            key = tuple(np.asarray(k, dtype=float))
            H_k = self._get_hamiltonian(k)
            H_mk = self._get_hamiltonian_h(k)
            eigenvals_e, evec_e = self._eigvals_and_eigvec(H_k)
            eigenvals_h, evec_h = self._eigvals_and_eigvec(H_mk)
            eig_cache[key] = (eigenvals_e, eigenvals_h, evec_e, evec_h)
        return eig_cache
    
    def _precompute_susceptibility_terms(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, any]] = []
        norm_factor = 1.0 / len(self.k_grid) if len(self.k_grid) else 1.0
        for irrep_idx in range(self.irrep_basis.num_irreps):
            psi_funcs, d_funcs = self.irrep_basis.get_basis_functions_for_irrep(irrep_idx)
            num_psi = len(psi_funcs)
            num_d = len(d_funcs)
            I_terms: List[Tuple[np.ndarray, float, float]] = []
            J_terms: List[Tuple[np.ndarray, float, float]] = []
            K_terms: List[Tuple[np.ndarray, float, float]] = []

            for k in self.k_grid:
                eigenvals_e, eigenvals_h, vec_e, vec_h = self._get_eig(k)
                psi_matrices = [func(k) for func in psi_funcs] if num_psi > 0 else []
                d_matrices = [func(k) for func in d_funcs] if num_d > 0 else []

                for j, Ej in enumerate(eigenvals_e):
                    Pj = vec_e[:, j]
                    for l, El in enumerate(eigenvals_h):
                        Pl = vec_h[:, l]
                        Ej_real = float(np.real(Ej))
                        El_real = float(np.real(El))

                        psi_term1 = psi_term2 = None
                        if num_psi > 0:
                            psi_term1 = np.array(
                                [Pl.conj().T @ psi_matrices[r1].conj().T @ Pj for r1 in range(num_psi)],
                                dtype=complex
                            )
                            psi_term2 = np.array(
                                [Pj.conj().T @ psi_matrices[r2] @ Pl for r2 in range(num_psi)],
                                dtype=complex
                            )
                            M_I = np.outer(psi_term1, psi_term2)
                            if not np.allclose(M_I, 0.0):
                                I_terms.append((M_I, Ej_real, El_real))

                        d_term1 = d_term2 = None
                        if num_d > 0:
                            d_term1 = np.array(
                                [Pl.conj().T @ d_matrices[r1].conj().T @ Pj for r1 in range(num_d)],
                                dtype=complex
                            )
                            d_term2 = np.array(
                                [Pj.conj().T @ d_matrices[r2] @ Pl for r2 in range(num_d)],
                                dtype=complex
                            )
                            M_J = np.outer(d_term1, d_term2)
                            if not np.allclose(M_J, 0.0):
                                J_terms.append((M_J, Ej_real, El_real))

                        if num_psi > 0 and num_d > 0:
                            # reuse psi_term1 from above and d_term2 from triplet calculation
                            if psi_term1 is None:
                                psi_term1 = np.array(
                                    [Pl.conj().T @ psi_matrices[r1].conj().T @ Pj for r1 in range(num_psi)],
                                    dtype=complex
                                )
                            if d_term2 is None:
                                d_term2 = np.array(
                                    [Pj.conj().T @ d_matrices[r2] @ Pl for r2 in range(num_d)],
                                    dtype=complex
                                )
                            M_K = np.outer(psi_term1, d_term2)
                            if not np.allclose(M_K, 0.0):
                                K_terms.append((M_K, Ej_real, El_real))

            data.append({
                'num_psi': num_psi,
                'num_d': num_d,
                'I_terms': I_terms,
                'J_terms': J_terms,
                'K_terms': K_terms,
                'norm': norm_factor
            })
        return data
    
    def _get_hamiltonian(self, k: np.ndarray) -> Tuple[np.ndarray]:
        p = self.params
        sigma = self.pauli

        tau0, tauz = sigma.tau_0, sigma.tau_z
        sig0, sigx, sigy, sigz = sigma.sigma_0, sigma.sigma_x, sigma.sigma_y, sigma.sigma_z

        a     = self.params.a
        mu         = 1.599376323162
        t1         = 0.031271584234
        t2         = 0.099523142476
        t3         = 0.002809001577
        t4         = -0.009937945837
        t5         = -0.011107096082
        lam1       = 0.015183347380
        lam2       = 0.001506932611
        t00        = 0.042370553809
        t01        = 0.009335596478
        t02        = 0.006024481632
        t03        = 0.008890930636

        kx, ky = float(k[0]), float(k[1])

        # angles
        alpha = 0.5 * a * kx
        beta  = 0.5 * np.sqrt(3.0) * a * ky

        xi = (
            mu
            + 2*t1*(np.cos(2*alpha) + 2*np.cos(alpha)*np.cos(beta))
            + 2*t2*(np.cos(2*beta)  + 2*np.cos(3*alpha)*np.cos(beta))
            + 2*t3*(np.cos(4*alpha) + 2*np.cos(2*alpha)*np.cos(2*beta))
            + 4*t4*(np.cos(alpha)*np.cos(3*beta)
                    + np.cos(4*alpha)*np.cos(2*beta)
                    + np.cos(5*alpha)*np.cos(beta))
            + 2*t5*(np.cos(6*alpha) + 2*np.cos(3*alpha)*np.cos(3*beta))
        )

        Lambda = (
            2*lam1*(np.sin(2*alpha) - 2*np.sin(alpha)*np.cos(beta))
            + 2*lam2*(np.sin(4*alpha) - 2*np.sin(2*alpha)*np.cos(2*beta))
        )

        H_10 = (
            xi * sig0
            + Lambda * sigz
        ).astype(complex)


        H_20 = (
            xi * sig0
            - Lambda * sigz 
        ).astype(complex)

        T = (
            t00+2*t01*(2*np.cos(alpha)*np.cos(beta)+np.cos(2*alpha))+2*t02*(np.cos(2*beta) + 2*np.cos(3*alpha)*np.cos(beta)) + 2*t03*(np.cos(4*alpha) + 2*np.cos(2*alpha)*np.cos(2*beta))
        ) * sig0

        H_k = np.block([[H_10, T],
                    [T.conj().T,  H_20]]).astype(complex)

        return H_k

    def _get_hamiltonian_h(self, k: np.ndarray) -> Tuple[np.ndarray]:

        H_mk = self._get_hamiltonian(-k)
        H_mk = -H_mk.conj()

        return H_mk

    
    @staticmethod
    def _eigvals_and_eigvec(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        evals, evecs = np.linalg.eigh(H)

        return evals, evecs

    @staticmethod
    def _matsubara_sum(Ej: float, El: float, T: float, tol: float = 1e-12) -> float:
        # Here T is the "temperature" in whatever units you're using
        B = 1.0 / T  # inverse temperature

        if abs(Ej - El) < tol:
            # Use the limiting expression S(E,E) = (B/4) * sech^2(B*E/2)
            Ej_half = B * Ej * 0.5
            return 0.25 * B / np.cosh(Ej_half)**2
        else:
            Ej_half = B * Ej * 0.5
            El_half = B * El * 0.5
            return -(np.tanh(El_half) - np.tanh(Ej_half)) / (2.0 * (Ej - El))

    def _calculate_susceptibility_for_irrep(self, irrep_idx: int, T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
       
        terms = self._susceptibility_terms[irrep_idx]
        num_psi = terms['num_psi']
        num_d = terms['num_d']
        norm = terms['norm']

        I = np.zeros((num_psi, num_psi), dtype=complex)
        J = np.zeros((num_d, num_d), dtype=complex)
        K = np.zeros((num_psi, num_d), dtype=complex)

        if num_psi > 0:
            for M, Ej, El in terms['I_terms']:
                S_jl = self._matsubara_sum(Ej, El, T)
                I += M * S_jl

        if num_d > 0:
            for M, Ej, El in terms['J_terms']:
                S_jl = self._matsubara_sum(Ej, El, T)
                J += M * S_jl

        if num_psi > 0 and num_d > 0:
            for M, Ej, El in terms['K_terms']:
                S_jl = self._matsubara_sum(Ej, El, T)
                K += M * S_jl

        return norm * I, norm * J, norm * K

    def _get_eig(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        key = tuple(np.asarray(k, dtype=float))
        return self._eig_cache[key]
    
    def _construct_V_matrix(self, irrep_idx: int, v_vector: np.ndarray) -> np.ndarray:
        
        if irrep_idx == 0 or irrep_idx == 4:

            v_vec_1=[v_vector[0],v_vector[1],v_vector[1],v_vector[1]]
            V = np.diag(v_vec_1)

        elif irrep_idx == 1 or irrep_idx == 3:
            v_vec_0=[v_vector[1]]
            V = np.diag(v_vec_0)

        elif irrep_idx == 2 or irrep_idx == 5:
            V = np.kron(np.eye(8), v_vector[1])
        
        return V

    def _solve_irrep_eigensystem(self, irrep_idx: int, T: float, v_vector: np.ndarray,
                                 return_vectors: bool = False, degeneracy_tol: float = 1e-8) -> Tuple[float, List[np.ndarray]]:
        """
        Compute the leading eigenvalue for a given irrep and, optionally, return all eigenvectors
        associated with that eigenvalue (useful when it is degenerate).
        """
        V = self._construct_V_matrix(irrep_idx, v_vector)

        I, J, K = self._calculate_susceptibility_for_irrep(irrep_idx, T)

        M = np.block([[I, K], [K.conj().T, J]])

        VM = V @ M

        if return_vectors:
            eigenvals, eigenvectors = eig(VM)
        else:
            eigenvals = eigvals(VM)
            eigenvectors = None
        
        real_eigenvals = np.real(eigenvals)
        
        imag_parts = np.abs(np.imag(eigenvals))
        if np.any(imag_parts > 1e-8):
            warnings.warn(f"Large imaginary parts in eigenvalues: max = {np.max(imag_parts)}")
        
        max_eigenval = np.max(real_eigenvals)

        max_eigenvectors: List[np.ndarray] = []
        if return_vectors and eigenvectors is not None:
            is_max = np.abs(real_eigenvals - max_eigenval) < degeneracy_tol
            max_indices = np.where(is_max)[0]
            
            for idx in max_indices:
                vec = eigenvectors[:, idx]
                norm = np.linalg.norm(vec)
                if norm != 0:
                    vec = vec / norm
                max_eigenvectors.append(vec)
        
        return max_eigenval, max_eigenvectors
        
    def _find_Tc_for_irrep(self, irrep_idx: int, v_vector: np.ndarray,
                        coarse_N=20, xtol=1e-5, rtol=1e-3, maxiter=300):
        
        T_min = 1e-6
        T_initial_max = float(self.params.T_max)
        T_search_max = T_initial_max
        f_cache: Dict[float, float] = {}

        def f(T):
            T_val = float(T)
            if T_val not in f_cache:
                eigval, _ = self._solve_irrep_eigensystem(irrep_idx, T_val, v_vector, return_vectors=False)
                f_cache[T_val] = float(eigval - 1.0)
            return f_cache[T_val]

        while f(T_search_max) > 0.0:
            T_search_max *= 5.0
            
            # 防止 T_max 无限增长的判断
            if T_search_max > 1000.0 * T_initial_max:
                print(f"Error: T_search_max exceeded 1000 times the initial T_max ({T_initial_max:.4e} eV). Assuming no Tc found in a reasonable range.")
                return None
        
        Ts = np.geomspace(T_min, T_search_max, int(coarse_N))
        Fs = np.array([f(T) for T in Ts])

        for a, b, fa, fb in zip(Ts[:-1], Ts[1:], Fs[:-1], Fs[1:]):
            if fa == 0.0: return float(a)
            if fa * fb < 0.0: 
                return float(brentq(f, float(a), float(b), xtol=xtol, rtol=rtol, maxiter=maxiter))

        return None

    def find_dominant_channel_at_coupling(self, v_vector: np.ndarray) -> Dict:

        results = {
            'v': np.array(v_vector, dtype=float).copy(),
            'irrep_Tcs': [],
            'irrep_labels': [],
            'dominant_irrep': None,
            'dominant_Tc': None
        }
        
        max_Tc = -1.0
        dominant_irrep = None
        
        print(f"Calculating Tc at v={v_vector}")
                
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
            results['dominant_Tc'] = max_Tc * self.params.eV2K
            _, eigvecs = self._solve_irrep_eigensystem(dominant_irrep, max_Tc, v_vector, return_vectors=True)
            results['dominant_eigenvecs'] = np.real(eigvecs)
        
        return results

    def compute_phase_diagram(self) -> Dict:

        p = self.params
        Nv0, Nv1 = len(p.v0_range), len(p.v1_range)

        nir = self.irrep_basis.num_irreps

        phase_data = {
            "v0_range": p.v0_range,
            "v1_range": p.v1_range,
            "dominant_irreps": np.full((Nv0, Nv1), -1, int),
            "dominant_Tcs": np.full((Nv0, Nv1), np.nan),
            "dominant_channel": np.empty((Nv0, Nv1), dtype=object),
            "all_irrep_Tcs": np.full((Nv0, Nv1, nir), np.nan),
            "irrep_labels": [irrep["label"] for irrep in self.irrep_basis.irrep_list]
        }

        print(f"Computing 2D phase diagram on {Nv0}×{Nv1} grid…")

        # ========== 打开 3 个输出文件 ==========
        dom_file = self.output.sub("dominant_summary.txt")
        tc_file = self.output.sub("Tc_all_irreps.txt")
        chan_file = self.output.sub("channels_summary.txt")

        f_dom = open(dom_file, "w")
        f_tc = open(tc_file, "w")
        f_chan = open(chan_file, "w")

        f_dom.write("# v0   v1   dominant_irrep   Tc[K]   degenerate_irreps   deg_count(dom)\n")

        ir_labels = [ir["label"] for ir in self.irrep_basis.irrep_list]
        header_tc = " ".join([f"{lab:8s}" for lab in ir_labels])
        f_tc.write("# v0   v1   " + header_tc + "\n")

        f_chan.write("# v0  v1  irrep  channel_idx  is_dominant  Tc[K]  deg_count(irrep)  real[...]  imag[...]\n")

        # ========== 主循环 ==========
        for i, v0 in enumerate(p.v0_range):
            for j, v1 in enumerate(p.v1_range):

                vvec = np.array([v0, v1], float)
                res = self.find_dominant_channel_at_coupling(vvec)

                TcK_dom = res["dominant_Tc"]
                if TcK_dom is None:
                    continue

                dom_idx = res["dominant_irrep"]
                dom_label = self.irrep_basis.irrep_list[dom_idx]["label"]

                # ---- 所有 Tc(K) ----
                Tc_list_K = [
                    tc * p.eV2K if tc is not None else None
                    for tc in res["irrep_Tcs"]
                ]

                # ---- 简并检测 ----
                degenerate_irreps = []
                for ir_idx, TcK in enumerate(Tc_list_K):
                    if TcK is None:
                        continue
                    if abs(TcK - TcK_dom) < 1e-6:
                        degenerate_irreps.append(self.irrep_basis.irrep_list[ir_idx]["label"])
                degenerate_str = ", ".join(degenerate_irreps)

                # ========== 写入 dominant_summary ==========
                dom_deg = len(res.get("dominant_eigenvecs", [])) if res.get("dominant_eigenvecs") is not None else 0
                f_dom.write(
                    f"{v0:.6f}  {v1:.6f}  {dom_label:8s}  {TcK_dom:.6f}  {degenerate_str}  {dom_deg:d}\n"
                )

                # ========== 写入 Tc_all_irreps ==========
                row_tc = f"{v0:.6f}  {v1:.6f}  "
                for tc in Tc_list_K:
                    if tc is None:
                        row_tc += "None     "
                    else:
                        row_tc += f"{tc:.6f} "
                f_tc.write(row_tc + "\n")

                # ========== 写入 channels_summary.txt ==========
                for ir_idx, Tc in enumerate(res["irrep_Tcs"]):
                    if Tc is None:
                        continue

                    TcK = Tc * p.eV2K
                    label = ir_labels[ir_idx]

                    _, eigvecs = self._solve_irrep_eigensystem(ir_idx, Tc, vvec, return_vectors=True)
                    deg_count = len(eigvecs)
                    for ch_idx, vec in enumerate(eigvecs):
                        is_dom = 1 if ir_idx == dom_idx else 0
                        real_s = " ".join(f"{x.real:.3f}" for x in vec)
                        f_chan.write(
                            f"{v0:.6f}  {v1:.6f}  {label:6s}  {ch_idx:2d}  {is_dom}  {TcK:.6f}  {deg_count:d}  "
                            f"[{real_s}]\n"
                        )

                # ========== 写回 phase_data ==========
                phase_data["dominant_irreps"][i, j] = dom_idx
                phase_data["dominant_Tcs"][i, j] = TcK_dom
                phase_data["dominant_channel"][i, j] = res["dominant_eigenvecs"]

                for ir_idx, tc in enumerate(res["irrep_Tcs"]):
                    if tc is not None:
                        phase_data["all_irrep_Tcs"][i, j, ir_idx] = tc * p.eV2K

        # 关闭文件
        f_dom.close()
        f_tc.close()
        f_chan.close()

        print("Saved:")
        print("  dominant_summary.txt")
        print("  Tc_all_irreps.txt")
        print("  channels_summary.txt")

        return phase_data


    def plot_phase_diagram_2d(self, phase_data: Dict):

        v0 = phase_data['v0_range']; v1 = phase_data['v1_range']
        V0, V1 = np.meshgrid(v0, v1, indexing='ij')

        Tc = phase_data['dominant_Tcs']
        dom = phase_data['dominant_irreps']

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        im0 = axes[0].pcolormesh(V0, V1, Tc, shading='nearest')
        fig.colorbar(im0, ax=axes[0], label=r'Highest $T_c$')
        axes[0].set_xlabel(r'$v_0$ (on-site)')
        axes[0].set_ylabel(r'$v_1$ (nearest-neighbor)')
        axes[0].set_title('Max $T_c$ over irreps')

        axes[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
        axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))


        dom_plot = dom.copy().astype(float)
        dom_plot[dom_plot < 0] = np.nan
        im1 = axes[1].pcolormesh(V0, V1, dom_plot, shading='nearest', cmap='tab10',
                                 vmin=-0.5, vmax=self.irrep_basis.num_irreps - 0.5)
        cbar = fig.colorbar(im1, ax=axes[1])
        cbar.set_label('Dominant irrep index')
        axes[1].set_xlabel(r'$v_0$ (on-site)')
        axes[1].set_ylabel(r'$v_1$ (nearest-neighbor)')
        axes[1].set_title('Dominant superconducting channel (irrep)')
        axes[1].xaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
        axes[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))        
        plt.tight_layout()

        output_png = self.output.png()
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"Phase diagram saved to: {output_png}")


def main():
    print("Starting 2D (v0, v1) superconductor phase-diagram analysis...")
    output = FileOutputSettings(prefix="mpi_output", tag="mpi_run")
    output.ensure_dir()

    params = Parameters(
        Nk=700,
        Nv0=5, v0_min=-0.005, v0_max=0.005  ,
        Nv1=5, v1_min=0.000, v1_max=0.005
    )
    solver = SuperconductorSolver(params, output)
    
    print(f"System setup:")
    print(f"  Nk = {params.Nk}")
    print(f"  Nv0 x Nv1 = {params.Nv0} x {params.Nv1}")
    print(f"  Number of irreps = {solver.irrep_basis.num_irreps}")
    

    print(f"\nIrreducible representations:")
    for i, irrep in enumerate(solver.irrep_basis.irrep_list):
        psi_funcs, d_funcs = solver.irrep_basis.get_basis_functions_for_irrep(i)
        print(f"  Irrep {i}: {irrep['label']:15s} - {len(psi_funcs)} psi funcs, {len(d_funcs)} d funcs")
    
    phase_data = solver.compute_phase_diagram()
    

    valid_solutions = [tc for tc in np.asarray(phase_data['dominant_Tcs'], float).ravel() if np.isfinite(tc)]
    print(f"\nPhase diagram results:")
    print(f"  Points with solutions: {len(valid_solutions)}")
    
    if valid_solutions:
        if np.isfinite(valid_solutions).any():
            print(f"  Tc range: {np.nanmin(valid_solutions):.5f} - {np.nanmax(valid_solutions):.5f}")
        

        max_Tc_idx = np.nanargmax(phase_data['dominant_Tcs'])
        ii, jj = np.unravel_index(
        np.nanargmax(phase_data['dominant_Tcs']),
        phase_data['dominant_Tcs'].shape
        )
        best_v0 = params.v0_range[ii]
        best_v1 = params.v1_range[jj]
        best_Tc = phase_data['dominant_Tcs'][ii, jj]
        best_irrep = phase_data['dominant_irreps'][ii, jj]
        
        print(f"\nGlobal optimum:")
        print(f"  Best v: {best_v0:.3f} {best_v1:.3f}")
        print(f"  Best Tc: {best_Tc:.6f}")
        print(f"  Best irrep: {best_irrep}")
    else:
        print("  No superconducting solutions found!")
    

    solver.plot_phase_diagram_2d(phase_data)
    

if __name__ == "__main__":
    main()
