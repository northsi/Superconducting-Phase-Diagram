import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import eig
from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional
from tqdm import tqdm
import warnings

@dataclass
class Parameters:
    hbar: float = 1.0
    m_star: float = 0.5          # m*=0.5 me
    mu: float = 0.0              
    beta_so_meV: float = 2.0     # β_so = 2 meV
    m_alphaR2_over2_meV: float = 8.0  # m* α_R^2 / 2 = 8 meV

    # k mesh
    Nk: int = 61
    k_max: float = 0.5          

    def __post_init__(self):
        # to eV 
        self.beta_so: float = 1e-3 * self.beta_so_meV  # 0.002 eV
        # α_R = sqrt[ 2*(mα_R^2/2) / m* ]
        self.alpha_R: float = float(
            np.sqrt(2.0 * 1e-3 * self.m_alphaR2_over2_meV / self.m_star)
        )
        # ~ 0.178885 eV·Å
        
    num_channels: int = 4
    v_min: float = 0.01         # Minimum coupling constant value
    v_max: float = 0.2          # Maximum coupling constant value
    v_points: int = 20
    T_min: float = 0.001
    T_max: float = 0.1
    
    def __post_init__(self):
        # to eV 
        self.beta_so: float = 1e-3 * self.beta_so_meV  # 0.002 eV
        # α_R = sqrt[ 2*(mα_R^2/2) / m* ]
        self.alpha_R: float = float(
            np.sqrt(2.0 * 1e-3 * self.m_alphaR2_over2_meV / self.m_star)
        )
        # ~ 0.178885 eV·Å
        self.v_range = np.linspace(self.v_min, self.v_max, self.v_points)

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

class BasisFunctions:
    
    def __init__(self, pauli: PauliMatrices):
        self.pauli = pauli
        self._setup_basis_functions()
    
    def _setup_basis_functions(self):
        p = self.pauli
        
        self.singlet_funcs = [
            # A1g 
            lambda k: np.kron(p.tau_0, 1.0 * (1j * p.sigma_y)),                    # ψ = 1
            lambda k: np.kron(p.tau_0, self._get_ek(k) * (1j * p.sigma_y)),          # ψ = e_k
            # A2g
            lambda k: np.kron(p.tau_0, 0.0 * (1j * p.sigma_y)),
    
            #Eg
            lambda k: np.kron(p.tau_0, self._get_ek_plus(k) * (1j * p.sigma_y)),
            lambda k: np.kron(p.tau_0, self._get_ek_minus(k) * (1j * p.sigma_y)),
    
            # A1u
            lambda k: np.kron(p.tau_z, 0.0 * (1j * p.sigma_y)),
    
            # A2u 
            lambda k: np.kron(p.tau_z, 1.0 * (1j * p.sigma_y)),                    # ψ = σ_z
            lambda k: np.kron(p.tau_z, self._get_ek(k) * (1j * p.sigma_y)),          # ψ = σ_z * e_k,
    
            #Eg
            lambda k: np.kron(p.tau_z, self._get_ek_plus(k) * (1j * p.sigma_y)),
            lambda k: np.kron(p.tau_z, self._get_ek_minus(k) * (1j * p.sigma_y)),
        ]
        
        self.triplet_funcs = [
            # A1g 表示 (偶宇称)
            lambda k: np.kron(p.tau_z, self._get_ok_z(k) * p.sigma_z @ (1j * p.sigma_y)),    # d = o_k ẑ

            # A2g 表示 (偶宇称)
            lambda k: np.kron(p.tau_0, self._get_ok_z(k) * p.sigma_z @ (1j * p.sigma_y)),    # d = o_k^- ẑ
            
            # A1u 表示 (奇宇称)
            lambda k: np.kron(p.tau_z, self._get_ok_xpm(k) * (1j * p.sigma_y)),              # d = σ_z * o_k x̂_±
            lambda k: np.kron(p.tau_z, self._get_ok_z(k) * p.sigma_z @ (1j * p.sigma_y))     # d = σ_z * o_k ẑ
        ]
    
    def _get_ek(self, k: np.ndarray) -> complex:
        """e_k = Σ_n cos(k·T_n)"""
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        return np.sum(np.cos(k_dot_R))

    def _get_ek_plus(self, k: np.ndarray) -> complex:
        omega = np.exp(1j * 2 * np.pi / 3)
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        omega_weights = np.array([1, omega, omega**2])
        return np.sum(omega_weights @ np.cos(k_dot_R))

    def _get_ek_minus(self, k: np.ndarray) -> complex:
        omega = np.exp(1j * 2 * np.pi / 3)
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        omega_weights = np.array([1, omega**2, omega])
        return np.sum(omega_weights @ np.sin(k_dot_R))
    
    def _get_ok(self, k: np.ndarray) -> complex:
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        return np.sum(np.sin(k_dot_R))
    
    def _get_ok_plus(self, k: np.ndarray) -> complex:
        omega = np.exp(1j * 2 * np.pi / 3)
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        omega_weights = np.array([1, omega, omega**2])
        return np.sum(omega_weights @ np.sin(k_dot_R))

    def _get_ok_minus(self, k: np.ndarray) -> complex:
        omega = np.exp(1j * 2 * np.pi / 3)
        R_vectors = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])
        k_dot_R = k @ R_vectors.T
        omega_weights = np.array([1, omega**2, omega])
        return np.sum(omega_weights @ np.sin(k_dot_R))
    
    def evaluate_singlet(self, k: np.ndarray) -> List[np.ndarray]:
        return [func(k) for func in self.singlet_funcs]
    
    def evaluate_triplet(self, k: np.ndarray) -> List[np.ndarray]:
        return [func(k) for func in self.triplet_funcs]

class SuperconductorSolver:
    
    def __init__(self, params: Parameters):
        self.params = params
        self.pauli = PauliMatrices()
        self.basis = BasisFunctions(self.pauli)
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
        H_I = p.beta_so * σ.sigma_z      # ε=+1

        H_k  = H0 + H_R + H_I
        H_mk = H0 - H_R + H_I    # k -> -k
        H_mk = H_mk.conj()            

        return H_k, H_mk
    
    @staticmethod
    def _eigvals_and_projectors_grouped(
        H: np.ndarray,
        tol_group: float = 1e-8,
        herm_tol: float = 1e-10,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
       
        is_herm = np.allclose(H, H.conj().T, atol=herm_tol)
        if is_herm:
            evals, evecs = np.linalg.eigh(H)
            print("Hermitian")
        else:
           
            evals, evecs = np.linalg.eig(H)
            print("Non-Hermitian")

        order = np.lexsort((evals.imag, evals.real))
        evals = evals[order]
        evecs = evecs[:, order]

        grouped_vals: List[complex] = []
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
            I = np.eye(P.shape[0], dtype=P.dtype)
            if not np.allclose(P, I, atol=1e-12, rtol=0):
                print(f"Warning: Eigenvectors in group {len(grouped_vals)} not orthonormal, applying QR.")
                V, _ = np.linalg.qr(V)  
                P = V @ V.conj().T

            projectors.append(P)
            grouped_vals.append(np.mean(evals[i:j])) 
            i = j

        return np.asarray(grouped_vals), projectors

    def _get_spectral_data(
        self,
        k: np.ndarray,
        tol_group: float = 1e-8,
        herm_tol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        H_k, H_mk = self._get_hamiltonian(k)

        eigenvals_e, proj_e = self._eigvals_and_projectors_grouped(
            H_k, tol_group=tol_group, herm_tol=herm_tol
        )
        eigenvals_h, proj_h = self._eigvals_and_projectors_grouped(
            H_mk, tol_group=tol_group, herm_tol=herm_tol
        )
        return eigenvals_e, eigenvals_h, proj_e, proj_h

    @staticmethod
    def _matsubara_sum(Ej: float, El: float, T: float, tol: float = 1e-10) -> float:
        beta = 1.0 / T
        # 辅助：sech^2
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

        # 一般情形
        return (np.tanh(beta * Ej / 2.0) + np.tanh(beta * El / 2.0)) / (2.0 * (Ej + El))

    def _calculate_susceptibility_matrices(self, T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_s = len(self.basis.singlet_funcs)
        num_t = len(self.basis.triplet_funcs)

        I = np.zeros((num_s, num_s), dtype=complex)
        J = np.zeros((num_t, num_t), dtype=complex)
        K = np.zeros((num_s, num_t), dtype=complex)

        for k in self.k_grid:
            eigenvals_e, eigenvals_h, proj_e, proj_h = self._get_spectral_data(k)
            num_groups_e = len(eigenvals_e)
            num_groups_h = len(eigenvals_h)

            psi_s_k = self.basis.evaluate_singlet(k)  # List/ndarray of matrices
            d_t_k   = self.basis.evaluate_triplet(k)

            for j in range(num_groups_e):
                Ej = float(np.real(eigenvals_e[j]))
                Pj = proj_e[j]

                for l in range(num_groups_h):
                    El = float(np.real(eigenvals_h[l]))
                    Pl = proj_h[l]

                    S_jl = self._matsubara_sum(Ej, El, T)
                    if abs(S_jl) < 1e-12:
                        continue

                    # I: singlet-singlet
                    for r1 in range(num_s):
                        A = psi_s_k[r1].conj().T @ Pj
                        for r2 in range(num_s):
                            M_I = np.trace(A @ psi_s_k[r2] @ Pl)
                            I[r1, r2] += M_I * S_jl

                    # J: triplet-triplet
                    for r1 in range(num_t):
                        A = d_t_k[r1].conj().T @ Pj
                        for r2 in range(num_t):
                            M_J = np.trace(A @ d_t_k[r2] @ Pl)
                            J[r1, r2] += M_J * S_jl

                    # K: singlet-triplet
                    for r1 in range(num_s):
                        A = psi_s_k[r1].conj().T @ Pj
                        for r2 in range(num_t):
                            M_K = np.trace(A @ d_t_k[r2] @ Pl)
                            K[r1, r2] += M_K * S_jl

        factor = -1.0 / len(self.k_grid)
        return factor * I, factor * J, factor * K
    
    def _construct_V_matrix(self, v_vector: np.ndarray) -> np.ndarray:
       
        num_s = len(self.basis.singlet_funcs)
        num_t = len(self.basis.triplet_funcs)
        
        # Create diagonal matrix with coupling constants
        v_diag = np.diag(v_vector)
        
        # Create I_{2×2} ⊗ diag{v_r} structure
        I_2x2 = np.eye(2)
        V = np.kron(I_2x2, v_diag)
        
        return V
    
    def _get_max_eigenvalue_and_vectors(self, T: float, v_vector: np.ndarray, 
                                       degeneracy_tol: float = 1e-10) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        
        I, J, K = self._calculate_susceptibility_matrices(T)
        
        # Construct susceptibility matrix
        M = np.block([[I, K], [K.conj().T, J]])
        
        # Construct V matrix from vector coupling constants
        V = self._construct_V_matrix(v_vector)
        
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

    def _get_max_eigenvalue(self, T: float, v_vector: np.ndarray) -> float:
        max_eigenval, _, _ = self._get_max_eigenvalue_and_vectors(T, v_vector)
        return max_eigenval
    
    def _analyze_eigenvector(self, eigenvector: np.ndarray) -> dict:
        num_s = len(self.basis.singlet_funcs)
        num_t = len(self.basis.triplet_funcs)
        
        # Extract singlet and triplet components
        singlet_part = eigenvector[:num_s]
        triplet_part = eigenvector[num_s:num_s+num_t]
        
        analysis = {
            'singlet_components': singlet_part,
            'triplet_components': triplet_part,
        }
        
        return analysis

    def _find_Tc_with_eigenvectors(self, v_vector: np.ndarray, 
                                  return_eigenvectors: bool = True) -> Optional[dict]:
        
        # λ_max(T) - 1 = 0
        def objective(T):
            return self._get_max_eigenvalue(T, v_vector) - 1.0
        
        try:
            f_low = objective(self.params.T_min)
            f_high = objective(self.params.T_max)
            
            if f_low < 0 or f_high > 0:
                return None  
            
            Tc = brentq(objective, self.params.T_min, self.params.T_max, xtol=1e-6)
            
            if not return_eigenvectors:
                return Tc
                
            max_eigenval, all_eigenvals, max_eigenvectors = self._get_max_eigenvalue_and_vectors(Tc, v_vector)
            
            eigenvector_analyses = []
            for i, vec in enumerate(max_eigenvectors):
                analysis = self._analyze_eigenvector(vec)
                analysis['degeneracy_index'] = i
                eigenvector_analyses.append(analysis)
            
            result = {
                'Tc': Tc,
                'max_eigenvalue': max_eigenval,
                'degeneracy': len(max_eigenvectors),
                'all_eigenvalues': all_eigenvals,
                'eigenvectors': max_eigenvectors,
                'eigenvector_analyses': eigenvector_analyses,
                'v_vector': v_vector
            }
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error in Tc calculation: {e}")
            return None
    
    def _find_Tc_optimized(self, v_vector: np.ndarray) -> Optional[float]:
        result = self._find_Tc_with_eigenvectors(v_vector, return_eigenvectors=False)
        return result
    
    def compute_phase_diagram_2d_slice(self, fixed_channels: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if fixed_channels is None:
            # By default, vary first two channels (v_0, v_1) and set others to minimum
            varying_channels = [0, 1]
            fixed_channels = {i: self.params.v_min for i in range(2, self.params.num_channels)}
        else:
            # Find which channels are not fixed (these will be varied)
            all_channels = set(range(self.params.num_channels))
            fixed_set = set(fixed_channels.keys())
            varying_channels = list(all_channels - fixed_set)
            if len(varying_channels) != 2:
                raise ValueError("Exactly 2 channels must be varied for 2D phase diagram")
        
        v_range = self.params.v_range
        Tc_map = np.full((len(v_range), len(v_range)), np.nan)
        
        total_points = len(v_range) * len(v_range)
        progress_bar = tqdm(total=total_points, desc="Calculating Tc")
        
        for i, v1 in enumerate(v_range):
            for j, v2 in enumerate(v_range):
                # Construct full v_vector
                v_vector = np.full(self.params.num_channels, self.params.v_min)
                
                # Set fixed channel values
                for channel, value in fixed_channels.items():
                    v_vector[channel] = value
                
                # Set varying channel values
                v_vector[varying_channels[0]] = v1
                v_vector[varying_channels[1]] = v2
                
                Tc = self._find_Tc_optimized(v_vector)
                if Tc is not None:
                    Tc_map[j, i] = Tc
                
                progress_bar.set_postfix({
                    f'v_{varying_channels[0]}': f'{v1:.3f}', 
                    f'v_{varying_channels[1]}': f'{v2:.3f}',
                    f'Tc': f'{Tc:.5f}' if Tc is not None else 'N/A'
                })
                progress_bar.update(1)
        
        progress_bar.close()
        return v_range, v_range, Tc_map
    
    def plot_phase_diagram_2d_slice(self, v_range1: np.ndarray, v_range2: np.ndarray, 
                                   Tc_map: np.ndarray, channel_labels: Tuple[str, str] = None):
        """
        Plot 2D phase diagram slice
        """
        if channel_labels is None:
            channel_labels = ('$v_0$', '$v_1$')
            
        plt.figure(figsize=(10, 8))
        
        V1, V2 = np.meshgrid(v_range1, v_range2)
        
        im = plt.imshow(Tc_map, extent=[v_range1[0], v_range1[-1], v_range2[0], v_range2[-1]], 
                       origin='lower', aspect='auto', cmap='viridis')
        
        valid_mask = ~np.isnan(Tc_map)
        if np.any(valid_mask):
            contours = plt.contour(V1, V2, Tc_map, levels=10, colors='white', alpha=0.7, linewidths=0.8)
            plt.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
        
        nan_mask = np.isnan(Tc_map)
        if np.any(nan_mask):
            V1_nan = V1[nan_mask]
            V2_nan = V2[nan_mask]
            plt.scatter(V1_nan, V2_nan, c='red', marker='x', s=20, alpha=0.6, label='No solution')
        
        plt.colorbar(im, label='$T_c$')
        plt.xlabel(channel_labels[0])
        plt.ylabel(channel_labels[1])
        plt.title('Superconducting Phase Diagram (2D Slice)')
        plt.grid(True, alpha=0.3)
        
        if np.any(nan_mask):
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def optimize_coupling_vector(self, initial_guess: np.ndarray = None, 
                               method: str = 'random_search', num_trials: int = 100) -> dict:

        if initial_guess is None:
            initial_guess = np.full(self.params.num_channels, 0.1)
        
        best_Tc = -np.inf
        best_v_vector = initial_guess.copy()
        best_result = None
        
        if method == 'random_search':
            print(f"Random search optimization with {num_trials} trials...")
            
            for trial in tqdm(range(num_trials), desc="Optimizing"):
                # Generate random v_vector
                v_vector = np.random.uniform(self.params.v_min, self.params.v_max, 
                                           self.params.num_channels)
                
                result = self._find_Tc_with_eigenvectors(v_vector, return_eigenvectors=True)
                
                if result is not None and result['Tc'] > best_Tc:
                    best_Tc = result['Tc']
                    best_v_vector = v_vector.copy()
                    best_result = result
        
        elif method == 'grid_search':
            # For simplicity, only do grid search for first 2 channels
            print("Grid search optimization (first 2 channels)...")
            
            for v0 in tqdm(self.params.v_range, desc="Grid search"):
                for v1 in self.params.v_range:
                    v_vector = initial_guess.copy()
                    v_vector[0] = v0
                    v_vector[1] = v1
                    
                    result = self._find_Tc_with_eigenvectors(v_vector, return_eigenvectors=True)
                    
                    if result is not None and result['Tc'] > best_Tc:
                        best_Tc = result['Tc']
                        best_v_vector = v_vector.copy()
                        best_result = result
        
        optimization_result = {
            'best_Tc': best_Tc,
            'best_v_vector': best_v_vector,
            'best_result': best_result,
            'method': method,
            'num_trials': num_trials if method == 'random_search' else len(self.params.v_range)**2
        }
        
        return optimization_result
    
    def analyze_specific_point(self, v_vector: np.ndarray):
        """
        Analyze detailed information for a specific parameter point
        """
        print(f"\n=== Analyzing point: v_vector = {v_vector} ===")
        
        result = self._find_Tc_with_eigenvectors(v_vector, return_eigenvectors=True)
        
        if result is None:
            print("No superconducting solution found at this point.")
            return None
        
        print(f"Tc = {result['Tc']:.6f}")
        print(f"Max eigenvalue = {result['max_eigenvalue']:.6f}")
        print(f"Degeneracy = {result['degeneracy']}")
        
        # Analyze each degenerate eigenvector
        for i, analysis in enumerate(result['eigenvector_analyses']):
            print(f"\nEigenvector {i}:")
            print(f"  Channel amplitudes: {analysis['channel_amplitudes']}")
            
            # Find dominant channels
            dominant_channels = np.argsort(analysis['channel_amplitudes'])[::-1][:3]
            print(f"  Dominant channels: {dominant_channels}")
            for ch in dominant_channels:
                print(f"    Channel {ch}: amplitude = {analysis['channel_amplitudes'][ch]:.4f}")
        
        return result

def main():
    print("Starting vector coupling constant analysis...")
    
    # Initialize parameters
    params = Parameters(
        Nk=12,                    
        num_channels=4,           # Number of coupling channels
        v_points=15
    )
    
    print(f"Nk={params.Nk}, v_points={params.v_points}")
    
    solver = SuperconductorSolver(params)
    
    # Method 1: Compute 2D slice varying first two channels
    print("\n=== Computing 2D phase diagram (v_0 vs v_1) ===")
    fixed_channels = {i: 0.05 for i in range(2, params.num_channels)}  # Fix other channels
    v_range1, v_range2, Tc_map = solver.compute_phase_diagram_2d_slice(fixed_channels)
    
    valid_points = ~np.isnan(Tc_map)
    if np.any(valid_points):
        max_Tc = np.nanmax(Tc_map)
        min_Tc = np.nanmin(Tc_map)
        print(f"\nComplete!")
        print(f"Valid points: {np.sum(valid_points)}/{Tc_map.size}")
        print(f"Tc range: {min_Tc:.5f} - {max_Tc:.5f}")
        
        # Find maximum Tc point in the slice
        max_idx = np.unravel_index(np.nanargmax(Tc_map), Tc_map.shape)
        max_v0 = v_range1[max_idx[1]]
        max_v1 = v_range2[max_idx[0]]
        print(f"Maximum Tc in slice: v_0 = {max_v0:.3f}, v_1 = {max_v1:.3f}")
        
        # Construct full v_vector for analysis
        best_v_vector_slice = np.full(params.num_channels, 0.05)
        best_v_vector_slice[0] = max_v0
        best_v_vector_slice[1] = max_v1
        
        solver.analyze_specific_point(best_v_vector_slice)
        
    else:
        print("Warning: No solution found in 2D slice!")

    # Plot the 2D slice
    solver.plot_phase_diagram_2d_slice(v_range1, v_range2, Tc_map, ('$v_0$', '$v_1$'))
    
    # Method 2: Global optimization
    print("\n=== Global optimization ===")
    optimization_result = solver.optimize_coupling_vector(method='random_search', num_trials=200)
    
    if optimization_result['best_result'] is not None:
        print(f"Global optimization found:")
        print(f"  Best Tc = {optimization_result['best_Tc']:.6f}")
        print(f"  Best v_vector = {optimization_result['best_v_vector']}")
        
        # Analyze the globally optimal point
        solver.analyze_specific_point(optimization_result['best_v_vector'])
    else:
        print("Global optimization failed to find any solution")

    # Method 3: Example of analyzing a specific point
    print("\n=== Analyzing specific example point ===")
    example_v_vector = np.array([0.15, 0.10, 0.08, 0.12])  # Example coupling vector
    solver.analyze_specific_point(example_v_vector)

if __name__ == "__main__":
    main()