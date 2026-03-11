import numpy as np
import matplotlib.pyplot as plt


nslab = 30
mu = 0.023
t1 = 0.015 
t2 = 0.1
t01 = -0.0179
lambda1 = 0.017
omega = np.exp(1j * 2 * np.pi / 3)
a = 3.44
N = 2 * nslab

sigma_0 = np.array([[1, 0],
                [0, 1]])
sigma_z = np.array([[1, 0],
                [0, -1]])

def H_mono(kx):
# ----------------------------
# H0: take SNN hopping and 1-order Ising SOC
# ----------------------------
    h00 = np.array([[mu, t1 - 1j*lambda1],
                    [t1 + 1j*lambda1, mu]], dtype=complex)

    h0p1 = np.array([[t2, t1 - 1j*lambda1],   
                    [0, t2]], dtype=complex)  

    h0m1 = np.array([[t2, 0],
                    [t1 + 1j*lambda1, t2]], dtype=complex)


    H0 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        H0[2 * l:2 * l + 2, 2 * l:2 * l + 2] = h00
    for l in range(nslab - 1):
        H0[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = h0p1   
    for l in range(1, nslab):
        H0[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = h0m1  


    h10 = np.array([[t1 + 1j*lambda1, t2],
                    [t1 - 1j*lambda1, t1 + 1j*lambda1]], dtype=complex)

    h11 = np.array([[0, t2],
                    [0, 0]], dtype=complex)

    h12 = np.array([[0, 0],
                    [t1 - 1j*lambda1, 0]], dtype=complex)

    H1 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        H1[2 * l:2 * l + 2, 2 * l:2 * l + 2] = h10
    for l in range(nslab - 1):
        H1[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = h11
    for l in range(1, nslab):
        H1[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = h12

    h20 = np.array([[t1 - 1j*lambda1, t1 + 1j*lambda1],
                    [t2, t1 - 1j*lambda1]], dtype=complex)

    h21 = np.array([[0, t1 + 1j*lambda1],
                    [0, 0]], dtype=complex)

    h22 = np.array([[0, 0],
                    [t2, 0]], dtype=complex)

    H2 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        H2[2 * l:2 * l + 2, 2 * l:2 * l + 2] = h20
    for l in range(nslab - 1):
        H2[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = h21
    for l in range(1, nslab):
        H2[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = h22

    h30 = np.array([[0, 0],
                    [t2, 0]], dtype=complex)

    h31 = np.zeros((2, 2), dtype=complex)

    h32 = np.array([[0, 0],
                    [t2, 0]], dtype=complex)

    H3 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        H3[2 * l:2 * l + 2, 2 * l:2 * l + 2] = h30
    for l in range(nslab - 1):
        H3[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = h31
    for l in range(1, nslab):
        H3[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = h32

    h40 = np.array([[0, t2],
                    [0, 0]], dtype=complex)

    h42 = np.zeros((2, 2), dtype=complex)

    h41 = np.array([[0, t2],
                    [0, 0]], dtype=complex)

    H4 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        H4[2 * l:2 * l + 2, 2 * l:2 * l + 2] = h40
    for l in range(nslab - 1):
        H4[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = h41
    for l in range(1, nslab):
        H4[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = h42

    exp_ikx = np.exp(1j * kx * a)
    exp_2ikx = np.exp(2j * kx * a)
    
    #spin-up space
    H11 = (H0 
           + H1 * exp_ikx 
           + H2 / exp_ikx        
           + H3 * exp_2ikx 
           + H4 / exp_2ikx)   

    #spin-down space
    H22 = (np.conj(H0) 
           + np.conj(H1) * exp_ikx 
           + np.conj(H2) / exp_ikx 
           + np.conj(H3) * exp_2ikx      
           + np.conj(H4) / exp_2ikx)   

    zero_block = np.zeros_like(H0)

    H_mono = np.block([
        [H11, zero_block],
        [zero_block, H22]
    ])

    if np.allclose(H_mono, H_mono.conj().T, atol=1e-12):
        H_mono = 0.5 * (H_mono + H_mono.conj().T)
    else:
        raise ValueError(f'H(kx={kx:.6f}) is non-Hermitian beyond tolerance')
    return H_mono


def H_bi(kx):

    #interlayer onsite and NN hopping
    hint00 = np.array([[t01, t01],
                    [t01, t01]], dtype=complex)

    hint01 = np.array([[0, t01],  
                    [0, 0]], dtype=complex) 

    hint02 = np.array([[0, 0],
                    [t01, 0]], dtype=complex)

    Hint0 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        Hint0[2 * l:2 * l + 2, 2 * l:2 * l + 2] = hint00
    for l in range(nslab - 1):
        Hint0[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = hint01  
    for l in range(1, nslab):
        Hint0[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = hint02  


    hint10 = np.array([[t01, 0],
                    [t01, t01]], dtype=complex)

    hint11 = np.array([[0, 0],
                    [0, 0]], dtype=complex)

    hint12 = np.array([[0, 0],
                    [t01, 0]], dtype=complex)

    Hint1 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        Hint1[2 * l:2 * l + 2, 2 * l:2 * l + 2] = hint10
    for l in range(nslab - 1):
        Hint1[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = hint11
    for l in range(1, nslab):
        Hint1[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = hint12

    hint20 = np.array([[t01, t01],
                    [0, t01]], dtype=complex)

    hint21 = np.array([[0, t01],
                    [0, 0]], dtype=complex)

    hint22 = np.array([[0, 0],
                    [0, 0]], dtype=complex)

    Hint2 = np.zeros((N, N), dtype=complex)
    for l in range(nslab):
        Hint2[2 * l:2 * l + 2, 2 * l:2 * l + 2] = hint20
    for l in range(nslab - 1):
        Hint2[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = hint21
    for l in range(1, nslab):
        Hint2[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = hint22    

    exp_ikx = np.exp(1j * kx * a)

    Hint =   (Hint0 
           + Hint1 * exp_ikx 
           + Hint2 / exp_ikx)

    H_bi = np.block([
        [H_mono(kx), np.kron(sigma_0, Hint)],
        [np.kron(sigma_0, Hint.conj().T), H_mono(-kx)]
    ])

    if np.allclose(H_bi, H_bi.conj().T, atol=1e-12):
        H_bi = 0.5 * (H_bi + H_bi.conj().T)
    else:
        raise ValueError(f'H(kx={kx:.6f}) is non-Hermitian beyond tolerance')
    return H_bi

def H_normal(kx):

    H_e = H_bi(kx)

    H_h = -np.conj(H_bi(-kx))

    zero_block = np.zeros_like(H_e)

    H = np.block([[H_e, zero_block],
                 [zero_block, H_h]])
    
    return H

def PairingMatrix(kx, pairing):
    """Construct the pairing matrix for the requested representation."""
    def S_plus_pairing(kx):
        
        exp_ikx = np.exp(1j * kx * a)
        exp_minus_ikx = np.exp(-1j * kx * a)

        delta00 = np.array([[0, 1j * omega / 2],
                            [-1j * omega / 2, 0]], dtype=complex)
        delta01 = np.array([[0, 1j * (omega**2) / 2],
                            [0, 0]], dtype=complex)
        delta02 = np.array([[0, 0],
                            [-1j * (omega**2) / 2, 0]], dtype=complex)

        Delta0 = np.zeros((N, N), dtype=complex)
        for l in range(nslab):
            Delta0[2 * l:2 * l + 2, 2 * l:2 * l + 2] = delta00
        for l in range(nslab - 1):
            Delta0[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = delta01
        for l in range(1, nslab):
            Delta0[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = delta02

        delta10 = np.array([[-1j / 2, 0],
                            [1j * (omega**2) / 2, -1j / 2]], dtype=complex)
        delta11 = np.zeros((2, 2), dtype=complex)
        delta12 = np.array([[0, 0],
                            [1j * omega / 2, 0]], dtype=complex)

        Delta1 = np.zeros((N, N), dtype=complex)
        for l in range(nslab):
            Delta1[2 * l:2 * l + 2, 2 * l:2 * l + 2] = delta10
        for l in range(nslab - 1):
            Delta1[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = delta11
        for l in range(1, nslab):
            Delta1[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = delta12

        delta20 = np.array([[1j / 2, -1j * (omega**2) / 2],
                            [0, 1j / 2]], dtype=complex)
        delta21 = np.array([[0, -1j * omega / 2],
                            [0, 0]], dtype=complex)
        delta22 = np.zeros((2, 2), dtype=complex)

        Delta2 = np.zeros((N, N), dtype=complex)
        for l in range(nslab):
            Delta2[2 * l:2 * l + 2, 2 * l:2 * l + 2] = delta20
        for l in range(nslab - 1):
            Delta2[2 * (l + 1):2 * (l + 1) + 2, 2 * l:2 * l + 2] = delta21
        for l in range(1, nslab):
            Delta2[2 * (l - 1):2 * (l - 1) + 2, 2 * l:2 * l + 2] = delta22

        S_plus = (Delta0
                + Delta1 * exp_ikx
                + Delta2 * exp_minus_ikx)

        return S_plus

    if pairing == 4:
        
        Delta11 = S_plus_pairing(kx)

        zero_block1 = np.zeros_like(Delta11)

        Delta_mono = np.block([
            [zero_block1, Delta11],
            [-Delta11, zero_block1]
        ])

        Delta_bi = np.kron(sigma_0, Delta_mono)

        zero_block2 = np.zeros_like(Delta_bi)

        Delta = np.block([
            [zero_block2, Delta_bi],
            [Delta_bi.conj().T, zero_block2]
        ])

        return Delta

    elif pairing == 1:

        Delta11 = S_plus_pairing(kx)

        zero_block1 = np.zeros_like(Delta11)

        Delta_mono = np.block([
            [-np.conj(Delta11), zero_block1],
            [zero_block1, -Delta11]
        ])

        Delta_bi = np.kron(sigma_z, Delta_mono)

        zero_block2 = np.zeros_like(Delta_bi)

        Delta = np.block([
            [zero_block2, Delta_bi],
            [Delta_bi.conj().T, zero_block2]
        ])

        return Delta


def H_total(kx):

    irre = 1
    v1 = 0.005
    H_n = H_normal(kx)
    Delta = v1 * PairingMatrix(kx,irre)

    if np.size( H_n , 1) == np.size(Delta, 1):
        H = H_n + Delta
    else:
        raise ValueError(f'H(kx={kx:.6f}) is not match with PairingMatrix')

    if np.allclose(H, H.conj().T, atol=1e-12):
        H = 0.5 * (H + H.conj().T)
    else:
        raise ValueError(f'H(kx={kx:.6f}) is non-Hermitian beyond tolerance')
    return H

nk = 200
kxs = np.linspace(-np.pi/a, np.pi/a, nk)
energies = []

for kx in kxs:
    Hk = H_total(kx)
    eigs = np.real(np.linalg.eigvalsh(Hk)) 
    energies.append(eigs)

energies = np.array(energies)


plt.figure(figsize=(6, 8))
for n in range(energies.shape[1]):
    plt.plot(kxs, energies[:, n], color='red', linewidth=0.8)

plt.xlim(-np.pi/a, np.pi/a)
plt.ylim(-1, 1)
plt.xlabel(r'$k_x$')
plt.ylabel('Energy (eV)')
plt.xticks([-np.pi/a, 0, np.pi/a], [r'$-\pi$', '0', r'$\pi$'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.title('Simplified 1st-order Hopping Band Structure')
plt.tight_layout()
plt.savefig(r'C:\Users\17167\Desktop\2H-NbSe2\edge\edge_state_d0.png', dpi=300)

# ----------------------------
# Spectra for kx points hosting zero-energy modes
# ----------------------------
zero_tol = 1e-4
zero_k_indices = np.where(np.any(np.isclose(energies, 0.0, atol=zero_tol), axis=1))[0]

if zero_k_indices.size == 0:
    closest_idx = int(np.argmin(np.min(np.abs(energies), axis=1)))
    zero_k_indices = np.array([closest_idx])
    print(f'No exact zero-energy states found; using kx ~= {kxs[closest_idx]:.6f} with minimal |E|.')
else:
    print(f'Found {zero_k_indices.size} kx points with ~zero-energy states.')

for idx in zero_k_indices:
    kx_target = kxs[idx]
    Hk_target = H_total(kx_target)
    eigvals_target, eigvecs_target = np.linalg.eigh(Hk_target)

    plt.figure(figsize=(5, 4))
    plt.scatter(np.arange(len(eigvals_target)), eigvals_target, s=12, color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Energy (eV)')
    plt.title(rf'Energy Spectrum at $k_x={kx_target:.4f}$')
    plt.tight_layout()
    safe_kx_label = f'{kx_target:.6f}'.replace('-', 'm').replace('.', 'p')
    plt.savefig(fr'C:\Users\17167\Desktop\2H-NbSe2\edge\Energy_Spectrum_d0_kx_{safe_kx_label}.png', dpi=300)

    mid_indices = np.argsort(np.abs(eigvals_target))[:2]
    mid_indices = mid_indices[np.argsort(eigvals_target[mid_indices])]
    component_index = np.arange(eigvecs_target.shape[0])

    plt.figure(figsize=(7, 4))
    for zero_idx in mid_indices:
        probability = np.abs(eigvecs_target[:, zero_idx])**2
        label = fr'$n={zero_idx}$, $E={eigvals_target[zero_idx]:.4f}\,\mathrm{{eV}}$'
        plt.plot(component_index, probability, linewidth=1.2, label=label)

    plt.xlabel('Component index')
    plt.ylabel(r'$|\psi|^2$')
    plt.title(rf'Probability Density at $k_x={kx_target:.4f}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fr'C:\Users\17167\Desktop\2H-NbSe2\edge\Zero_mode_d0_kx_{safe_kx_label}.png', dpi=300)

plt.show()


