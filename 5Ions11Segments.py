import numpy as np, matplotlib.pyplot as plt
from PulseShape import *
from scipy.linalg import null_space

import TrICal.trical as trical
import TrICal.trical.misc.constants as cst

#using TrICal for calculating normal mode frequencies and eigenvectors
N = 5  # Number of ions
mass = cst.convert_m_a(171)  # Mass of a Yb 171 ion

# Trapping strength (in rad/s), using values from paper
omega_x = 2 * np.pi * 2.59e6 # Direction x
omega_y = 2 * np.pi * 2.59e6  # Direction y
omega_z = 2 * np.pi * .315e6  # Direction z
omega = np.array([omega_x, omega_y, omega_z])
# Coefficients of the multivariate polynomial consistent with the trapping sxtrength above for a harmonic potential
alpha = np.zeros((3, 3, 3))
alpha[2, 0, 0] = mass * (omega_x) ** 2 / 2
alpha[0, 2, 0] = mass * (omega_y) ** 2 / 2
alpha[0, 0, 2] = mass * (omega_z) ** 2 / 2

pp = trical.classes.PolynomialPotential(alpha, N=N)

#calculate the frequencies and eigenvectors based on the specified 3D harmonic potential
ti = trical.classes.TrappedIons(N, pp, m=mass)
ti.equilibrium_position();
ti.normal_modes();
ti.principle_axis();

#We are interested in the transverse modes:
w_m = ti.w_pa[:N]  * 4.8377687e-17 
b = ti.b_pa[:N, :N]

dk = 2 * 2 * np.pi / (355e-9 / 5.2917725e-11) 
P = 11
i, j = 0, 1
tau = 200e-6 / 4.8377687e-17 #s
m = 2.83846e-25 / 1.8218779e-30  #kg for Yb 171 converted to rydberg
nBar = np.array([5] * N)
n = 400
muArr = np.linspace(2.38e6, 2.68e6, n) * 2 * np.pi * 4.8377687e-17 #Array of detunings
fid = np.zeros_like(muArr, dtype=complex)
Omega = np.zeros((n, P))

for l, mu in enumerate(muArr):
    C = C_Matrix(mu, w_m, b, tau, P, dk, i, j, m)
    C_r = np.real(C[:N, :])
    C_I = np.imag(C[:N, :])
    CIonOne = np.vstack((C_r, C_I))

    #Solve for an Omega that makes are alphas zero simultaneously. 
    A = null_space(CIonOne)
   
    Omega[l,:] = A[:, 0]

    #normalize Omega to satisfy required geometric phase
    D = D_matrix(mu, w_m, b, tau, P, dk, i, j, m)
    geoPhase = np.dot(Omega[l,:].T, np.dot(D, Omega[l,:]))
    geoPhase = np.abs(geoPhase)
    const = np.sqrt(np.pi / (4 * geoPhase))
    Omega[l,:] *= const
    fid[l] = fidelity(C, D, Omega[l,:], nBar)


muArr2 = muArr / 2 / np.pi / 1e6 / 4.8377687e-17

#plot detuning Vs Fidelity
plt.plot(muArr2, fid)
plt.xlabel("Detuning (MHz)", size=17)
plt.ylabel("Fidelity", size=17)
plt.tick_params(labelsize=15)
plt.show()

#Plot detuning Vs Rabi Frequency (~laser intensity)
plt.plot(muArr2, np.max(np.abs(Omega / 4.8377687e-17 / 1e6), axis=1), label="Segment = {}".format(i + 1))
plt.xlabel("Detuning (MHz)", size=17)
plt.ylabel("Rabi Frequency (MHz)", size=17)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15)

plt.show()

