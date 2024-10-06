import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import null_space
from PulseShape import *

dk = 2 * 2 * np.pi / (355e-9 / 5.2917725e-11) 
P = 5
i, j = 0, 1
tau = 100e-6 / 4.8377687e-17 #s
m = 2.83846e-25 / 1.8218779e-30  #kg for Yb 171 converted to rydberg
N = 2
#trap frequencies:
wx = 2 * np.pi * 4.38e6 * 4.8377687e-17
wz = 2 * np.pi *  600e3 * 4.8377687e-17 

#For the transverse modes of a 2 ion chain:
w_m = np.array([wx, np.sqrt(wx ** 2 - wz ** 2)])
b = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

#average number of phonons in the 2 modes
nBar = np.array([200, 200])
n = 340
muArr = np.linspace(4.28e6, 4.44e6, n) * 2 * np.pi * 4.8377687e-17 #Array of detunings
fid = np.zeros_like(muArr, dtype=complex)
Omega = np.zeros((n, P))


for l, mu in enumerate(muArr):
    C = C_Matrix(mu, w_m, b, tau, P, dk, i, j, m)
    C_r = np.real(C[:N, :])
    C_I = np.imag(C[:N, :])
    CIonOne = np.vstack((C_r, C_I))

    #Solve for an Omega that makes all alphas zero simultaneously. 
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

plt.plot(muArr2, fid)
plt.xlabel("Detuning (MHz)", size=17)
plt.ylabel("Fidelity", size=17)
plt.tick_params(labelsize=15)
plt.show()


