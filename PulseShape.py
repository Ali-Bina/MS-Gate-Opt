####################################################################################
# Implements the optimization technique outlined in chapter 4 of Manning's thesis' #
# Uses temporal pulse shaping to optimize the fidelity of MS entangling gate       #
####################################################################################

import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
import math, cmath
#unless otherwise specified, everything is in atomic Rydberg units
hbar =  1

def Cintegral(t0, t1, mu, w_m):

    preFactor = 1 / (mu ** 2 - w_m ** 2)
    term1 = np.exp(1j * w_m * t1) * (mu * np.cos(mu * t1) - 1j * w_m * np.sin(mu * t1))
    term2 = np.exp(1j * w_m * t0) * (mu * np.cos(mu * t0) - 1j * w_m * np.sin(mu * t0))
    integral = preFactor * (term1 - term2)

    #Numerical integration yields the same result
    f = lambda t: -np.sin(mu * t) * np.exp(1j * w_m * t)
    time = np.linspace(t0, t1, int(1e5))
    #  integral = np.trapz(f(time), x=time)
    integral =  (np.exp(complex(0, 1) * w_m * t1) * (complex(0, 1) * w_m * cmath.sin(mu * t1) - mu * cmath.cos(mu * t1)) - np.exp(complex(0, 1) * w_m * t0) * (complex(0, 1) * w_m * cmath.sin(mu * t0) - mu * cmath.cos(mu * t0))) / (-mu ** 2 + w_m ** 2) 
    return integral


def C_Matrix(mu, w_m, b, tau, P, dk, i, j, mass):

    """ Matrix describing the alpha parameters of the displacement operator for a segmented pulse
         Input:
         mu: detuning
         w_m: vector of normal mode frequencies
         b: matrix of corresponding normal mode eigenvector
         tau: gate time
         P: Number of intervals
        dk: difference in ramman beam wave-vectors
        i, j: ions participating in the MS gate
        m: ion mass"""

    
    N = len(w_m)
    Ci = np.empty((N ,P), dtype=complex)
    Cj = np.empty((N ,P), dtype=complex)
    for m in range(N):

        for p in range(1, P+1):
            w = w_m[m]
            eta_im = dk * np.sqrt(hbar / (2 * mass * w)) * b[i, m]
            eta_jm = dk * np.sqrt(hbar / (2 * mass * w)) * b[j, m]

            Ci [m, p - 1] = eta_im * Cintegral((p - 1) * tau / P, p * tau / P, mu, w)
            Cj [m, p -1] = eta_jm * Cintegral((p - 1) * tau / P, p * tau / P, mu, w)


    Ctot = np.empty((2*N, P), dtype=complex)
    Ctot[0:N, :] = Ci
    Ctot[N:, :] = Cj

    return Ctot

def Dintegral(T0, T1, tau0, tau1, mu, w_m, final=False):

    #evaluating the double integral 

    #analytical expression for integral evaluated in Maple
    if not final:
       integral = 0.1e1 / (mu - w_m) ** 2 / (mu + w_m) ** 2 * (math.sin(T0 * mu + T0 * w_m + tau0 * mu - tau0 * w_m) * mu ** 2 - math.sin(T0 * mu + T0 * w_m + tau0 * mu - tau0 * w_m) * w_m ** 2 + math.sin(T0 * mu - T0 * w_m - tau1 * mu + tau1 * w_m) * mu ** 2 + math.sin(T0 * mu - T0 * w_m - tau1 * mu + tau1 * w_m) * w_m ** 2 + math.sin(T0 * mu + T0 * w_m - tau0 * mu - tau0 * w_m) * mu ** 2 + math.sin(T0 * mu + T0 * w_m - tau0 * mu - tau0 * w_m) * w_m ** 2 + math.sin(T0 * mu - T0 * w_m + tau1 * mu + tau1 * w_m) * mu ** 2 - math.sin(T0 * mu - T0 * w_m + tau1 * mu + tau1 * w_m) * w_m ** 2 - math.sin(T0 * mu - T0 * w_m + tau0 * mu + tau0 * w_m) * mu ** 2 + math.sin(T0 * mu - T0 * w_m + tau0 * mu + tau0 * w_m) * w_m ** 2 - math.sin(T0 * mu + T0 * w_m - tau1 * mu - tau1 * w_m) * mu ** 2 - math.sin(T0 * mu + T0 * w_m - tau1 * mu - tau1 * w_m) * w_m ** 2 - math.sin(T0 * mu - T0 * w_m - tau0 * mu + tau0 * w_m) * mu ** 2 - math.sin(T0 * mu - T0 * w_m - tau0 * mu + tau0 * w_m) * w_m ** 2 - math.sin(T0 * mu + T0 * w_m + tau1 * mu - tau1 * w_m) * mu ** 2 + math.sin(T0 * mu + T0 * w_m + tau1 * mu - tau1 * w_m) * w_m ** 2 - math.sin(T1 * mu + T1 * w_m + tau0 * mu - tau0 * w_m) * mu ** 2 + math.sin(T1 * mu + T1 * w_m + tau0 * mu - tau0 * w_m) * w_m ** 2 - math.sin(T1 * mu - T1 * w_m - tau1 * mu + tau1 * w_m) * mu ** 2 - math.sin(T1 * mu - T1 * w_m - tau1 * mu + tau1 * w_m) * w_m ** 2 - math.sin(T1 * mu + T1 * w_m - tau0 * mu - tau0 * w_m) * mu ** 2 - math.sin(T1 * mu + T1 * w_m - tau0 * mu - tau0 * w_m) * w_m ** 2 - math.sin(T1 * mu - T1 * w_m + tau1 * mu + tau1 * w_m) * mu ** 2 + math.sin(T1 * mu - T1 * w_m + tau1 * mu + tau1 * w_m) * w_m ** 2 + math.sin(T1 * mu - T1 * w_m + tau0 * mu + tau0 * w_m) * mu ** 2 - math.sin(T1 * mu - T1 * w_m + tau0 * mu + tau0 * w_m) * w_m ** 2 + math.sin(T1 * mu + T1 * w_m - tau1 * mu - tau1 * w_m) * mu ** 2 + math.sin(T1 * mu + T1 * w_m - tau1 * mu - tau1 * w_m) * w_m ** 2 + math.sin(T1 * mu - T1 * w_m - tau0 * mu + tau0 * w_m) * mu ** 2 + math.sin(T1 * mu - T1 * w_m - tau0 * mu + tau0 * w_m) * w_m ** 2 + math.sin(T1 * mu + T1 * w_m + tau1 * mu - tau1 * w_m) * mu ** 2 - math.sin(T1 * mu + T1 * w_m + tau1 * mu - tau1 * w_m) * w_m ** 2 + 2 * math.sin(T0 * mu - T0 * w_m - tau1 * mu + tau1 * w_m) * mu * w_m - 2 * math.sin(T0 * mu + T0 * w_m - tau0 * mu - tau0 * w_m) * mu * w_m + 2 * math.sin(T0 * mu + T0 * w_m - tau1 * mu - tau1 * w_m) * mu * w_m - 2 * math.sin(T0 * mu - T0 * w_m - tau0 * mu + tau0 * w_m) * mu * w_m - 2 * math.sin(T1 * mu - T1 * w_m - tau1 * mu + tau1 * w_m) * mu * w_m + 2 * math.sin(T1 * mu + T1 * w_m - tau0 * mu - tau0 * w_m) * mu * w_m - 2 * math.sin(T1 * mu + T1 * w_m - tau1 * mu - tau1 * w_m) * mu * w_m + 2 * math.sin(T1 * mu - T1 * w_m - tau0 * mu + tau0 * w_m) * mu * w_m) / 4

       t2Lower = lambda x: tau0
       t2Upper = lambda x:  tau1
       #integral =  dblquad(f, T0, T1, t2Lower, t2Upper)[0]

    else:
       integral = -0.1e1 / (mu - w_m) ** 2 / (mu + w_m) ** 2 * (-math.sin(2 * mu * T0) * w_m ** 3 - math.sin(mu * T0 + T0 * w_m + tau0 * mu - tau0 * w_m) * mu ** 3 - math.sin(mu * T0 + T0 * w_m - tau0 * mu - tau0 * w_m) * mu ** 3 + math.sin(mu * T0 - T0 * w_m + tau0 * mu + tau0 * w_m) * mu ** 3 + math.sin(mu * T0 - T0 * w_m - tau0 * mu + tau0 * w_m) * mu ** 3 + math.sin(2 * mu * T1) * w_m ** 3 + math.sin(mu * T1 + T1 * w_m + tau0 * mu - tau0 * w_m) * mu ** 3 + math.sin(mu * T1 + T1 * w_m - tau0 * mu - tau0 * w_m) * mu ** 3 - math.sin(mu * T1 - T1 * w_m + tau0 * mu + tau0 * w_m) * mu ** 3 - math.sin(mu * T1 - T1 * w_m - tau0 * mu + tau0 * w_m) * mu ** 3 + 2 * T1 * mu ** 3 * w_m - 2 * T1 * mu * w_m ** 3 + math.sin(2 * mu * T0) * mu ** 2 * w_m + math.sin(mu * T0 + T0 * w_m + tau0 * mu - tau0 * w_m) * mu * w_m ** 2 + 2 * math.sin(mu * T0 + T0 * w_m - tau0 * mu - tau0 * w_m) * mu ** 2 * w_m - math.sin(mu * T0 + T0 * w_m - tau0 * mu - tau0 * w_m) * mu * w_m ** 2 - math.sin(mu * T0 - T0 * w_m + tau0 * mu + tau0 * w_m) * mu * w_m ** 2 + 2 * math.sin(mu * T0 - T0 * w_m - tau0 * mu + tau0 * w_m) * mu ** 2 * w_m + math.sin(mu * T0 - T0 * w_m - tau0 * mu + tau0 * w_m) * mu * w_m ** 2 - math.sin(2 * mu * T1) * mu ** 2 * w_m - math.sin(mu * T1 + T1 * w_m + tau0 * mu - tau0 * w_m) * mu * w_m ** 2 - 2 * math.sin(mu * T1 + T1 * w_m - tau0 * mu - tau0 * w_m) * mu ** 2 * w_m + math.sin(mu * T1 + T1 * w_m - tau0 * mu - tau0 * w_m) * mu * w_m ** 2 + math.sin(mu * T1 - T1 * w_m + tau0 * mu + tau0 * w_m) * mu * w_m ** 2 - 2 * math.sin(mu * T1 - T1 * w_m - tau0 * mu + tau0 * w_m) * mu ** 2 * w_m - math.sin(mu * T1 - T1 * w_m - tau0 * mu + tau0 * w_m) * mu * w_m ** 2 - 2 * T0 * mu ** 3 * w_m + 2 * T0 * mu * w_m ** 3) / mu / 4
       

    

    return integral




def D_matrix(mu, w_m, b, tau, P, dk, i, j, mass):
    """Matrix specifying the geometric phase, chi_ij for a segmented pulse"""

    N  = len(w_m)
    D = np.zeros( (P, P), dtype=complex)

    for m in range(N):

         w = w_m[m]
         eta_im = dk * np.sqrt(hbar / (2 * mass * w)) * b[i, m]
         eta_jm = dk * np.sqrt(hbar / (2 * mass * w)) * b[j, m]

         for l in range(1, P+1):

            for k in range(1, l + 1):

    
               
                if k < l:
                    D[l - 1, k -1] += Dintegral((l-1) * tau / P, l * tau / P, (k - 1) * tau / P, k * tau / P, mu, w) * eta_im * eta_jm

                else:
                    D[l - 1, k -1] += Dintegral((l-1) * tau / P, l * tau / P, (k - 1) * tau / P, k * tau / P, mu, w, final=True) * eta_im * eta_jm

    
    return D * 2


def B_matrix(nBar, Cmatrix):

    """Matrix describing the approximarte infidelity (1-F)
         inputs:
         nBar: vector of average phonon numbers in each motional mode m
         """

    N = len(nBar)
    P = Cmatrix.shape[1]
    B = np.zeros((P,P), dtype=complex)

    Ci = Cmatrix[0:N, :]
    Cj = Cmatrix[N:, :]

    for l in range(P):
        for k in range(P):

            for m in range(N):

                if nBar[m] == 0:
                    beta_m = 1
                else:
                    beta_m = 1 / np.tanh(0.5 * np.log(1 + 1 / nBar[m]))
                B[l, k] += beta_m * (np.conj(Ci[m, l]) * Ci[m, k]  + np.conj(Cj[m, l]) * Cj[m, k])

    return B/4

def alpha(Omega, Ci):
    """Use the optimized set of pulse powers to calculate the alpha parameters at the end of the pulse"""
    return np.dot(Ci, Omega)

def chi(mu, w_m, b, tau, P, dk, i, j, mass, Omega):
    """Geometric phase implemented numerically. Sooo Slow implement in C!"""
    N = len(w_m)
    geo  = 0j
    t2Lower = lambda x: 0
    t2Upper = lambda x:  x
    def O(t):
        ind = np.int(np.floor(t / tau * P))
        return Omega[ind]

  

    for m in range(N):

        w = w_m[m]
        f = lambda t2, t1: O(t1) * O(t2) * np.sin(mu * t1) * np.sin(mu * t2) * np.sin(w * (t2 - t1))
        eta_im = dk * np.sqrt(hbar / (2 * mass * w)) * b[i, m]
        eta_jm = dk * np.sqrt(hbar / (2 * mass * w)) * b[j, m]

        geo +=   dblquad(f, 0, tau, t2Lower, t2Upper)[0] * eta_im * eta_jm * 2

    return geo
   

def fidelity(Cmat, Dmat, Omega, nBar):
    """Unapproximated fidelity between a bell state |00>-i|11> and the final state of the trapped ion system"""
    N = len(nBar)
    alphaI = np.dot(Cmat[:N, :], Omega)
    alphaJ = np.dot(Cmat[N:, :], Omega)

    Beta = 1/2 * np.array([1 if n == 0 else (1 / np.tanh(0.5*np.log(1 + 1/ n))) for n in nBar])
    #Beta = 4 * (nBar + 1/2) #from stefanie Miller M.s. thesis

    GammaI = np.exp(-np.sum((np.abs(alphaI) ** 2 ) * Beta))
    GammaJ = np.exp(-np.sum((np.abs(alphaJ) ** 2 ) * Beta))
    GammaPlus = np.exp(-np.sum((np.abs(alphaI + alphaJ) ** 2) * Beta))
    GammaMinus = np.exp(-np.sum((np.abs(alphaI - alphaJ) ** 2) * Beta))
    geoPhase = np.abs(np.dot(Omega.T, np.dot(Dmat, Omega)))
  
    geoPhase = np.pi/4
    F = 1/8 *  (2 + 2 * np.sin(2 * geoPhase) *  (GammaI + GammaJ) + GammaPlus + GammaMinus)

    return F










  


