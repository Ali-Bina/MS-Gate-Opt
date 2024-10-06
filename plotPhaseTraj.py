##############################################################################
# Code for plotting phase space trajectories for segmented pulses as well as #
# for plotting the pulse itself                                              #
##############################################################################

import numpy as np, matplotlib.pyplot as plt
from PulseShape import  C_Matrix
import TrICal.trical as trical
import TrICal.trical.misc.constants as cst
hbar = 1

tau = 200e-6 / 4.8377687e-17
mu =    2.492048192771084e6 * 2 * np.pi / 2.0670687e16
dk = 2 * 2 * np.pi / (355e-9 / 5.2917725e-11)
m = 2.83846e-25 / 1.8218779e-30 #ARU




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
ti.equilibrium_position()
ti.normal_modes()
ti.principle_axis()

#We are interested in the transverse modes:
w_m = ti.w_pa[:N]  * 4.8377687e-17 
b = ti.b_pa[:N, :N]


nBar = np.array([0] * N)
P = 9
i, j = 1, 2


#Rabi frequency of each pulse segment obtined from optimization
Omega = np.array([-0.26023739, -0.54436495, -0.6778568 , -0.79672018, -0.88024713,
       -0.80116632, -0.68295309, -0.54660868, -0.26088467])* 1e6  / 2.0670687e16

#Plot trajectory
n = 1000
time = np.linspace(0, tau, n)
alpha = np.empty((n, 2 * N), dtype=complex)

for k in range(n):
    C = C_Matrix(mu, w_m, b, time[k], P, dk, i, j, m)
    alpha[k, :] = np.dot(C, Omega).T

c =  plt.cm.get_cmap('hsv', P + 1) #Picking a set of P colors so that each segment is a different color
ind = 0
for k in range(1, P+1):
    lowerInd = int(np.floor(n * (k - 1) / P))
    upperInd = int(np.floor(n * k / P))
    plt.plot(np.real(alpha[lowerInd:upperInd, ind]), np.imag(alpha[lowerInd:upperInd, ind]),c=c(k-1), label="Segment: {}".format(k))
plt.legend(fontsize=15)
plt.plot(np .real(alpha)[-1, ind], np.imag(alpha)[-1, ind], 'o', markersize=8, color='k')
plt.hlines(y = 0, xmin=-0.1, xmax=0.4, color='k')
plt.vlines(x=0, ymin=-0.2, ymax=0.2, color='k')
plt.xlabel("Re{$\\alpha$}", size=17)
plt.ylabel("Im{$\\alpha$}", size=17)
plt.tick_params(labelsize=15)
plt.show()


#Plot pulse as a function of time
time *=  4.8377687e-17 * 1e6
Omega *= 2.0670687e16 / 1e6
for k in range(1, P+1):
    lowerInd = int(np.floor(n * (k - 1) / P))
    upperInd = int(np.floor(n * k / P))
    length = upperInd - lowerInd

    plt.plot(time[lowerInd:upperInd], np.array([Omega[k - 1]] * length), c=c(k-1))
    plt.fill_between(time[lowerInd:upperInd], np.array([Omega[k - 1]] * length), color=c(k-1))

plt.xlabel("Time ($\\mu$s)", size=17)
plt.ylabel("Rabbi Frequency (MHz)", size=17)
plt.tick_params(labelsize=15)
plt.show()

        
#Line plot of mode frequency and detunings
mu2  = mu / 2 / np.pi *  2.0670687e16 / 1e6
w = w_m / 2/ np.pi *  2.0670687e16 / 1e6
for freq in w:
    plt.vlines(x=freq, ymin=0, ymax=10, color='r')

plt.vlines(x=mu2, ymin=0, ymax=12, color='g')

plt.hlines(y = 0, xmin=-4.8, xmax=4.8, color='k')
plt.yticks([], " ")
plt.xticks(list(w) + [mu2], ["{:.2f}".format(x) for x in list(w) + [mu2]])
#plt.axis('off')
#plt.xscale('log')
plt.tick_params(labelsize=15)
plt.xlabel("Normal mode frequency (MHz)", size=17)
plt.show()
