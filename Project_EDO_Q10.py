__author__ = "Jean-Claude,Bienvenue,Gilchrist"
__copyright__ = "Copyright @2021 Jean-Claude,Bienvenue,Gilchrist"
__version__ = "1.0"
__date__ = "18 Novembre 2021"
__email__ = "jeanclaudemitchozounou018@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import linregress


# définition des paramètres du modèle
k = 1.
x0 = 3  # population initiale des proies
y0 = 5  # population initiale des prédateurs
X0 = np.array([x0, y0])

# définition du système différentiel
def f(X, t=0):
    return np.array([X[0] - X[0] * X[1],
                     k * (X[0] * X[1] - X[1])])


def EulerExplicite(f, T, N, X0):
    h = T / N
    time = [0]
    Y = np.zeros((2, N))
    Y[:, 0] = X0
    t0 = 0
    for k in range(N - 1):
        X0 = X0 + h * f(X0, t0)
        t0 = t0 + h
        Y[:, k + 1] = X0
        time.append(t0)
    return Y


def EulerImplicite(f, T, N, X0):
    time = [0]
    h = T / N
    Y = np.zeros((2, N))
    Y[:, 0] = X0
    t0 = 0

    def g(X):
        return X - h * f(X, t0) - X0

    for k in range(N - 1):
        X0 = fsolve(g, X0)
        t0 = t0 + h
        Y[:, k + 1] = X0
        time.append(t0)
    return Y


def EulerSimplectique(f, T, N, X0):
    YEE3 = EulerExplicite(f, T, N, X0)
    h = T/N
    Y2 = [X0[1]]
    t0 = 0
    for k in range(N-1):
        X0[1] = X0[1] - h * f([YEE3[0, :][k+1], X0[1]], t0)[1]
        Y2.append(X0[1])


    YES = np.zeros((2, N))
    YES[0, :] = YEE3[0, :]
    YES[1, :] = Y2
    return YES


def RungeKutta_4(f, T, N, X0):
    h = T / N
    time = [0]
    Y = np.zeros((2, N))
    Y[:, 0] = X0
    t0 = 0
    for k in range(N - 1):
        k1 = f(X0, t0)
        k2 = f(X0 + h/2 * k1, t0)
        k3 = f(X0 + h/2 * k2, t0)
        k4 = f(X0 + h * k3, t0)
        X0 = X0 + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h
        Y[:, k + 1] = X0
        time.append(t0)
    return Y


def Lyap_fonct(X):
    return X[0, :] + X[1, :] - np.log(X[0, :] * X[1, :])





#QUESTION 10
"""	PRENONS COMME SOLUTION DE RÉFERENCE CELLE OBTENUE AVEC N = 20000 PAS DE DISCRÉTISATION,
	REPRÉSENTONS LES ERREURS AU TEMPS T=40 TROUVÉES EN UTILISANT SUCCESSIVEMENT 
	NN = [10000, 10200, 10400, 10600, 10800] PAS DE DISCRÉTISATION """

#definition des variables
T = 40

N = 20000 
t1 = np.linspace(0,T,N+1) #Decoupage de l'intervalle [0,T] en N
NN = [10000, 10200, 10400, 10600, 10800]

"""
	ee_r : euler explicite en temps final
	ei_r : euler Implicite en temps final
	es_r : euler Symplectique en temps final
	rk_r : Runge Kutta en temps final
	NN    : pour stocker les valeurs de N , n= [10, 20, 40, 60, 80]
	H = []   : pour stocker le parametre h=T/N+1
	eEE = [] : pour stocker les erreurs avec Euler Explicite
	eEI = [] : pour stocker les erreurs avec Euler Implicite
	eES = [] : pour stocker les erreurs avec Euler Simplectique
	eRK = [] : pour stocker les erreurs avec Runge Kutta-4
"""


YEE = EulerExplicite(f, T, N, X0)

ee_r = YEE[:, N-1]

YEI = EulerImplicite(f, T, N, X0)

ei_r = YEI[:, N-1]


YRK4 = RungeKutta_4(f, T, N, X0)

rk_r = YRK4[:, N-1]


YES = EulerSimplectique(f, T, N, X0)

es_r = YES[:, N-1]


eEE, eEI, eES, eRK = [], [], [], []
H = []

for N in NN:
    YEE1 = EulerExplicite(f, T, N, X0)
    eEE.append(np.linalg.norm(ee_r - YEE1[:, N-1]))

    YEI1 = EulerImplicite(f, T, N, X0)
    eEI.append(np.linalg.norm(ei_r - YEI1[:, N-1]))

    YRK4_1 = RungeKutta_4(f, T, N, X0)
    eRK.append(np.linalg.norm(rk_r - YRK4_1[:, N-1]))

    YES1 = EulerSimplectique(f, T, N, X0)
    eES.append(np.linalg.norm(ei_r - YES1[:, N - 1]))

    H.append(40/(N+1))

plt.figure()
plt.plot(NN, eEE, '-x', label='EE')
plt.plot(NN, eEI, '-o', label='EI')
plt.plot(NN, eES, '-x', label='ES')
plt.plot(NN, eRK, '-o', label='RK')
plt.plot(NN, NN, label='O(N)')
plt.plot(NN,[n**2 for n in NN], label='O(N^3)')
plt.xlabel('N')
plt.ylabel('erreur temps final')
plt.title('erreurs avec EE, EI, ES, RK')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

pEE = linregress(np.log(np.array(H)), np.log(np.array(eEE)))
print("L'ordre de convergence d'euler explicite est : ", pEE)
print("  ")
pEI = linregress(np.log(np.array(H)), np.log(np.array(eEI)))
print("L'ordre de convergence d'euler implicite est : ", pEI)
print("  ")
pES = linregress(np.log(np.array(H)), np.log(np.array(eES)))
print("L'ordre de convergence d'euler symplectique est : ", pES)
print("  ")
pRK = linregress(np.log(np.array(H)), np.log(np.array(eRK)))
print("L'ordre de convergence de runge Kutta 4 est : ", pRK)
