import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize



""" Test pour N = 100000 et comparaison avec la solution donnée par odeint"""
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
    return Y, time


def EulerImplicite(f, T, N, X0):
    time = [0]
    h = T / N
    Y = np.zeros((2, N))
    Y[:, 0] = X0
    t0 = 0

    def g(X):
        return X - h * f(X, t0) - X0

    for k in range(N - 1):
        X0 = optimize.fsolve(g, X0)
        t0 = t0 + h
        Y[:, k + 1] = X0
        time.append(t0)
    return Y, time


def EulerSimplectique(f, T, N, X0):
    YEE3, time = EulerExplicite(f, T, N, X0)
    h = T/N
    Y2 = [X0[1]]
    t0 = 0
    for k in range(N-1):
        X0[1] = X0[1] - h * f([YEE3[0, :][k+1], X0[1]], t0)[1]
        Y2.append(X0[1])


    YES = np.zeros((2, N))
    YES[0, :] = YEE3[0, :]
    YES[1, :] = Y2
    return YES, time


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
    return Y, time


# définition des variables
t0_odint = 0
T_odint = 40
nbpoints = 20000
t_odint = np.linspace(t0_odint, T_odint, nbpoints)

# calcul de la dynamique de la population
X = integrate.odeint(f, X0, t_odint)
proies, predateurs = X.T

"""
----------------------------------Méthode d'Euler Explicite/ N = 100000----------------------------------------
"""

# Solution de l'edo approchée par la méthode d'Euler Explicite pour N = 100000

Y2EE, time = EulerExplicite(f, 40, 100000, X0)
plt.figure()
plt.plot(time, Y2EE[0, :], 'b--', label=u'$EE_{Proies}$')
plt.plot(time, Y2EE[1, :], 'g--', label=u'$EE_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 1:Schéma d'Euler Explicite/ N = 100000")

"""
--------------------------------Méthode d'Euler Implicite/ N = 100000-------------------------------------------
"""

# Solution de approchée par la méthode d'Euler Implicite pour N = 100000

Y2EI, time = EulerImplicite(f, 40, 100000, X0)
plt.figure()
plt.plot(time, Y2EI[0, :], 'r-.', label=u'$EI_{Proies}$')
plt.plot(time, Y2EI[1, :], 'y-.', label=u'$EI_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 2:Schéma d'Euler Implicite/ N = 100000")


"""
--------------------------------Méthode de Runge Kutta/ N = 100000-------------------------------------------------
"""
# Solution de l'edo approchée par la méthode de Runge Kutta pour N = 100000

Y2RK, time = RungeKutta_4(f, 40, 100000, X0)
plt.figure()
plt.plot(time, Y2RK[0, :], 'r-', label=u'$RK4_{Proies}$')
plt.plot(time, Y2RK[1, :], 'y-', label=u'$RK4_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 4:Schéma de Runge Kutta_4/ N = 100000")


"""
--------------------------------Méthode d'Euler Simplectique/ N = 100000-------------------------------------------
"""


# Solution de l'edo approchée par la méthode d'Euler Simplectique pour N = 100000


Y1ES, time = EulerSimplectique(f, 40, 100000, X0)
plt.figure()
plt.plot(time, Y1ES[0, :], color='black', label=u'$ES_{Proies}$')
plt.plot(time, Y1ES[1, :], color='purple', label=u'$ES_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 3:Schéma d'Euler Simplectique/ N = 100000")


"""
--------------------------Comparaison avec la solution donnée par odeint: N = 100000-------------------------------
"""

plt.figure()
plt.plot(t_odint, proies, 'b-', label=u'$odeint_{Proies}$')
plt.plot(time, Y2EE[0, :], 'b--', label=u'$EE_{Proies}$')
plt.plot(time, Y2EI[0, :], 'r-.', label=u'$EI_{Proies}$')
plt.plot(time, Y1ES[0, :], color='black', label=u'$ES_{Proies}$')
plt.plot(time, Y2RK[0, :], 'r-', label=u'$RK4_{Proies}$')

plt.plot(t_odint, predateurs, 'g-', label=u'$odeint_{Prédateurs}$')
plt.plot(time, Y2EE[1, :], 'g--', label=u'$EE_{Prédacteurs}$')
plt.plot(time, Y2EI[1, :], 'y-.', label=u'$EI_{Prédacteurs}$')
plt.plot(time, Y1ES[1, :], color='purple', label=u'$ES_{Prédacteurs}$')
plt.plot(time, Y2RK[1, :], 'y-', label=u'$RK4_{Prédacteurs}$')
plt.title("Comparaison odeint_EE_EI_ES_RK4/ N = 100000")
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.show()
