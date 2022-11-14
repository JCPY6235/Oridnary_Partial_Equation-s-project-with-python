# # -*- coding: utf-8 -*-

__author__ = "Jean-Claude,Bienvenue,Gilchrist"
__copyright__ = "Copyright @2021 Jean-Claude,Bienvenue,Gilchrist"
__version__ = "1.0"
__date__ = "18 Novembre 2021"
__email__ = "jeanclaudemitchozounou018@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

# Schéma numérique EulerExplicite: Y_{n+1} = Y_n + hf(Y_n, t_n)

# Implémentation

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

# Schéma numérique EulerImplicite: Y_{n+1} = Y_n + hf(Y_{n+1}, t_n)

# Implémentation


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


# Schéma numérique RungeKutta_4: Yn+1 = Yn + (h/6)*(k1 + 2k2 + 2k3 + k4)
#                   k1 = f(tn, Yn)
#                   k2 = f(tn + h/2, Yn + h/2 k1)
#                   k3 = f(tn + h/2, Yn + h/2 k2)
#                   k4 = f(tn + h, Yn + hk3)

# Implémentation

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

#Schéma numérique EulerSimplectique

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


# définition des paramètres du modèle
k = 1.

x0 = 3  # population initiale des proies
y0 = 5  # population initiale des prédateurs
X0 = np.array([x0, y0])


# définition du système différentiel
def f(X, t=0):
    return np.array([X[0] - X[0] * X[1],
                     k * (X[0] * X[1] - X[1])])


"""
    Question 5: Tracé de la solution du problèmede Cauchy (2-3)
         par la fonction scipy.integrate.odeint

"""

# définition des variables
t0_odint = 0
T_odint = 40
nbpoints = 20000
t_odint = np.linspace(t0_odint, T_odint, nbpoints)

# calcul de la dynamique de la population
X = integrate.odeint(f, X0, t_odint)
proies, predateurs = X.T

# tracé de la dynamique de la population
fig = plt.figure()
plt.plot(t_odint, proies, 'b-', label=u'$odeint_{Proies}$')
plt.plot(t_odint, predateurs, 'g-', label=u'$odeint_{Prédateurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title('Modèle Proies/Prédacteurs de Lotka-Volterra\n'+
            'solution du problèmede Cauchy (2-3) avec\n'+
                 'la fonction scipy.integrate.odeint')


"""
    Question 6: Implémenter la méthode d’Euler explicite, la méthode d’Euler implicite, la méthode
                d’Euler symplectique et la méthode de Runge-Kutta d’ordre 4. On pourra prendre α = 1;
                T = 40; N = 20000, puis tester avec N = 100000. Comparer avec la solution donnée par odeint

"""

"""
----------------------------------Méthode d'Euler Explicite/ N = 20000----------------------------------------
"""

# Solution de l'edo approchée par la méthode d'Euler Explicite pour N = 20000

Y1EE, time = EulerExplicite(f, 40, 20000, X0)
plt.figure()
plt.plot(time, Y1EE[0, :], 'b--', label=u'$EE_{Proies}$')
plt.plot(time, Y1EE[1, :], 'g--', label=u'$EE_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 1:Schéma d'Euler Explicite/ N = 20000")

"""
--------------------------------Méthode d'Euler Implicite/ N = 20000-------------------------------------------
"""

# Solution de l'edo approchée par la méthode d'Euler Implicite pour N = 20000

Y1EI, time = EulerImplicite(f, 40, 20000, X0)
plt.figure()
plt.plot(time, Y1EI[0, :], 'r-.', label=u'$EI_{Proies}$')
plt.plot(time, Y1EI[1, :], 'y-.', label=u'$EI_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 2:Schéma d'Euler Implicite/ N = 20000")



"""
----------------------------------Méthode de Runge Kutta/ N = 20000----------------------------------------
"""


# Solution de l'edo approchée par la méthode de Runge Kutta pour N = 20000

Y1RK, time = RungeKutta_4(f, 40, 20000, X0)
plt.figure()
plt.plot(time, Y1RK[0, :], 'r-', label=u'$EI_{Proies}$')
plt.plot(time, Y1RK[1, :], 'y-', label=u'$EI_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 3:Schéma de Runge Kutta_4/ N = 20000")


"""
----------------------------------Méthode d'Euler simplectique/ N = 20000----------------------------------------
"""

# Solution de l'edo approchée par la méthode d'Euler Simplectique pour N = 20000

Y1ES, time = EulerSimplectique(f, 40, 20000, X0)
plt.figure()
plt.plot(time, Y1ES[0, :], color='black', label=u'$ES_{Proies}$')
plt.plot(time, Y1ES[1, :], color='purple', label=u'$ES_{Prédacteurs}$')
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title("Cas 3:Schéma d'Euler Simplectique/ N = 20000")


"""
--------------------------Comparaison avec la solution donnée par odeint: N = 20000-------------------------------
"""

plt.figure()
plt.plot(t_odint, proies, 'b-', label=u'$odeint_{Proies}$')
plt.plot(time, Y1EE[0, :], 'b--', label=u'$EE_{Proies}$')
plt.plot(time, Y1EI[0, :], 'r-.', label=u'$EI_{Proies}$')
plt.plot(time, Y1ES[0, :], color='black', label=u'$ES_{Proies}$')
plt.plot(time, Y1RK[0, :], 'r-', label=u'$RK4_{Proies}$')


plt.plot(t_odint, predateurs, 'g-', label=u'$odeint_{Prédateurs}$')
plt.plot(time, Y1EE[1, :], 'g--', label=u'$EE_{Prédacteurs}$')
plt.plot(time, Y1EI[1, :], 'y-.', label=u'$EI_{Prédacteurs}$')
plt.plot(time, Y1ES[1, :], color='purple', label=u'$ES_{Prédacteurs}$')
plt.plot(time, Y1RK[1, :], 'y-', label=u'$RK4_{Prédacteurs}$')
plt.title("Comparaison odeint_EE_EI_ES_RK/ N = 20000")
plt.grid()
plt.legend()
plt.xlabel('Temps')
plt.ylabel('Population')

plt.show()

