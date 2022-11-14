# # -*- coding: utf-8 -*-

__author__ = "Jean-Claude,Bienvenue,Gilchrist"
__copyright__ = "Copyright @2021 Jean-Claude,Bienvenue,Gilchrist"
__version__ = "1.0"
__date__ = "18 Novembre 2021"
__email__ = "jeanclaudemitchozounou018@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize


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


"""
        Question 7: Traçons les solutions approchées dans le plan (u; v) pour chaque méthode employée cidessus puis 
                    comparer avec la solution donnée par odeint obtenue dans la question 5).Vérifions une nouvelle fois
                     le caractère périodique de la solution. Traçons le champ desvitesses de phase, utiliser la fonction
                      matplotlib.quiver.
"""

# définition des variables
t0_odint = 0
T_odint = 40
nbpoints = 20000
t_odint = np.linspace(t0_odint, T_odint, nbpoints)

# calcul de la dynamique de la population
X = integrate.odeint(f, X0, t_odint)
proies, predateurs = X.T

# tracé dans le plan (u,v) de  la dynamique de la population
plt.figure()
plt.plot(proies, predateurs, 'b-')
plt.grid()
plt.xlabel('$_{odeint}$')
plt.ylabel('$Prédacteurs_{odeint}$')
plt.title('Modèle Proies/Prédacteurs de Lotka-Volterra ')




# Solution de l'edo dans le plan (u,v)pour la méthode d'Euler Explicite pour N = 20000

Y1EE, time = EulerExplicite(f, 40, nbpoints, X0)
plt.figure()
plt.plot(Y1EE[0, :], Y1EE[1, :], 'g-')
plt.plot(Y1EE[0, 0], Y1EE[1, 0], 'r*', label=u'Depart')
plt.plot(Y1EE[0, nbpoints-1], Y1EE[1, nbpoints-1], 'r^', label=u'Arrivée')
plt.grid()
plt.legend()
plt.xlabel('$Proies_{EE}$')
plt.ylabel('$Prédacteurs_{EE}$')
plt.title("Cas 1:Euler Explicite dans le plan (u,v)/ N = 20000")




# Solution de l'edo dans le plan (u,v) pour la méthode d'Euler Implicite pour N = 20000

Y1EI, time = EulerImplicite(f, 40, nbpoints, X0)
plt.figure()
plt.plot(Y1EI[0, :], Y1EI[1, :], 'y-')
plt.plot(Y1EE[0, 0], Y1EI[1, 0], 'b*', label=u'Depart')
plt.plot(Y1EE[0, nbpoints-1], Y1EI[1, nbpoints-1], 'b^', label=u'Arrivée')
plt.grid()
plt.legend()
plt.xlabel('$Proies_{EI}$')
plt.ylabel('$Prédacteurs_{EI}$')
plt.title("Cas 2:Euler Implicite dans le plan (u,v)/ N = 20000")


# Solution de l'edo dans le plan (u,v) pour la méthode Runge Kutta pour N = 20000

Y1RK, time = RungeKutta_4(f, 40, nbpoints, X0)
plt.figure()
plt.plot(Y1RK[0, :], Y1RK[1, :], 'r-')
plt.plot(Y1RK[0, 0], Y1RK[1, 0], 'g+', label=u'Depart')
plt.plot(Y1RK[0, nbpoints-1], Y1RK[1, nbpoints-1], 'g*', label=u'Arrivée')
plt.grid()
plt.legend()
plt.xlabel('$Proies_{RK4}$')
plt.ylabel('$Prédacteurs_{RK4}$')
plt.title("Cas 4:Runge Kutta_4 dans le plan (u,v)/ N = 20000")


# # Solution de l'edo dans le plan (u,v)pour la méthode d'Euler Simplectique pour N = 20000

Y1ES, time = EulerSimplectique(f, 40, nbpoints, X0)
plt.figure()
plt.plot(Y1ES[0, :], Y1ES[1, :], 'g-')
plt.plot(Y1ES[0, 0], Y1ES[1, 0], 'r*', label=u'Depart')
plt.plot(Y1ES[0, nbpoints-1], Y1ES[1, nbpoints-1], 'r^', label=u'Arrivée')
plt.grid()
plt.legend()
plt.xlabel('$Proies_{ES}$')
plt.ylabel('$Prédacteurs_{ES}$')
plt.title("Cas 3:Euler Simplectique dans le plan (u,v)/ N = 20000")




"""
--------------------------Comparaison avec la solution donnée par odeint: N = 20000-------------------------------
"""

plt.figure()
plt.plot(proies,predateurs, 'b-', label=u'odeint')
plt.plot(Y1EE[0, :], Y1EE[1, :], 'g-', label=u'EE')
plt.plot(Y1EI[0, :], Y1EI[1, :], 'y-', label=u'EI')
plt.plot(Y1ES[0, :], Y1ES[1, :], color='black', label=u'ES')
plt.plot(Y1RK[0, :], Y1RK[1, :], 'r-', label=u'RK4')
plt.grid()
plt.legend()
plt.xlabel('$Proies_{odeint-EE-EI-ES_RK4}$')
plt.ylabel('$Prédacteurs_{odeint-EE-EI-Es_RK4}$')
plt.title("Comparaison à odeint dans le plan (u,v) des EE-EI-RK/ N = 20000")


"""
---------------------------------------Tracé du champ de vecteur--------------------------------------------------------
"""

plt.figure()
plt.plot(proies, predateurs, 'b-')
xmax =plt.xlim(xmin=0)[1]
ymax = plt.ylim(ymin=0)[1]
nb_points = 40
x = np.linspace(0, xmax, nb_points)
y = np.linspace(0, ymax, nb_points)
X1, Y1 = np.meshgrid(x, y)
DX1, DY1 = f([X1, Y1])
Q = plt.quiver(X1, Y1, DX1, DY1)
plt.title(u'Modèle de Lotka-Volterra - Portrait de phase/Champ de vecteurs vitesses')
plt.xlabel(u'Proies')
plt.ylabel(u'Prédateurs')
plt.grid()
plt.show()


