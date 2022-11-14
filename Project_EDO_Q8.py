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
        Question 8: Tracer les isovaleurs de la fonction L (fonction matplotlib.pyplot.contour).
"""
plt.figure()
#                               Isovaleurs méthodes d'Euler Explicite

YEE, time = EulerExplicite(f, 40, 1000, X0)
[X, Y] = np.meshgrid(YEE[0, :], YEE[1, :])

Z = X + Y + np.log(np.fabs(X * Y))
plt.subplot(2, 2, 1)
plt.contour(X, Y, Z)
plt.title('Isovaleur Euler Explicite')
plt.xlabel('Proies')
plt.ylabel('Prédacteurs')

#                               Isovaleurs méthodes d'Euler Implicite

YEI, time1 = EulerImplicite(f, 40, 1000, X0)
[X, Y] = np.meshgrid(YEI[0, :], YEI[1, :])

Z = X + Y + np.log(X * Y)
plt.subplot(2, 2, 2)
plt.contour(X, Y, Z)
plt.title('Isovaleur Euler Implicite')
plt.xlabel('Proies')
plt.ylabel('Prédacteurs')


#                               Isovaleurs méthodes de Runge Kutta

YRK4, time3 = RungeKutta_4(f, 40, 1000, X0)
[X, Y] = np.meshgrid(YRK4[0, :], YRK4[1, :])

Z = X + Y + np.log(X * Y)
plt.subplot(2, 2, 3)
plt.contour(X, Y, Z)





plt.title('Isovaleur de Runge Kutta')
plt.xlabel('Proies')
plt.ylabel('Prédacteurs')

#                               Isovaleurs méthodes d'Euler Simplicite

YES, time2 = EulerSimplectique(f, 40, 1000, X0)
[X, Y] = np.meshgrid(YES[0, :], YES[1, :])

Z = X + Y + np.log(X * Y)
plt.subplot(2, 2, 4)
plt.contour(X, Y, Z)
plt.title('Isovaleur EulerSimplicite')
plt.xlabel('Proies')
plt.ylabel('Prédacteurs')

plt.show()
w