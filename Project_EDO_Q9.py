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


def Lyap_fonct(X):
    return X[0, :] + X[1, :] - np.log(X[0, :] * X[1, :])


"""
        Question 9:Pour chacune des méthodes implémentées dans la question 6), représenter l’évolution
                    de l’hamiltonien approché L(un; vn) par chaque méthode
"""

plt.figure()
#----------Evolution de l'hamiltonien par la méthode d'Euler Explicite---------------

YEE, time = EulerExplicite(f, 40, 20000, X0)
Ham_EE = Lyap_fonct(YEE)

plt.subplot(2, 2, 1)
plt.plot(Ham_EE, time, '-r')
plt.grid()
plt.title("Hamiltonien approché par la méthode d'Euler Explicite")



# #----------Evolution de l'hamiltonien par la méthode d'Euler Implicite---------------


YEI, time = EulerImplicite(f, 40, 20000, X0)
Ham_EI = Lyap_fonct(YEI)

plt.subplot(2, 2, 2)
plt.plot(Ham_EI, time, '-b')
plt.grid()
plt.title("Hamiltonien approché par la méthode d'Euler Implicite")


#----------Evolution de l'hamiltonien par la méthode de Runge Kutta_4---------------

YRK4, time = RungeKutta_4(f, 40, 20000, X0)
Ham_RK4 = Lyap_fonct(YRK4)

plt.subplot(2, 2, 3)
plt.plot(Ham_RK4, time, '-y')
plt.grid()
plt.title("Hamiltonien approché par la méthode de Runge Kutta")

#----------Evolution de l'hamiltonien par la méthode d'Euler Simplectique---------------


YES, time = EulerSimplectique(f, 40, 20000, X0)
Ham_ES = Lyap_fonct(YES)

plt.subplot(2, 2, 4)
plt.plot(Ham_ES, time, '-g')
plt.grid()
plt.title("Hamiltonien approché par la méthode d'Euler Simplectique")

plt.show()

