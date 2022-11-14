# -*- coding: utf-8 -*-
"""
 Modèle Proies/Prédacteurs de Lotka-Volterra
"""

_author__ = "JC"
__copyright__ = "Copyright 2021 -Jean-Claude,Bienvenue,Gilchrist"
__version__ = "1.0"
__date__ = "18 Novembre 2021"
__email__ = "blabla"

from numpy import array, linspace
from scipy import integrate
from matplotlib.pylab import figure, plot, grid, xlabel, ylabel, title, xlim, ylim, meshgrid,quiver, show

# définition des paramètres du modèle
k = 1.
x0 = 5  # population initiale des proies
y0 = 2  # population initiale des prédateurs
X0 = array([x0, y0])



# définition du système différentiel
def f(X, t=0):
    return array([X[0] - gamma*X[0]**2 - X[0] * X[1],
                  k * (X[0] * X[1] - X[1])])


# définition des variables
t0 = 0
T = 40
nbpoints = 1000
t = linspace(t0, T, nbpoints)




f2 = figure()
# tracé des différentes orbites
values = linspace(0.3, 0.9, 20)  # définition du point d'équilibre non trivial du système
for gamma in values:
    X_f1 = array([1, 1 - gamma])
    X0 = gamma * X_f1
    X = integrate.odeint(f, X0, t)
    plot(X[:, 0], X[:, 1], 'r-')


# tracé du champ de vecteurs
xmax = xlim(xmin=0)[1]
ymax = ylim(ymin=0)[1]
nb_points = 20
x = linspace(0, xmax, nb_points)
y = linspace(0, ymax, nb_points)
X1, Y1 = meshgrid(x, y)
DX1, DY1 = f([X1, Y1])
Q = quiver(X1, Y1, DX1, DY1)
title(u'Modèle de Lotka-Volterra -Extension: Quelques trajectoires dans l\'espace de phase')
xlabel(u'Proies')
ylabel(u'Prédateurs')
grid()
show()