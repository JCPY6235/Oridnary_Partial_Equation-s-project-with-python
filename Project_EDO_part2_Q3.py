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


#definition de la fonction vectorielle
def f(X,t):
	u = X[0];
	v = X[1];   #X=(u,v)
	return np.array ([u*(1-v) , alpha*v*(u-1)])

def L(u,v):
	return u + v + np.log(u * v)

def eulerExplicite(f,X0,t):
	n = len(t)  #Longueur de la subdivision faite sur t. Dans notre cas ici, c'est N
	X = np.zeros((len(t),) + np.shape(X0)) #shape de numpy: affiche la taille de la matrice
	X[0] = X0
	h = (t[-1]-t[0])/n
	for i in range(n-1):
		X[i+1] = X[i] + h * f(X[i],t[i])
	return X

def eulerImplicite(f,X0,t):
	n = len(t)  #Longueur de la subdivision faite sur t. Dans notre cas ici, c'est N
	X = np.zeros((len(t),) + np.shape(X0)) #shape de numpy: affiche la taille de la matrice
	X[0] = X0
	h = (t[-1]-t[0])/n
	for i in range(n-1):
		# Equation à résoudre X[i+1] = X[i] + h * f(X[i+1], t[i+1])
		# X[i+1] = solution de l'éq : X[i+1] - X[i] - h * f(X[i+1], t[i+1]) = 0
		# Z - X[i] - h * f(Z,t[i+1]) = 0
		g = lambda Z : Z - X[i] - h * f(Z, t[i+1])
		pointinitial = X[i]
		X[i+1] = fsolve(g, pointinitial)
	return X

def eulerSymplectique(f,X0,t):
	n = len(t)  #Longueur de la subdivision faite sur t. Dans notre cas ici, c'est N
	X = np.zeros((len(t),) + np.shape(X0)) #shape de numpy: affiche la taille de la matrice
	X[0] = X0
	h = (t[-1]-t[0])/n
	for i in range(n-1):
		X[i+1,0] = X[i,0] + h * X[i,0]*(1 - X[i,1])
		X[i+1,1] = X[i,1] - h * alpha * X[i,1] * (X[i+1,0] - 1)
	return X

def RK4(f,X0,t):    #Runge Kutta d'ordre 4
	"""
	Entrees:
		t     : contient les points de la subdivision
		X0    : les conditions initiales
		f     : la fonction connue
	Sorties:
		X     : Solutions approchées aux points de t
	"""
	n = len(t)
	X = np.zeros((len(t),) + np.shape(X0)) #Construction d'une matrice de 0
	X[0] = X0 #Stockage des valeurs initiales
	h = (t[-1]-t[0])/n
	for i in range(n-1):
		k1 = h * f(X[i], t[i])
		k2 = h * f(X[i] + 0.5 * k1, t[i] + (0.5 * h))
		k3 = h * f(X[i] + 0.5 * k2, t[i] + (0.5 * h))
		k4 = h * f(X[i] + k3, t[i] + h)
		X[i+1]= X[i] + ((k1 + 2 * k2 + 2 * k3 + k4 )/ 6)
	return X



#QUESTION 10
"""	PRENONS COMME SOLUTION DE RÉFERENCE CELLE OBTENUE AVEC N = 20000 PAS DE DISCRÉTISATION,
	REPRÉSENTONS LES ERREURS AU TEMPS T=40 TROUVÉES EN UTILISANT SUCCESSIVEMENT 
	NN = [10000, 10200, 10400, 10600, 10800] PAS DE DISCRÉTISATION """

#definition des variables
alpha = 1
T = 40
ci= np.array([3,5])

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
"""


ee_r = eulerExplicite(f,ci,t1)[-1]
ei_r = eulerImplicite(f,ci,t1)[-1]
es_r = eulerSymplectique(f,ci,t1)[-1]
rk_r = RK4(f,ci,t1)[-1]

eEE, eEI, eES, eRK = [], [], [], []
H = []


for N in NN:
	tt = np.linspace(0, T, N)
	eEE.append(np.linalg.norm(ee_r - eulerExplicite(f, ci, tt)[-1]))
	eEI.append(np.linalg.norm(ei_r - eulerImplicite(f, ci, tt)[-1]))
	eES.append(np.linalg.norm(es_r - eulerSymplectique(f, ci, tt)[-1]))
	eRK.append(np.linalg.norm(rk_r - RK4(f, ci, tt)[-1]))
	H.append(40/(N+1))

plt.figure()
plt.plot(NN,eEE,'-x',label='EE')
plt.plot(NN,eEI,'-o',label='EI')
plt.plot(NN,eES,'-x',label='ES')
plt.plot(NN,eRK,'-o',label='RK')
plt.plot(NN,NN,label='O(N)')
plt.plot(NN,[n**2 for n in NN],label='O(N^3)')
plt.xlabel('N')
plt.ylabel('erreur temps final')
plt.title('erreurs avec EE, EI, ES, RK')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

pEE = linregress(np.log(np.array(H)),np.log(np.array(eEE)))
print("L'ordre de convergence d'euler explicite est : ", pEE)
print("  ")
pEI = linregress(np.log(np.array(H)),np.log(np.array(eEI)))
print("L'ordre de convergence d'euler implicite est : ", pEI)
print("  ")
pES = linregress(np.log(np.array(H)),np.log(np.array(eES)))
print("L'ordre de convergence d'euler symplectique est : ", pES)
print("  ")
pRK = linregress(np.log(np.array(H)),np.log(np.array(eRK)))
print("L'ordre de convergence de runge Kutta 4 est : ", pRK)

