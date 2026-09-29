import numpy as np

# PARAMETRES

n = 5000                # nombre d'atomes
dt = 5e-14              # pas de temps
sigma = 4e-10           # distance interatomique
L = n*sigma             # longueur de la chaîne
kB = 1.38e-23
epsilon = 400           # 400 K

k = 72*epsilon*kB/(sigma**2)   # constante de raideur

M = 57e-3
N_Avogadro = 6.022e23
m = M / N_Avogadro      # masse de l'atome en kg => 9.5e-26 kg

v_moy = 209             # vitesse moyenne des particules en m/s


# FONCTIONS

def calcul_accelerations(A, X, k, m, sigma):
    '''On fait le calcul à partir du bilan des forces sur la chaîne d'atomes.'''
    A[0] = k/m * (X[1] - X[0] - sigma)              # a_0 = k/m * (x_1 - x_0 - sigma)
    A[1:-1] = k/m * (X[2:] - 2*X[1:-1] + X[:-2])    # a_i = k/m * (x_(i+1) - 2x_i + x_(i-1))
    A[-1] = k/m * (X[-2] - X[-1] + sigma)           # a_(N-1) = k/m * (x_(N-2) - x_(N-1) + sigma)

    return A


# INITIALISATION DES TABLEAUX

N = np.arange(n)                                        # tableau index
X = np.arange(0,L,sigma)                                # positions des particules
V = np.random.normal(loc=0, scale=v_moy, size=n)        # vitesse des particules
A = calcul_accelerations(np.zeros(n), X, k, m, sigma)   # accélération des particules

# Matrice d'état initiale
C = np.stack((N,X,V,A))


# ALGORITHME DE VERLET VITESSE

V_ij = V + A * dt/2
X_n = X + V_ij * dt
A_n = calcul_accelerations(np.zeros(n), X_n, k, m, sigma)
V_n = V_ij + A_n * dt/2

# Matrice d'état à t + dt
C_n = np.stack((N, X_n, V_n, A_n))

print("Etat initial C :")
print(C)
print("\nEtat après 1 pas de l'algorithme")
print(C_n)