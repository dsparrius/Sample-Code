import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *;evalf

#Implementation of Eulers method for SDEs

#Uses sympy to create function notation for easier readability and coding. Converted to a numpy form to be faster.
x,t = symbols('x t')
f1 = 2*(t+1)**2
g1 = x**2*(1-t)
f = njit(lambdify([x,t],f1,'numpy'))
g = njit(lambdify([x,t],g1,'numpy'))

n = 2000
x0 = 1
repetitions = 4

def Euler(n,x0,f,g):
    h = 2/n
    X = np.zeros(n+1)
    X[0]=x0

    for j in range(1,n+1):

        #Models a case where equation starts deterministic
        if j < 500:
            X[j] = X[j-1] + f(X[j-1],j*h)*h
        #at time t=0.5, a random collision pushes solution off of original deterministic path
        if j == 500:
            X[j] = X[j-1] + f(X[j-1],j*h)*h + 100*np.random.normal(0,h**0.5)
        #after time t=0.5, random oscilations occur based off initial hit at t=0.5 and decay with time until t=1
        if 500<j < 1000:
            X[j] = X[j-1] + f(X[j-1],j*h)*h + g(X[500]-X[499],j*h)*np.random.normal(0,h**0.5)
        #after time t=1, solution returns to deterministic case following path it lands on during random process
        if j >= 1000:
            X[j] = X[j-1] + f(X[j-1],j*h)*h

    return X

#plots 4 repetitions of the approximation to see multiple possible sample paths
for k in range(repetitions):
    xstep= Euler(n,x0,f,g)
    df = pd.DataFrame({'x':[i*(2/n) for i in range(n+1)],'y':xstep})
    plt.plot(df.x,df.y)
plt.show()