import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *;evalf

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
        if j < 500:
            X[j] = X[j-1] + f(X[j-1],j*h)*h
        if j == 500:
            X[j] = X[j-1] + f(X[j-1],j*h)*h + 100*np.random.normal(0,h**0.5)
        if 500<j < 1000:
            X[j] = X[j-1] + f(X[j-1],j*h)*h + g(X[500]-X[499],j*h)*np.random.normal(0,h**0.5)
        if j >= 1000:
            X[j] = X[j-1] + f(X[j-1],j*h)*h

    return X

for k in range(repetitions):
    xstep= Euler(n,x0,f,g)
    df = pd.DataFrame({'x':[i*(2/n) for i in range(n+1)],'y':xstep})
    plt.plot(df.x,df.y)
plt.show()