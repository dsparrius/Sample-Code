import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *

#This program studies an SDE whose underlying deterministic equation exhibits stability

#sample size used to find average error
sample = 10000
#step size
n = 1000
#repeats approximation to see variance in plots
repetitions = 4

#uses sympy to create function notation for easies code to read
x,t = symbols('x t')
#deterministic term
f = lambdify([x,t],2*sin(pi*(x+t)), 'numpy')
#random term that rapidly decays
g = lambdify([x,t],0.5*exp(-1/(1.2-t)**2), 'numpy')
x0 = 0

#Implements Euler's method
def Euler(n,f,g,x0):
    h = 1/n
    X = np.array([i*h for i in range(n)])
    Y = np.zeros(n+1)
    Y[0] = x0
    for j in range(1,n+1):
        w = np.random.normal(0,h**0.5)
        Y[j] = Y[j-1] + f(Y[j-1],j*h)*h + g(Y[j-1],j*h)*w
    return X,Y

start = time.time()
#runs Euler's method multiple times and plots each occurance
for k in range(repetitions):
    xstep,ystep = Euler(n,f,g,x0)
    df = pd.DataFrame({'x':[i*(1/n) for i in range(n+1)],'y':ystep})
    plt.plot(df.x,df.y)
end = time.time()
print(ystep[n-3:n])
print(end-start)
plt.show()