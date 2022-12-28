import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *

sample = 10000
n = 1000
repetitions = 4

x,t = symbols('x t')
f = lambdify([x,t],2*sin(np.pi*(x+t)), 'numpy')

def Milstein(n,f,x0):
    h = 1/n
    X = np.array([i*h for i in range(n)])
    Y = np.zeros(n+1)
    Y[0] = x0
    for j in range(1,n+1):
        w = np.random.normal(0,h**0.5)
        Y[j] = Y[j-1] + f(Y[j-1],j*h)*h
    return X,Y

start = time.time()
for x0 in [0,1,0.5,1.5,7/6]:
    xstep,ystep = Milstein(n,f,x0)
    df = pd.DataFrame({'x':[i*(1/n) for i in range(n+1)],'y':ystep})
    plt.plot(df.x,df.y)
end = time.time()
#dg = pd.DataFrame({'x':[i*(1/n) for i in range(n+1)],'y':[7/6-i*(1/n) for i in range(n+1)]})
#plt.plot(dg.x,dg.y)
print(ystep[n-3:n])
print(end-start)
plt.show()