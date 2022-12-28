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
f = lambdify([x,t],7*cos(2*pi*t), 'numpy')
g = lambdify([x,t],0.5*exp(t-x), 'numpy')
dg = lambdify([x,t],diff(0.5*exp(t-x),x), 'numpy')
x0 = 0

#Implements Milstein's method
def Milstein(n,f,g,dg,x0):
    h = 1/n
    X = np.array([i*h for i in range(n)])
    Y = np.zeros(n+1)
    Y[0] = x0
    for j in range(1,n+1):
        w = np.random.normal(0,h**0.5)
        Y[j] = Y[j-1] + f(Y[j-1],j*h)*h + g(Y[j-1],j*h)*w + 0.5*g(Y[j-1],j*h)*dg(Y[j-1],j*h)*(w**2-h)
    return X,Y

start = time.time()
#runs Milstein's method multiple times and plots each occurance
for k in range(repetitions):
    xstep,ystep = Milstein(n,f,g,dg,x0)
    df = pd.DataFrame({'x':[i*(1/n) for i in range(n+1)],'y':ystep})
    plt.plot(df.x,df.y)
end = time.time()
print(ystep[n-3:n])
print(end-start)
plt.show()