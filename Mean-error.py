import numpy as np
from numba import njit,prange
import matplotlib.pyplot as plt
import pandas as pd
import time
from sympy import *

x,t = symbols('x t')
f = njit(lambdify(x,x**(1/3)/3+6*x**(2/3), 'numpy'))
g = njit(lambdify(x,x**(2/3), 'numpy'))
dg = njit(lambdify(x,diff(x**(2/3),x), 'numpy'))

sample = 400000
repetition= 1
x0 = 1

@njit(parallel=True)
def MeanError(n,f,g,dg,x0,sample):
    h = 1/n
    X,Y = np.zeros(n+1),np.zeros(n+1)
    X[0] = x0
    Y[0] = x0
    E1,E2 = np.zeros(sample),np.zeros(sample)
    for i in prange(sample):
        for j in prange(1,n+1):
            w = np.random.normal(0,h**0.5)
            X[j] = X[j-1]+f(X[j-1])*h+g(X[j-1])*np.random.normal(0,h**0.5)
            Y[j] = Y[j-1]+f(Y[j-1])*h+g(Y[j-1])*w + 0.5*g(Y[j-1])*dg(Y[j-1])*(w**2-h)
        E1[i],E2[i] = X[n],Y[n]
    return np.absolute(E1.sum()/sample - 28),np.absolute(E2.sum()/sample - 28)

@njit
def Error_analysis(exponents,size):
    errorE,errorM = np.zeros(size),np.zeros(size)
    j=0
    for k in range(repetition):
        for r in exponents:
            n = 2**r
            errorE[j],errorM[j]=MeanError(n,f,g,dg,x0,sample)
            j+=1
    return errorE,errorM

start = time.time()
errorx = np.array(repetition*[-1*r for r in range(5,10)])
error1,error2 = Error_analysis(np.array(range(5,10)),repetition*len(range(5,10)))

plt.plot(errorx,np.log2(error1),'o')
plt.plot(errorx,np.log2(error2),'o')

m1,b1 = np.polyfit(errorx,np.log2(error1),1)
m2,b2 = np.polyfit(errorx,np.log2(error2),1)

plt.plot(errorx,m1*errorx+b1,label = 'Euler')
plt.plot(errorx,m2*errorx+b2,label = 'Milstein')
plt.legend() 
plt.title(f'Sample size of {sample}')
end = time.time()
print(end-start)
plt.show()  