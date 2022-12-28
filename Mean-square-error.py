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

sample = 10000
x0 = 1

@njit
def Brownian():
    A = np.zeros(2**15)
    for i in prange(2**15):
        r = np.random.normal(0,1/(2**(15/2)))
        A[i] = r

    return A

@njit(parallel=True)
def MeanSquareError(f,g,dg,x0,sample,exponents,length):
    E1,E2 = np.zeros((length,sample)),np.zeros((length,sample))
    errorE,errorM = np.zeros(length),np.zeros(length)

    for i in prange(sample):
        h2 = 1/(2**15)
        A = Brownian()
        A1 = np.zeros(2**15)
        A1[0] = 0

        for m in prange(1,2**15+1):
            A1[m] = A1[m-1]+A[m-1]

        for r in prange(length):
            n=2**exponents[r]
            h1=1/n
            k = int(2**15/n)
            B = np.zeros(n)

            for _ in prange(n):
                B[_] = A1[(_+1)*k] - A1[_*k]

            X,Z,Y = np.zeros(n+1),np.zeros(n+1),np.zeros(2**15+1)
            X[0] = x0
            Z[0] = x0
            Y[0] = x0

            for j in prange(1,n+1):
                X[j] = X[j-1] + f(X[j-1])*h1 + g(X[j-1])*B[j-1]
                Z[j] = Z[j-1] + f(Z[j-1])*h1 + g(Z[j-1])*B[j-1] + 0.5*g(Z[j-1])*dg(Z[j-1])*(B[j-1]**2-h1)

            for k in prange(1,2**15+1):
                Y[k] = Y[k-1] + f(Y[k-1])*h2 + g(Y[k-1])*A[k-1] + 0.5*g(Y[k-1])*dg(Y[k-1])*(A[k-1]**2-h2)

            E1[r,i] = (X[n]-Y[2**15])**2
            E2[r,i] = (Z[n]-Y[2**15])**2

    for r in prange(length):
        errorE[r],errorM[r] = E1[r].sum()/sample,E2[r].sum()/sample
    return errorE,errorM

start = time.time()
errorx = np.array([-1*r for r in range(9,15)])
xcopy = np.copy(errorx)
error1,error2 = MeanSquareError(f,g,dg,x0,sample,np.array(range(9,15)),len(range(9,15)))

plt.plot(errorx,np.log2(error1),'o')
plt.plot(errorx,np.log2(error2),'o')

m1,b1 = np.polyfit(errorx,np.log2(error1),1)
m2,b2 = np.polyfit(errorx,np.log2(error2),1)

plt.plot(errorx,m1*errorx+b1,label = 'Euler')
plt.plot(errorx,m2*errorx+b2,label = 'Milstein')
plt.legend() 
end = time.time()
print(end-start)
plt.show()     