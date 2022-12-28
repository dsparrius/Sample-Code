import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd


n=2**8
m=2**10
h1=1/n
h2=1/m

def Brownian(n,m):
    A = np.zeros(m)
    C = np.zeros(m)
    for i in range(m):
        r = np.random.normal(0,1/(m**0.5))
        A[i] = r
        C[i] = r

    B = np.zeros(n)
    k = int(m/n)
    C.reshape(n,k)

    for j in range(n):
        B[j] = np.sum(C[j])

    return A,B

A,B = Brownian(n,m)
A1 =[0]
k = int(m/n)

for i in range(1,m+1):
    A1.append(A1[i-1]+A[i-1])

B1 = [A1[k*j] for j in range(n+1)]

df =pd.DataFrame({'x':[j*h1 for j in range(n+1)],'y':B1})
dg = pd.DataFrame({'x':[j*h2 for j in range(m+1)],'y':A1})
plt.plot(df.x,df.y,label = "64 steps")
plt.plot(dg.x,dg.y,'r-',label = "256 steps")
plt.legend()
plt.show()