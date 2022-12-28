import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd


n=2**12
h1=1/n

#Generates change in Brownian Motion over the given time subintervals in the interval [0,1]
def Brownian(n):
    B = np.zeros(n)

    for j in range(n):
        B[j] = np.random.normal(0,h1**0.5)

    return B

B = Brownian(n)

#Creates list to store location of Brownian motion starting at 0
B1 =[0]

#Expands list to include location of Brownian motion after each time subinterval
for i in range(1,n+1):
    B1.append(B1[i-1]+B[i-1])

#Plots graph of Brownian Motion
df =pd.DataFrame({'x':[j*h1 for j in range(n+1)],'y':B1})
plt.plot(df.x,df.y)
plt.show()