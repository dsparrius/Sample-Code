import numpy as np
from sympy import *
init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt

x,h = symbols('x h')
f = x
a,b=0,1
alph,bet = 0,1

hvals = [.1,.05,.01]

hval = .01
n = int(1/hval)
grid = [j*hval for j in range(n+1)]

F = Matrix(list(map(lambda y: alph if y==0 else(bet if y == 1 else f.subs(x,y)),grid)))

A = Matrix(n+1 ,n+1, lambda i,j: 1 if (i==0 and j==0) else (
    -2 if (j==i and 0<i<n) else(
        1 if (j==i-1 and 0<i<n) else(
            1 if (j==i+1 and 0<i<n) else(
                -hval if (j==n-1 and i==n) else(
                    hval if (j==n and i==n) else(0)
            )
        )
    )
)))

U = hval**2*A.inv()*F
approxsoln = list(map(lambda i:U[i],range(n+1)))

df = pd.DataFrame({'x':[_ for _ in grid],'y':[_ for _ in approxsoln]})
dg = pd.DataFrame({'x':[_ for _ in grid],'y':[_**3/6+_/2 for _ in grid]})

plt.plot(df.x,df.y)
plt.plot(dg.x,dg.y)
plt.show()