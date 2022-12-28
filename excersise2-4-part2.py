import numpy as np
from sympy import *
init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt

x,h = symbols('x h')
f = x
a,b=0,1
alph,bet = 0,1
soln = x**3/6+x/2

hvals = [.2,.1,.02,.01]
errors = []

for hval in hvals:
    n = int(1/hval)
    m = int(2/hval)
    grid = [j*hval for j in range(n+1)]
    grid2 = [j*hval/2 for j in range(m+1)]

    F = Matrix(list(map(lambda y: alph if y==0 else(bet if y == 1 else f.subs(x,y)),grid)))
    F2 = Matrix(list(map(lambda y: alph if y==0 else(bet if y == 1 else f.subs(x,y)),grid2)))

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

    A2 = Matrix(m+1 ,m+1, lambda i,j: 1 if (i==0 and j==0) else (
        -2 if (j==i and 0<i<m) else(
            1 if (j==i-1 and 0<i<m) else(
                1 if (j==i+1 and 0<i<m) else(
                    -hval/2 if (j==m-1 and i==m) else(
                        hval/2 if (j==m and i==m) else(0)
                )
            )
        )
    )))

    U = hval**2*A.inv()*F
    U2 = (hval/2)**2*A2.inv()*F2

    U3 = Matrix(list(map(lambda i:(4*U2[2*i] - U[i])/3,range(n+1))))
    S = Matrix([soln.evalf(subs={x:_}) for _ in grid])
    E = S-U3
    error = E.norm(oo)
    errors.append(error)

de = pd.DataFrame({'x': [np.log10(float(_)) for _ in hvals], 'y': [np.log10(float(_)) for _ in errors]})

plt.plot(de.x,de.y)
plt.show()