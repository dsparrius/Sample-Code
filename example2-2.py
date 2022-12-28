import numpy as np
from sympy import *
init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt


x,h = symbols('x h')
f,a,b = cos(x),0,0
soln = -cos(x) + (cos(1) - 1)*x + 1
hvals = [.5,.2,.1,.05,.01]

errors=[]
graphdata = []
finegrid = [j*.01 for j in range(101)]

for hval in hvals:
    n = int(1/hval)
    grid = [j*hval for j in range(n+1)]

    F = Matrix(list(
        map(lambda y: f.subs(x,y)-a/(hval**2) if y==0 else(f.subs(x,y) - b/(hval**2) if y == 1 else f.subs(x,y)),grid[1:n])
        ))
    A = Matrix(n-1,n-1, lambda i,j: -2 if i==j else(1 if j==i+1 or j==i-1 else 0))

    U = ((1/hval**2)*A).inv()*F
    approxsoln = list(map(lambda i: a if i==0 else(b if i==n else U[i-1]),range(n+1)))
    
    S = Matrix([soln.subs(x,_) for _ in grid[1:n]])
    E = S-U

    error = E.norm(oo)
    errors.append(error)
    
    if hval == .1:
        graphdata.append([grid,approxsoln])


df = pd.DataFrame({'x':[_ for _ in graphdata[0][0]],'y':[_ for _ in graphdata[0][1]]})
dh = pd.DataFrame({'x':[_ for _ in finegrid],'y':[soln.subs(x,_) for _ in finegrid]})
de = pd.DataFrame({'x': [np.log10(float(_)) for _ in hvals], 'y': [np.log10(float(_)) for _ in errors]})

plt.plot(df.x,df.y)
plt.plot(dh.x,dh.y)
plt.show()
plt.plot(de.x,de.y)
plt.show()