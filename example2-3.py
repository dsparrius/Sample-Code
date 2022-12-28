import numpy as np
from sympy import *
init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt


x,h = symbols('x h')
f,a,b = exp(x),1,0
soln = exp(x)-exp(1)
hvals = [.2,.1,.05,.01]

errors=[]
graphdata = []
finegrid = [j*.01 for j in range(101)]

for hval in hvals:
    n = int(1/hval)
    grid = [j*hval for j in range(n+1)]
    #print(grid)

    F = Matrix(list(
        map(lambda y: a if y==0 else(b if y == 1 else f.subs(x,y)),grid)
        ))
    #pprint(F)
    A = Matrix(
        n+1,n+1, lambda i,j: -2 if (i==j and 0<i<n) else(
            1 if (j==i+1 and i >= 1) or (j==i-1 and 1<=i<n) else(
                -3*hval/2 if (i==0 and j== 0) else(
                    2*hval if (i==0 and j==1) else(
                        -hval/2 if (i==0 and j==2) else(
                            hval**2 if(i==n and j==n) else 0
                            ))))))
    #pprint(A)
    U = hval**2*A.inv()*F
    #pprint(U)
    approxsoln = list(map(lambda i:U[i],range(n+1)))
    #print(approxsoln)
    
    S = Matrix([soln.subs(x,_) for _ in grid])
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