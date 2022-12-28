import numpy as np
import time
from sympy import *
init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt

start = time.time()
x = symbols('x')
T = 2*np.pi
a,b = 0.7,0.7
g = sin(x)

n=100
hval = T/n
grid = [j*hval for j in range(n+1)]
U = Matrix(list(map(lambda y: a if y==0 else(b if y==T else 0.7 - exp(-1/(y**2)-1/((T-y)**2))),grid)))

Delta = []
df = pd.DataFrame({'x':[_ for _ in grid],'y':[U[i] for i in range(n+1)]})
plt.plot(df.x,df.y, label = '0')

for k in range(6):
    G = -1*Matrix(list(map(lambda i: (1/hval)**2*(U[i-1]-2*U[i]+U[i+1]) + g.evalf(subs={x:U[i]}),range(1,n))))
    J = Matrix(n-1,n-1, lambda i,j: -2 + hval**2*cos(U[i+1]) if i==j else(1 if (j==i+1 or j==i-1) else 0))

    delta = hval**2*J.inv()*G
    Delta.append(delta.norm(oo))

    newsoln = [a]
    for m in range(1,n):
        newsoln.append(U[m]+delta[m-1])
    newsoln.append(b)
    U = Matrix(newsoln)
    df = pd.DataFrame({'x':[_ for _ in grid],'y':[U[i] for i in range(n+1)]})
    plt.plot(df.x,df.y, label = str(k+1))

print(Delta)
plt.legend()
end=time.time()
print(end-start)

plt.show()
