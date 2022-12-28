import numpy as np
from sympy import *
init_printing(use_unicode=True)
import pandas as pd
import matplotlib.pyplot as plt

a,b,alph,bet = 0,1,-1.5,1.5
w = .5*(a-b+bet-alph)
xbar = .5*(a+b-alph-bet)
print(xbar)
x,h = symbols('x h')

epsilon = .05
hvals = [.1,.05,.01]
tilde_u = x-xbar+w*tanh(w*(x-xbar)/(2*epsilon))

hval = .01

n = int((b-a)/hval)
grid = [a + j*hval for j in range(n+1)]

U = Matrix(list(map(lambda y: alph if y == 0 else( bet if y==1 else tilde_u.subs(x,y)),grid)))
df = pd.DataFrame({'x':[_ for _ in grid],'y':[U[_] for _ in range(n+1)]})
plt.plot(df.x,df.y, label = '0')
k=0

while k<5:
    G = -1*Matrix(list(map(lambda i: epsilon*(U[i-1]-2*U[i]+U[i+1])/(hval**2) + U[i]*((U[i+1]-U[i-1])/(2*hval) - 1), range(1,n))))

    J = Matrix(n-1,n-1, lambda i,j:(-2*epsilon)/(hval**2) +(U[i+1]-U[i-1])/(2*hval) if i==j else(
    epsilon/(hval**2) + U[i]/(2*hval) if j==i+1 else(
        epsilon/(hval**2) - U[i]/(2*hval) if j==i-1 else 0
    )
    ))

    delta = hval**2*J.inv()*G
    newsoln = [alph]
    for m in range(1,n):
        newsoln.append(U[m]+delta[m-1])
    newsoln.append(bet)
    U = Matrix(newsoln)
    k+=1
    df = pd.DataFrame({'x':[_ for _ in grid],'y':[U[_] for _ in range(n+1)]})
    plt.plot(df.x,df.y, label = str(k))


dh = pd.DataFrame({'x': [_ for _ in grid],'y':[_+alph-a for _ in grid]})
dg = pd.DataFrame({'x': [_ for _ in grid],'y':[_+bet-b for _ in grid]})


plt.plot(dh.x,dh.y)
plt.plot(dg.x,dg.y)
plt.legend()
plt.show()