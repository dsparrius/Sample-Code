from sympy import *
import sympy
init_printing(use_unicode=True)

#Finds coeffs for kth derivative
#xbar is the point of interest for the approximation
#x is a list of grid points. xbar need not be in x
def fdcoeffV(k,xbar,x):
    n = len(x)

    if k>= n:
        raise Exception('k must be small than the length of x')
    
    diff = [y - xbar for y in x]
    rows = []

    i = 0
    while i < n:
        row = []
        for h in diff:
            row.append(h**i/sympy.factorial(i))
        
        rows.append(row)
        i+=1

    A = Matrix(rows)
    R = Matrix(n,1, lambda i,j: 1 if i==k else 0)

    return A.solve(R,'GJ')

#requires a uniform step size h
#indices in j need not be uniformly spaced
#Finds coeffs for kth derivative approximation
def fdsymcoeffV(k,j):
    n = len(j)
    if k>= n:
        raise Exception('k must be small than the length of x')

    x,h = symbols('x h')

    hstep = [a*h for a in j]
    rows=[]

    i = 0
    while i < n:
        row = []
        for b in hstep:
            row.append(b**i/sympy.factorial(i))

        rows.append(row)
        i+=1

    A = Matrix(rows)
    R = Matrix(n,1, lambda i,m: 1 if i==k else 0)

    return A.solve(R,'GJ')
