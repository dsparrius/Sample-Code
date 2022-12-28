import matplotlib
import numpy as np
import sympy
from sympy import *
init_printing(use_unicode=True)
import fdcoeffV
import tabulate
import pandas as pd
import matplotlib.pyplot as plt


x,h = symbols('x h')
f = sin(2*x)
D2f = diff(diff(f,x),x)

coeffs = fdcoeffV.fdsymcoeffV(2,[-2,-1,0,1,2])

hvals = np.logspace(-1,-4,13)

column_names = ['h','error']

data = []
errors = []

terms = Matrix([[f.subs(x,1-2*h),f.subs(x,1-h),f.subs(x,1),f.subs(x,1+h),f.subs(x,1+2*h)]])
linearcomp = terms*coeffs

for hval in hvals:
    error = D2f.subs(x,1)-linearcomp[0].subs(h,hval)
    data.append([hval,N(error)])
    errors.append(float(N(error)))

print(tabulate.tabulate(data, headers = column_names))

df = pd.DataFrame({'x':[hval for hval in hvals],'y':[np.absolute(error) for error in errors]})
logx,logy = np.log10(df.x),np.log10(df.y)

plt.plot(logx, logy)
plt.show()