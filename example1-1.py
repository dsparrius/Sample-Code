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
f = sin(x)
Df = diff(f,x)

coeffsplus = fdcoeffV.fdsymcoeffV(1,[0,1])
coeffsminus = fdcoeffV.fdsymcoeffV(1,[-1,0])
coeffs0 = fdcoeffV.fdsymcoeffV(1,[-1,1])
coeffs3 = fdcoeffV.fdsymcoeffV(1,[-2,-1,0,1])

termsplus = Matrix([[f.subs(x,1),f.subs(x,1+h)]])
termsminus = Matrix([[f.subs(x,1-h),f.subs(x,1)]])
terms0 = Matrix([[f.subs(x,1-h),f.subs(x,1+h)]])
terms3 = Matrix([[f.subs(x,1-2*h),f.subs(x,1-h),f.subs(x,1),f.subs(x,1+h)]])

linearcompplus = termsplus*coeffsplus
linearcompminus = termsminus*coeffsminus
linearcomp0 = terms0*coeffs0
linearcomp3 = terms3*coeffs3


hvals = [.1,.05,.01,.005,.001]

column_names = ['h','error plus', 'error minus', 'error 0', 'error 3']
data = []
errorsplus = []
errors0 = []
errors3 = []


for hval in hvals:
    errorplus = Df.subs(x,1) - linearcompplus[0].subs(h,hval)
    errorminus = Df.subs(x,1) - linearcompminus[0].subs(h,hval)
    error0 = Df.subs(x,1) - linearcomp0[0].subs(h,hval)
    error3 = Df.subs(x,1) - linearcomp3[0].subs(h,hval)

    data.append([hval,N(errorplus),N(errorminus),N(error0),N(error3)])
    errorsplus.append(float(N(errorplus)))
    errors0.append(float(N(error0)))
    errors3.append(float(N(error3)))

print(tabulate.tabulate(data, headers = column_names))

df = pd.DataFrame({'x':[hval for hval in hvals],'y':[np.absolute(errorplus) for errorplus in errorsplus]})
dg = pd.DataFrame({'x':[hval for hval in hvals],'y':[np.absolute(error0) for error0 in errors0]})
dh = pd.DataFrame({'x':[hval for hval in hvals],'y':[np.absolute(error3) for error3 in errors3]})

xlog,ylog,zlog,wlog = np.log10(df.x),np.log10(df.y),np.log10(dg.y),np.log10(dh.y)

plt.plot(xlog, ylog)
plt.plot(xlog,zlog)
plt.plot(xlog,wlog)
plt.show()

