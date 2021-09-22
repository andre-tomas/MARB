from sympy import *
from sympy.abc import T,eta,t
import matplotlib.pyplot as plt

def cot(x):
   return cos(x)/sin(x)



T = Symbol('T', positive=True)
eta = Symbol('eta', positive=True)
t = Symbol('t', positive=True)

def alfa(t):
   return (pi/2)*(sin(pi*t/T))**2

def beta(t):
    return eta*(1-cos(alfa(t)))

def alfap(t):
    return (pi**2)/(2*T)*sin(2*pi*t/T)

def betap(t):
    return eta*alfap(t)*sin(alfa(t))


def omega(t):
    return 2*(betap(t)*cot(alfa(t))*sin(beta(t)) + alfap(t)*cos(beta(t)))

 
result = integrate(omega, (t,0,T/2) )
print(result)
