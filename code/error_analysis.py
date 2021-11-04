import numpy as np
from scipy.integrate import complex_ode, solve_ivp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from collections import defaultdict

T = 1.0 # Period of one loop

# Basis vectors
kete1 = np.array([1, 0, 0, 0, 0, 0])
kete2 = np.array([0, 1, 0, 0, 0, 0])
ket1  = np.array([0, 0, 1, 0, 0, 0])
ket2  = np.array([0, 0, 0, 1, 0, 0])
ket3  = np.array([0, 0, 0, 0, 1, 0])
keta  = np.array([0, 0, 0, 0, 0, 1])

def cot(t):
    return np.cos(t)/np.sin(t)


def u(t):
    return (np.pi/2)*(np.sin(np.pi*t/T))**2

def v(t):
    return eta*(1 - np.cos(u(t)))

def up(t):
    return ((np.pi**2)/(2*T))*np.sin(2*np.pi*t/T)

def vp(t):
    return eta*up(t)*np.sin(u(t))
    


def n_mean(data):
    result = [sum(x) / len(x) for x in zip(*data)]
    return result
    

def genStates(par):
    chi = par[0]; xi = par[1]; theta = par[2]; phi = par[3];

    c1 = np.cos(theta)
    c2 = np.exp(1j*chi)*np.sin(theta)*np.cos(phi)
    c3 = np.exp(1j*xi)*np.sin(theta)*np.sin(phi)

    N1 = 1/(np.sqrt(1-np.abs(c3)**2))
    N2 = np.abs(c3)/(np.sqrt(1-np.abs(c3)**2))

    if np.isclose(np.abs(c1)**2, 1.0):
        d = ket1
        b1 = ket2
        b2 = -np.exp(1j*xi)*ket3
    else:
        d = np.array(c1*ket1 + c2*ket2 + c3*ket3);
        b1 = np.array(N1* ( -np.conj(c2)*ket1 + c1*ket2 ));
        b2 = np.array(N2*(c1*ket1 + c2*ket2 + (c3 - 1/np.conj(c3))*ket3 ));


# returns (Omega_1, Omega_2, Omega_a) at time t
def omegas(t,delta):
    if t ==  0.0 or t == T:
        O1 = 0; O2 = 0; Oa = 0;
    else:
        O1 = -2*up(t)
        O2 =  2*(vp(t)*cot(u(t))*np.sin(v(t)) + up(t)*np.cos(v(t)))
        Oa =  2*(vp(t)*cot(u(t))*np.cos(v(t)) - up(t)*np.sin(v(t)))


    return [O1*(1+delta),O2*(1+delta),Oa*(1+delta)]

def Ham(t,b1,b2,phi1,phi2,delta):
    
    if  t < T/2:
        phi1 = 0; phi2 = 0
    
    O = omegas(t,delta)
    
    T1 = (O[0]/2)*np.exp(-1j*phi1)*np.outer(np.conj(b1),kete1)
    T2 = (O[1]/2)*np.exp(-1j*phi2)*np.outer(np.conj(b2),kete2)
    Ta = (O[2]/2)*np.outer(np.conj(keta),kete2)

    H = np.array(T1 + T2 + Ta)
    H = np.array(H + np.conj(np.transpose(H)))
    
    return H

def f(t,y,delta,b1,b2,phi1,phi2):

    result =-1j*Ham(t,b1,b2,phi1,phi2,delta)@y
    
    return result

def unitaryGate(d,b1,b2,gamma1, gamma2):
    
    return np.outer(d,np.conj(d)) + np.exp(1j*gamma1)*np.outer(b1,np.conj(b1)) + np.exp(1j*gamma2)*np.outer(b2,np.conj(b2))
    

def Fidelity(p,q):
    return np.abs(np.dot(np.conj(p),q))


def fidLoop(y0,par1,par2,method,step, deltas):
    y0 = y0/np.linalg.norm(y0)
    phi1_1 = -par1[-2]; phi2_1 = -par1[-1]
    phi1_2 = -par2[-2]; phi2_2 = -par2[-1]
    
    d_1, b1_1, b2_1 = genStates(par1)
    d_2, b1_2, b2_2 = genStates(par2)
    
    U = unitaryGate(d_2,b1_2,b2_2,-phi1_2,-phi2_2) @ unitaryGate(d_1,b1_1,b2_1,-phi1_1,-phi2_1)
    y_exact = U @ y0
  

    fids = []
    for delt in deltas:
        sol1 = solve_ivp(f,(0,T),y0,method=method,
                         args=(delt, b1_1,b2_1,phi1_1,phi2_1),max_step=step)
        
        y1 = np.array(sol1.y[:,-1])
    
        sol2 = solve_ivp(f,(0,T),y1,method=method,
                         args=(delt, b1_2,b2_2,phi1_2,phi2_2),max_step=step)
        
        y_num = np.array(sol2.y[:,-1])

        fids.append(Fidelity(y_exact,y_num))

    return deltas, fids

def X_par():
    par1 = [0, 0, np.pi/4, np.pi/2, 0, np.pi]
    par2 = [0, 0, np.pi/2, np.pi/4, 0, np.pi]

    return par1,par2

def Z_par():
    par = [0,0,0,0,2*np.pi/3,4*np.pi/3]
    prob, U, finalState = singleLoop(initialState,par)
    return prob, U, finalState



y0 = np.array([0,0,1,0,1,0]); y0 = y0/np.linalg.norm(y0)
methods = ['RK45', 'BDF','RK23','DOP853']
steps = [1.0/2**n for n in range(1,10)]
deltas = [0.0]
par1, par2 = X_par()

phi1_1 = -par1[-2]; phi2_1 = -par1[-1]
phi1_2 = -par2[-2]; phi2_2 = -par2[-1]
    
d_1, b1_1, b2_1 = genStates(par1)
d_2, b1_2, b2_2 = genStates(par2)

global eta
eta = 4.0

U = unitaryGate(d_2,b1_2,b2_2,-phi1_2,-phi2_2) @ unitaryGate(d_1,b1_1,b2_1,-phi1_1,-phi2_1)
y_exact = U @ y0


res = np.zeros((len(steps),len(methods)))
for i,method in enumerate(methods):
    for j,step in enumerate(steps):

        sol1 = solve_ivp(f,(0,T),y0,method=method,
                         args=(0, b1_1,b2_1,phi1_1,phi2_1),max_step=step,rtol = 100, atol = 100)
        
        y1 = np.array(sol1.y[:,-1])
    
        sol2 = solve_ivp(f,(0,T),y1,method=method,
                         args=(0, b1_2,b2_2,phi1_2,phi2_2),max_step=step,rtol = 100, atol = 100)
        
        y_num = np.array(sol2.y[:,-1])
        error = np.linalg.norm(y_exact - y_num)
        res[j,i] = error


print(res)

plt.loglog(steps,res,'*--')
plt.grid()
plt.show()


    

