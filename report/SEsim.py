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
###############################

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


    if np.isclose(np.abs(c1)**2, 1.0):
        d = ket1
        b1 = ket2
        b2 = -np.exp(1j*xi)*ket3

    else:
        N1 = 1/(np.sqrt(1-np.abs(c3)**2))
        N2 = np.abs(c3)/(np.sqrt(1-np.abs(c3)**2))
        d = c1*ket1 + c2*ket2 + c3*ket3;
        b1 = N1* ( -np.conj(c2)*ket1 + c1*ket2 );
        b2 = N2*(c1*ket1 + c2*ket2 + (c3 - 1/np.conj(c3))*ket3 ); 

    return d,b1,b2 



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
    if  t < T/2.0:
        phi1 = 0; phi2 = 0
    
    O = omegas(t,delta)
    
    T1 = (O[0]/2)*np.exp(-1j*phi1)*np.outer(np.conj(b1),kete1)
    T2 = (O[1]/2)*np.exp(-1j*phi2)*np.outer(np.conj(b2),kete2)
    Ta = (O[2]/2)*np.outer(np.conj(keta),kete2)

    H = np.array(T1 + T2 + Ta)
    H = np.array(H + np.conj(np.transpose(H)))
    
    return H

# Schrodinger equation 
def f(t,y,delta,b1,b2,phi1,phi2):
    return -1j*Ham(t,b1,b2,phi1,phi2,delta)@y

# Generate unitary from parameters
def unitaryGate(d,b1,b2,gamma1, gamma2):
    return np.outer(d,np.conj(d)) + np.exp(1j*gamma1)*np.outer(b1,np.conj(b1)) + np.exp(1j*gamma2)*np.outer(b2,np.conj(b2))
    
def Fidelity(p,q):
    return np.abs(np.dot(np.conj(p),q))

# Calculates the fidelities with a given initial condition for a set of deltas
def fidLoop(y0,par,method, deltas):
    FLAG = False
    if len(par) > 6:
        FLAG = True
        par1 = par[:6]
        par2 = par[6:]
        phi1 = -par1[-2]; phi2 = -par1[-1]
        phi1_2 = -par2[-2]; phi2_2 = -par2[-1]
    
        d, b1, b2 = genStates(par1)
        d_2, b1_2, b2_2 = genStates(par2)
    
        U = unitaryGate(d_2,b1_2,b2_2,-phi1_2,-phi2_2) @ unitaryGate(d,b1,b2,-phi1,-phi2)

    else:
        phi1 = -par[-2]; phi2 = -par[-1]
        d, b1, b2 = genStates(par)
        U = unitaryGate(d,b1,b2,-phi1,-phi2)
        

    fids = []
    for delt in deltas:
        
        sol1 = solve_ivp(f,(0,T),y0,method=method,
                         args=(delt, b1,b2,phi1,phi2))
        y_num = np.array(sol1.y[:,-1]); #y_num = y_num/np.linalg.norm(y_num)

        if FLAG:
            sol2 = solve_ivp(f,(0,T),y_num,method=method,
                         args=(delt, b1_2,b2_2,phi1_2,phi2_2))
            y_num = np.array(sol2.y[:,-1]); y_num = y_num/np.linalg.norm(y_num)
            
        fids.append(Fidelity(y_exact[2:5],y_num[2:5]))
    return deltas, fids

# Averages fidelity over many initial states
def averageFid(par,n):
    k = 20 # number of points sampled [+k*delta, -k*delta]
    delta = 0.015 # error size
    deltas = [x*delta for x in range(-k,k+1)]
    fids = []
    for k in range(n):
            y0 = np.array(np.random.rand(3),dtype="complex_")
            y0 = y0/np.linalg.norm(y0)
            y0 = np.concatenate(([0.0,0.0],y0,[0.0]),dtype="complex_")
       
            x,y = fidLoop(y0,par,'BDF', deltas)
            fids.append(y)

    return x, n_mean(fids)

# Calculates the average fidelity sampled over n states using the parameters of par_func
def Fidplot(n, etas,par_func):
    par = par_func()
    name =  par_func.__name__; name = name[0]
    print(name)
    
    global eta

    eta = etas[0]
    x1,y1 = averageFid(par,n, False)
    eta = etas[1]
    _,y2 = averageFid(par,n, False)

    return x1,y1,y2

## Gate parameters
def X_par():
    par1 = [0, 0, np.pi/4, np.pi/2, 0, np.pi]
    par2 = [0, 0, np.pi/2, np.pi/4, 0, np.pi]
    return par1 + par2

def Z_par():
    par = [0,0,0,0,2*np.pi/3,4*np.pi/3]
    return par

def T_par():
    par = [0,0,0,0,2*np.pi/9.0,-2*np.pi/9.0]
    return par

def H_par():
    par = [6.41010859e-04, 6.55568952e-04, 4.75667128e-01, 7.85362474e-01,1.58054108e+00, 1.56302702e+00, 9.81289849e-03, 3.56878815e-18,1.18743379e+00, 2.15063745e+00, 9.74301696e-17, 1.56882773e+00]
    return par

etas = [0.0, 4.0] # eta values
n = 500 # Number of sampled states

x, Ty1, Ty2 = Fidplot(n,etas,T_par)
_, Xy1, Xy2 = Fidplot(n,etas,X_par)
_, Hy1, Hy2 = Fidplot(n,etas,H_par)
_, Zy1, Zy2 = Fidplot(n,etas,Z_par)

fig, axs  = plt.subplots(2,2)
#gs = fig.add_gridspec((2,2), hspace=0)
#axs = gs.subplots(sharex=True, sharey=True)
axs[0,0].plot(x,Ty1,'*--',label=f"$\eta =$ {etas[0]}")
axs[0,0].plot(x,Ty2,'*--',label=f"$\eta =$ {etas[1]}")
axs[0,0].set_title(f"T-Gate")
axs[0,1].plot(x,Xy1,'*--',label=f"$\eta =$ {etas[0]}")
axs[0,1].plot(x,Xy2,'*--',label=f"$\eta =$ {etas[1]}")
axs[0,1].set_title(f"X-Gate")
axs[1,0].plot(x,Hy1,'*--',label=f"$\eta =$ {etas[0]}")
axs[1,0].plot(x,Hy2,'*--',label=f"$\eta =$ {etas[1]}")
axs[1,0].set_title(f"H-Gate")
axs[1,1].plot(x,Zy1,'*--',label=f"$\eta =$ {etas[0]}")
axs[1,1].plot(x,Zy2,'*--',label=f"$\eta =$ {etas[1]}")
axs[1,1].set_title(f"Z-Gate")

fig.suptitle(f'Fidelity averaged over {n} states')
for ax in fig.get_axes():
    ax.set_ylim(ymax=1)
    ax.set(xlabel='$\delta\Omega$', ylabel='Fidelity')
    ax.label_outer()
    ax.grid()
    ax.legend()
plt.show()


    

