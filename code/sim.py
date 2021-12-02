import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize

T = 1.0 # Period of one loop
t = np.linspace(0, T, 1000)
eta = 4.0 # Coupling "strength" to auxilary state

# Basis vectors
kete1 = np.array([1, 0, 0, 0, 0, 0])
kete2 = np.array([0, 1, 0, 0, 0, 0])
ket1 = np.array([0, 0, 1, 0, 0, 0])
ket2 = np.array([0, 0, 0, 1, 0, 0])
ket3 = np.array([0, 0, 0, 0, 1, 0])
ket4 = np.array([0, 0, 0, 0, 0, 1])
    
# Parametrizes the dark paths
def u(t):
    return (np.pi/2)*(np.sin(np.pi*t/T))**2
def v(t):
    return eta*(1 - np.cos(u(t)))

def D1(t,phi1,b1):
    if t<T/2.0:
        phi1 = 0
    return np.exp(-1j*phi1)*np.cos(u(t))*b1 + 1j*np.sin(u(t))*kete1

def D2(t,phi2,b2):
    if t<T/2.0:
        phi2 = 0
    return np.cos(u(t))*np.cos(v(t))*np.exp(-1j*phi2)*b2 - 1j*np.sin(u(t))*kete2 - np.cos(u(t))*np.sin(v(t))*ket4


# Generates the new basis states in the dark-bright basis given a set of parameters
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

# Generates the starting coeffcients required for a given initial state
def genInitialCoeff(d,b1,b2,initialState):
    A = np.transpose(np.array([d[2:5],b1[2:5],b2[2:5]]))
    f = np.matmul(np.linalg.inv(A),initialState)
    
    return f

# Returns the a analytical unitary based on the parameters
def unitaryGate(d,b1,b2,gamma1, gamma2):    
    return np.outer(d,np.conj(d)) + np.exp(1j*gamma1)*np.outer(b1,np.conj(b1)) + np.exp(1j*gamma2)*np.outer(b2,np.conj(b2))


# Checking used for debugging
def check(d,b1,b2,F):
    print("CHECK")
    print(f"<d|b1> = {np.round(np.dot(np.conj(d),b1),5)}")
    print(f"<d|b2> = {np.round(np.dot(np.conj(d),b2),5)}")
    print(f"<b2|b1> = {np.round(np.dot(np.conj(b2),b1),5)}")
    print(f"<d|d> = {np.round(np.dot(np.conj(d),d),5)}")
    print(f"<b1|b1> = {np.round(np.dot(np.conj(b1),b1),5)}")
    print(f"<b2|b2> = {np.round(np.dot(np.conj(b1),b1),5)}")


# Steps through one loop and saves the state populations
def simulate(d,b1,b2,F,gamma1,gamma2):
    Prob = []
    state = []
    for x in t:
        X = F[0]*d + F[1]*D1(x,-gamma1,b1) + F[2]*D2(x, -gamma2, b2)
        state.append(X)
        pe1 = np.abs(X[0])**2
        pe2 = np.abs(X[1])**2
        p1 = np.abs(X[2])**2
        p2 = np.abs(X[3])**2
        p3 = np.abs(X[4])**2
        pa = np.abs(X[5])**2
        Prob.append([pe1, pe2, p1, p2, p3, pa])

    finalState = np.array(state[-1])
    return Prob, finalState


# par - parameters = (chi, xi, theta, phi, gamma1, gamma2)
# I - inital state - 1x3 normalized vector
# returns Prob, U
def singleLoop(initialState,par):
    gamma1 = par[4]; gamma2 = par[5]
    d,b1,b2 = genStates(par)
    F = genInitialCoeff(d,b1,b2,initialState)
    check(d,b1,b2,F)
    Prob, finalState = simulate(d,b1,b2,F,gamma1,gamma2)
    U = unitaryGate(d,b1,b2, gamma1, gamma2)

    return Prob, U, finalState

# Applies singleLoop() twice, first with par1, then par2
def doubleLoop(initialState, par1, par2):
    prob1, U1, intermedState = singleLoop(initialState, par1)
    prob2, U2, finalState = singleLoop(intermedState[2:5], par2)
    U = np.matmul(U2,U1)
    prob = prob1 + prob2

    return prob, U, finalState


    # Finds optimized parameters
def opt_parameters(V, x0):
    par1 = x0[:6]
    par2 = x0[6:]
    par = par1 + par2
    b = (0.0, 2*np.pi)
    bnds = (b,b,b,b,b,b,b,b,b,b,b,b)

    sol = minimize(objective, par,args = (V), method='SLSQP',bounds=bnds,options={'maxiter':10000,'disp':True})
    return sol

# Objective function for optimization
def objective(par,V):
        par1 = par[:6]
        par2 = par[6:]
        
        d,b1,b2 = genStates(par1)
        U1 = unitaryGate(d,b1,b2,par1[4],par1[5])
        d,b1,b2 = genStates(par2)
        U2 = unitaryGate(d,b1,b2,par2[4],par2[5])
        
        U = np.matmul(U2[2:5,2:5],U1[2:5,2:5])
        
        return np.linalg.norm(np.abs(V-U), 'fro')

# Parameters for X-gate
def XGate(initialState):
    par1 = [0, 0, np.pi/4, np.pi/2, 0, np.pi]
    par2 = [0, 0, np.pi/2, np.pi/4, 0, np.pi]
    prob, U, finalState = doubleLoop(initialState, par1, par2)

    return prob, U, finalState

# Parameters for Z-gate
def ZGate(initialState):
    par = [0,0,0,0,2*np.pi/3,4*np.pi/3]
    prob, U, finalState = singleLoop(initialState,par)
    return prob, U, finalState

# Parameters for T-gate
def TGate(initialState):
    par = [0,0,0,0,2*np.pi/9,-2*np.pi/9]
    prob, U, finalState = singleLoop(initialState,par)
    return prob, U, finalState


# parameters found via optimization and thus not fancy as for X,Z and T
def HGate(initialState):
    par = [6.41010859e-04, 6.55568952e-04, 4.75667128e-01, 7.85362474e-01,1.58054108e+00, 1.56302702e+00, 9.81289849e-03, 3.56878815e-18,1.18743379e+00, 2.15063745e+00, 9.74301696e-17, 1.56882773e+00]

    prob, U, finalState = doubleLoop(initialState,par[:6],par[6:])
    return prob, U, finalState


print("\nRunning main...\n")
# Analytical qutrit gates (used for comparision/optimization)
Tg = np.array([[1,0,0],[0,np.exp(2*np.pi*1j/9),0],[0,0,np.exp(-2*np.pi*1j/9)]])
Hg = (1/np.sqrt(3))*np.array([[1, 1, 1], [1, np.exp(2*np.pi*1j/3), np.exp(4*np.pi*1j/3)], [1, np.exp(4*np.pi*1j/3), np.exp(2*np.pi*1j/3)]])
Xg = np.array([[0,0,1], [1,0,0], [0,1,0]])
Zg = np.array([[1,0,0], [0,np.exp(2*np.pi*1j/3),0], [0,0,np.exp(4*np.pi*1j/3)]])

# Set initialstate and normalize it
initialState = np.array([1,1,1]);
initialState = initialState/np.linalg.norm(initialState)


# Optimizations 
#B = np.pi/4
#par1 = [B,B,B,B,B,B]
#par2 = [B,B,B,B,B,B]
#par0 = par1 + par2
#par = opt_parameters(Hg, par0).x


prob, U, finalState = HGate(initialState)
#prob, U, finalState = singleLoop(initialState, par1)
#prob, U, finalState = doubleLoop(initialState, par[:6],par[6:])

print("Final states ")
print(f"exact: {np.round(U[2:5,2:5] @ initialState,6)}")
print(f"numerical: {np.round(finalState[2:5],6)}")

plt.figure()
tt = np.linspace(0,len(prob)/len(t),len(prob))
labels = ["p_e1","p_e2","p_1","p_2","p_3","p_a"]
probPlot = plt.plot(tt,prob,'--')
plt.legend(iter(probPlot), labels)
plt.title(f"Probabilities with $\eta$ = {eta}")
plt.xlabel("time")
plt.ylabel("Probability")
plt.axis([0, len(prob)/len(t), 0, 1.1])
plt.grid()

plt.show()



