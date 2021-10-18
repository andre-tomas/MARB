import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize

T = 1.0
t = np.linspace(0, T, 100)
eta = 4.0

kete1 = np.array([1, 0, 0, 0, 0, 0])
kete2 = np.array([0, 1, 0, 0, 0, 0])
ket1 = np.array([0, 0, 1, 0, 0, 0])
ket2 = np.array([0, 0, 0, 1, 0, 0])
ket3 = np.array([0, 0, 0, 0, 1, 0])
ket4 = np.array([0, 0, 0, 0, 0, 1])
    
    

def u(t,T):
    return (np.pi/2)*(np.sin(np.pi*t/T))**2

def v(t,T):
    return eta*(1 - np.cos(u(t,T)))

def D1(t,phi1,C):
    _, b1, _ = giveStateVectors(C)

    if t<T/2.0:
        phi1 = 0
    
    return np.exp(-1j*phi1)*np.cos(u(t,T))*b1 + 1j*np.sin(u(t,T))*kete1

def D2(t,phi2,C):
    _, _, b2 = giveStateVectors(C)

    if t<T/2.0:
        phi2 = 0

    return np.cos(u(t,T))*np.cos(v(t,T))*np.exp(-1j*phi2)*b2 - 1j*np.sin(u(t,T))*kete2 - np.cos(u(t,T))*np.sin(v(t,T))*ket4


def genCoeff(par):
    chi = par[0]; xi = par[1]; theta = par[2]; phi = par[3];
    
    c1 = np.exp(1j*chi)*np.sin(theta)*np.cos(phi)
    c2 = np.exp(1j*xi)*np.sin(theta)*np.sin(phi)
    c3 = np.cos(theta)
    
    N1 = np.abs(c1)/(np.sqrt(1-np.abs(c1)**2))
    N2 = 1/(np.sqrt(1-np.abs(c1)**2))
    
    return np.array([c1, c2, c3, N1, N2]) 

# given ([c1, c2, c3], initialstate, phi_2), return [f1,f2,f3] that fullfills that state
def genInitialCoeff(C,initialState,phi_2):
    c1 = C[0]; c2 = C[1]; c3 = C[2]; N1 = C[3]; N2 = C[4]

    phi_2 = 0 # Since gamma2 = 0 if t<T/2
    
    A = [[c1, N1*(c1 - 1/np.conj(c1)), 0],
         [c2, N1*c2, -N2*np.exp(-1j*phi_2)*c3],
         [c3, N1*c3, N2*np.exp(-1j*phi_2)*np.conj(c2)]]
    
    f = np.matmul(np.linalg.inv(A),initialState)
    return f

# Returns the d, b1, b2 state vectors
def giveStateVectors(C):
    c1 = C[0]; c2 = C[1]; c3 = C[2]; N1 = C[3]; N2 = C[4]

    d = c1*ket1 + c2*ket2 + c3*ket3;
    b1 = N1*(  (c1 - 1/np.conj(c1)) *ket1 + c2*ket2 + c3*ket3)
    b2 = N2*(-np.conj(c3)*ket2 + np.conj(c2)*ket3)

    return d,b1,b2

def unitaryGate(C,gamma1, gamma2):
    d, b1, b2 = giveStateVectors(C)

    return np.outer(d,np.conj(d)) + np.exp(1j*gamma1)*np.outer(b1,np.conj(b1)) + np.exp(1j*gamma2)*np.outer(b2,np.conj(b2))



def check(C,F):
    d, b1, b2 = giveStateVectors(C);

    #Start  = F[0]*d + F[1]*D1(0,-gamma1,C) + F[2]*D2(0, -gamma2, C)
    #print(f"The constructed starting state is {np.round(Start[2:5],4)}, should match with given initial state.")

    
    print("CHECK")
    print(f"<d|b1> = {np.round(np.dot(np.conj(d),b1),5)}")
    print(f"<d|b2> = {np.round(np.dot(np.conj(d),b2),5)}")
    print(f"<b2|b1> = {np.round(np.dot(np.conj(b2),b1),5)}")
    
    print(f"<d|d> = {np.round(np.dot(np.conj(d),d),5)}")
    print(f"<b1|b1> = {np.round(np.dot(np.conj(b1),b1),5)}")
    print(f"<b2|b2> = {np.round(np.dot(np.conj(b1),b1),5)}")

    print(f"c1 = {np.round(C[0],3)}")
    print(f"c2 = {np.round(C[1],3)}")
    print(f"c3 = {np.round(C[2],3)}")



def simulate(t,C,F,gamma1,gamma2):
    d, b1, b2 = giveStateVectors(C)
    
    Prob = []
    state = []
    for x in t:
        X = F[0]*d + F[1]*D1(x,-gamma1,C) + F[2]*D2(x, -gamma2, C)
        state.append(X)
        pe1 = np.abs(X[0])**2
        pe2 = np.abs(X[1])**2
        p1 = np.abs(X[2])**2
        p2 = np.abs(X[3])**2
        p3 = np.abs(X[4])**2
        p4 = np.abs(X[5])**2
        Prob.append([pe1, pe2, p1, p2, p3, p4])

        finalState = np.array(state[-1])
    return Prob, finalState


# par - parameters = (chi, xi, theta, phi, gamma1, gamma2)
# I - inital state - 1x3 normalized vector
# returns Prob, U
def singleLoop(initialState,par):
    #print(f"Running simulation with \npar = {np.round(par,6)} \nfrom initial state {initialState} ")
    chi = par[0]; xi = par[1]; theta = par[2]; phi = par[3]; gamma1 = par[4]; gamma2 = par[5]
    C = genCoeff(par)
    F = genInitialCoeff(C, initialState, -gamma2)

    #check(C,F)
    
    Prob, finalState = simulate(t,C,F,gamma1,gamma2)
    U = unitaryGate(C, gamma1, gamma2)

    return Prob, U, finalState

# Applies singleLoop() twice, first with par1, then par2
def doubleLoop(initialState, par1, par2):

    prob1, U1, intermedState = singleLoop(initialState, par1)
    prob2, U2, finalState = singleLoop(intermedState[2:5], par2)

    U = np.matmul(U2,U1)
    prob = prob1 + prob2

    return prob, U, finalState


def XGate(initialState):
    par1 = [0, 0, np.pi/4, np.pi/2, 0, np.pi]
    par2 = [0, 0, np.pi/2, np.pi/4, np.pi, 0]
    initialState = initialState/np.linalg.norm(initialState) # Make sure state is normalized

    prob, U, finalState = doubleLoop(initialState, par1, par2)

    return prob, U, finalState


def ZGate(initialState):
    par1 = [2*np.pi/3, 2*np.pi/3, np.pi, np.pi, 2*np.pi/3, 2*np.pi/3]
    par2 = [0, 2*np.pi/3, np.pi/2, np.pi/2, 4*np.pi/3, 4*np.pi/3]
    initialState = initialState/np.linalg.norm(initialState)

    prob, U, finalState = doubleLoop(initialState,par1,par2)

    return prob, U, finalState
    


def Gate_analysis(initialState):
    print(f"initial state: {np.round(initialState,3)}")
    global eta


    prob_X, X, finalState_X = XGate(initialState)
    prob_Z, Z, finalState_Z = ZGate(initialState)

    print(f"Unitary is X =\n{np.round(X[2:5,2:5],3)}")
    print(f"Unitary is Z =\n{np.round(Z[2:5,2:5],3)}")

    plt.figure()
    tt = np.linspace(0,len(prob_Z)/len(t),len(prob_Z))
    labels = ["pe1","pe2","p1","p2","p3","p4"]
    probPlotZ = plt.plot(tt,prob_Z,'--')
    plt.legend(iter(probPlotZ), labels)
    plt.title(f"Probability amplitudes of Z-gate with $\eta$ = {eta}")
    plt.xlabel("time")
    plt.ylabel("Probability")
    plt.axis([0, len(prob_Z)/len(t), 0, 1.1])
    plt.grid()

    plt.figure()
    tt = np.linspace(0,len(prob_X)/len(t),len(prob_X))
    labels = ["pe1","pe2","p1","p2","p3","p4"]
    probPlotX = plt.plot(tt,prob_X,'--')
    plt.legend(iter(probPlotX), labels)
    plt.title(f"Probability amplitudes of X-gate with $\eta$ = {eta}")
    plt.xlabel("time")
    plt.ylabel("Probability")
    plt.axis([0, len(prob_X)/len(t), 0, 1.1])
    plt.grid()

    plt.show()



def bruteForce(H):
    phase = [2*np.pi/3, 4*np.pi/3, np.pi, np.pi/3, np.pi/4, 2*np.pi/7, 2*np.pi/9,0 ]
    angle = [np.pi/4, np.pi/2]
    #H = np.array([[1, 1, 1], [1, np.exp(2*np.pi*1j/3), np.exp(4*np.pi*1j/3)], [1, np.exp(4*np.pi*1j/3), np.exp(2*np.pi*1j/3)]])
    print(H)
    par = []
    for chi, xi, gamma1, gamma2 in itertools.product(phase,phase,phase,phase):
        for theta, phi in itertools.product(angle,angle):
            par.append([chi,xi,theta,phi,gamma1,gamma2])
    print(len(par))

    i = 0
    for par1 in par:
        i = i + 1
        if i % 100000 == 0: print(i)
        C = genCoeff(par1)
        U = unitaryGate(C,par1[4],par1[5])[2:5,2:5]
        if np.isclose(np.linalg.norm(np.abs(H-U)),0.0):
                print(np.round(par1,2))
                print(np.round(par2,2))
                print(np.round(U,3))
    
def bruteForceDouble(H):
    phase = [2*np.pi/3, 4*np.pi/3, np.pi, 2*np.pi/7, 2*np.pi/9,]
    angle = [np.pi/4, np.pi/2]
    
    print(H)
    par = []
    for chi, xi,gamma1,gamma2 in itertools.product(phase,phase,phase,phase):
            for theta, phi in itertools.product(angle,angle):
                par.append([chi,xi,theta,phi,gamma1,gamma2])
    print(len(par)*len(par))
    par = remDup(par)
    print(len(par)*len(par))
    
    i = 0
    for par1 in par:
        C1 = genCoeff(par1)
        U1 = unitaryGate(C1,par1[4],par1[5])
        for par2 in par:
            i = i + 1
            if i % 1000000 == 0: print(i)
            C2 = genCoeff(par2)
            U2 = unitaryGate(C2,par2[4],par2[5])
            U = np.matmul(U2[2:5,2:5],U1[2:5,2:5])
            if np.isclose(np.linalg.norm(np.abs(H-U)),0.0):
                print("Match found!")
                print(f"par1 = {np.round(par1,2)}")
                print(f"par2 = {np.round(par2,2)}")
                print(np.round(U,3))
                
                
def remDup(list):
    new_list = [tuple(row) for row in list]
    uniques = np.unique(new_list, axis = 0)
    return uniques


def objective(par):
        par1 = par[:6]
        par2 = par[6:]
        
        C1 = genCoeff(par1)
        U1 = unitaryGate(C1,par1[4],par1[5])
        C2 = genCoeff(par2)
        U2 = unitaryGate(C2,par2[4],par2[5])
        U = np.matmul(U2[2:5,2:5],U1[2:5,2:5])
        
        H = (1/np.sqrt(3))*np.array([[1, 1, 1],
                                   [1, np.exp(2*np.pi*1j/3), np.exp(4*np.pi*1j/3)],
                                   [1, np.exp(4*np.pi*1j/3), np.exp(2*np.pi*1j/3)]])

        
        return np.linalg.norm(np.abs(H-U1))
    
def optPar():
    
    par1 = [0, 0, np.pi/4, np.pi/4, 0, np.pi]
    par2 = [0, 0, np.pi/4, np.pi/4, np.pi, 0]
    par = par1 + par2
    b = (0.0, 2*np.pi)
    bnds = (b,b,b,b,b,b,b,b,b,b,b,b)

    
    sol = minimize(objective, par, method='SLSQP',bounds=bnds)
    return sol
    
    
print("\nRunning main...\n")

initialState = np.array([1,0,0]);
initialState = initialState/np.linalg.norm(initialState)


#sol = optPar()
#print(sol.x)

#par1 = [0, 0, np.pi/4, np.pi/4, 0, np.pi]
#par2 = [0, 0, np.pi/4, np.pi/4, np.pi, 0]
par1 = [1.79946272e-03, 1.24708556e-06, 8.50904186e-01, 1.06932640, 9.03263451e-07, 1.31996338]
par2 = [1.12352545e-06, 8.56695375e-07, 1.23972414, 3.50879440e-01, 3.14159379, 2.50834119e-01]

#Gate_analysis(initialState)

prob, U, finalState = doubleLoop(initialState, par1,par2)

print(np.round(U[2:5,2:5],3))


tt = np.linspace(0,len(prob)/len(t),len(prob))
labels = ["pe1","pe2","p1","p2","p3","p4"]
probPlot = plt.plot(tt,prob,'--')
plt.legend(iter(probPlot), labels)
plt.title(f"Probability amplitudes with $\eta$ = {eta}")
plt.xlabel("time")
plt.ylabel("Probability")
plt.axis([0, len(prob)/len(t), 0, 1.1])
plt.grid()

plt.show()







#T = np.array([[1,0,0],[0,np.exp(2*np.pi*1j/9),0],[0,0,np.exp(-2*np.pi*1j/9)]])
#H = (1/np.sqrt(3))*np.array([[1, 1, 1], [1, np.exp(2*np.pi*1j/3), np.exp(4*np.pi*1j/3)], [1, np.exp(4*np.pi*1j/3), np.exp(2*np.pi*1j/3)]])
#X = np.array([[0,0,1], [1,0,0], [0,1,0]])
#Z = np.array([[1,0,0], [0,np.exp(2*np.pi*1j/3),0], [0,0,np.exp(4*np.pi*1j/3)]])


#print(np.round(H,3))









