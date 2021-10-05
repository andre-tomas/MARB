import numpy as np
import matplotlib.pyplot as plt


T = 1
eta = 4

kete1 = np.array([1, 0, 0, 0, 0, 0])
kete2 = np.array([0, 1, 0, 0, 0, 0])
ket1 = np.array([0, 0, 1, 0, 0, 0])
ket2 = np.array([0, 0, 0, 1, 0, 0])
ket3 = np.array([0, 0, 0, 0, 1, 0])
ket4 = np.array([0, 0, 0, 0, 0, 1])
    
    

def u(t,T):
    return np.pi/2*(np.sin(np.pi*t/T))**2

def v(t,T, eta):
    return eta*(1 - np.cos(u(t,T)))

def D1(t,phi1,C):
    _, b1, _ = giveStateVectors(C)

    return np.cos(u(t,T))*b1 + 1j*np.exp(1j*phi1)*np.sin(u(t,T))*kete1

def D2(t,phi2,C):
    _, _, b2 = giveStateVectors(C)

    return np.cos(u(t,T))*np.cos(v(t,T,eta))*np.exp(-1j*phi2)*b2 - 1j*np.sin(u(t,T))*kete2 - np.cos(u(t,T))*np.sin(v(t,T,eta))*ket4

    
    

# takes 4 angles (chi, xi, theta, phi) and returns the coeffs [c1, c2, c3]
def genCoeff(chi, xi, theta, phi):
    c1 = np.exp(1j*chi)*np.sin(theta)*np.cos(phi)
    c2 = np.exp(1j*xi)*np.sin(theta)*np.sin(phi)
    c3 = np.cos(theta)
    return np.array([c1, c2, c3])

# given ([c1, c2, c3], initialstate, phi_2), return [f1,f2,f3] that fullfills that state
def genInitialCoeff(C,initialState,phi_2):
    c1 = C[0]; c2 = C[1]; c3 = C[2];
    N1 = np.abs(c1)/(np.sqrt(1-np.abs(c1)**2)); N2 = 1/(np.sqrt(1-np.abs(c1)**2))

    A = [[c1, N1*(c1 - 1/np.conj(c1)), 0], [c2, N1*c2, -np.exp(-1j*phi_2)*N2*c3], [c3, N1*c3, N2*np.exp(-1j*phi_2)*np.conj(c2)]]

    #A = [[c1, N1*c1*((np.abs(c1)**2 - 1)/(1 - np.abs(c2)**2 - np.abs(c3)**2)), 0], [c2, N1*c2, np.exp(-1j*phi_2)*N2*c3], [c3, N1*c3, N2*np.exp(-1j*phi_2)*np.conj(c2)]]

    print(f" in {initialState}")
    f = np.matmul(np.linalg.inv(A),initialState)
    print(f"f {f}")
    return f

# Returns the d, b1, b2 state vectors
def giveStateVectors(C):
    c1 = C[0]; c2 = C[1]; c3 = C[2];
    N1 = np.abs(c1)/(np.sqrt(1-np.abs(c1)**2)); N2 = 1/(np.sqrt(1-np.abs(c1)**2))


    d = c1*ket1 + c2*ket2 + c3*ket3;
    b1 = N1*((c1- 1/np.conj(c1))*ket1 + c2*ket2 + c3*ket3)
    b2 = N2*(-np.conj(c3)*ket2 + np.conj(c2)*ket3)

    return d,b1,b2

def unitaryGate(C,gamma1, gamma2):
    d, b1, b2 = giveStateVectors(C)

    return np.transpose(d)*d + np.exp(1j*gamma1)*np.transpose(b1)*b1 + np.exp(1j*gamma2)*np.transpose(b2)*b2


    
    

print("Running main...\n")

chi = 0; xi = 0; theta = np.pi/4; phi = np.pi/4;
gamma1 = 0; gamma2 = np.pi

initialState = np.array([1, 1, 0]);
initialState = initialState/np.linalg.norm(initialState) 
C = genCoeff(chi, xi, theta, phi)
F = genInitialCoeff(C, initialState, -gamma2)
N1 = np.abs(C[0])/(np.sqrt(1-np.abs(C[0])**2)); N2 = 1/(np.sqrt(1-np.abs(C[0])**2))

d, b1, b2 = giveStateVectors(C)

onestate = np.real(F[0]*d + F[1]*D1(0,gamma1,C) + F[2]*D2(0,gamma2,C))

print(f"Initial state = {onestate},\ncomputation state = {onestate[2:-1]}.")

t = np.linspace(0, T, 50)
P = []
for x in t:
    X = F[0]*d + F[1]*D1(x,gamma1,C) + F[2]*D2(x, gamma2, C)
    pe1 = np.abs(X[0])**2
    pe2 = np.abs(X[1])**2
    p1 = np.abs(X[2])**2
    p2 = np.abs(X[3])**2
    p3 = np.abs(X[4])**2
    p4 = np.abs(X[5])**2
    P.append([pe1, pe2, p1, p2, p3, p4])
    
labels = ["pe1","pe2","p1","p2","p3","p4"]
probPlot = plt.plot(t,P,'--')
plt.legend(iter(probPlot), labels)
plt.xlabel("time")
plt.ylabel("Probability")
plt.axis([0, T, 0, 1])
plt.show()
