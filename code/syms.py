import numpy as np
import sympy as sp


# Basis vectors
kete1 = np.array([1, 0, 0, 0, 0, 0])
kete2 = np.array([0, 1, 0, 0, 0, 0])
ket1  = np.array([0, 0, 1, 0, 0, 0])
ket2  = np.array([0, 0, 0, 1, 0, 0])
ket3  = np.array([0, 0, 0, 0, 1, 0])
keta  = np.array([0, 0, 0, 0, 0, 1])


def simpStates(c1,c2,c3,chi,xi):

    c1 = c1
    c2 = c2*sp.exp(1j*chi)
    c3 = c3*sp.exp(1j*xi)
    
    N1 = 1/(sp.sqrt(1-np.abs(c3)**2))
    N2 = np.abs(c3)/(sp.sqrt(1-np.abs(c3)**2))

    d = c1*ket1 + c2*ket2 + c3*ket3;
    b1 = N1* ( -np.conj(c2)*ket1 + c1*ket2 );
    b2 = N2*(c1*ket1 + c2*ket2 + (c3 - 1/np.conj(c3))*ket3 );

    return d,b1,b2
    


def states(par):
    chi = par[0]; xi = par[1]; theta = par[2]; phi = par[3];

    c1 = sp.cos(theta)
    c2 = sp.exp(1j*chi)*sp.sin(theta)*sp.cos(phi)
    c3 = sp.exp(1j*xi)*sp.sin(theta)*sp.sin(phi)
    N1 = 1/(sp.sqrt(1-np.abs(c3)**2))
    N2 = np.abs(c3)/(sp.sqrt(1-np.abs(c3)**2))

    d = c1*ket1 + c2*ket2 + c3*ket3;
    b1 = N1* ( -np.conj(c2)*ket1 + c1*ket2 );
    b2 = N2*(c1*ket1 + c2*ket2 + (c3 - 1/np.conj(c3))*ket3 ); 

    return d,b1,b2
     

def unitaryGate(d,b1,b2,gamma1, gamma2):

    res= np.outer(d,np.conj(d)) + sp.exp(1j*gamma1)*np.outer(b1,np.conj(b1))+ sp.exp(1j*gamma2)*np.outer(b2,np.conj(b2))
    return res


g1_1 = sp.Symbol('g1_1',positive=True)
g2_1 = sp.Symbol('g2_1',positive=True)

g1_2 = sp.Symbol('g1_2',positive=True)
g2_2 = sp.Symbol('g2_2',positive=True)

t1 = sp.Symbol('t1',positive=True)
t2 = sp.Symbol('t2',positive=True)

p1 = sp.Symbol('p1',positive=True)
p2 = sp.Symbol('p2',positive=True)

x1 = sp.Symbol('x1',positive=True)
x2 = sp.Symbol('x2',positive=True)

c1 = sp.Symbol('c2',positive=True)
c2 = sp.Symbol('c2',positive=True)



a1 = sp.Symbol('a1',positive=True)
a2 = sp.Symbol('a2',positive=True)
a3 = sp.Symbol('a3',positive=True)

b1 = sp.Symbol('b1',positive=True)
b2 = sp.Symbol('b2',positive=True)
b3 = sp.Symbol('b3',positive=True)


par1 = [c1,x1,t1,p1,g1_1,g2_1]
par2 = [c2,x2,t2,p2,g1_2,g2_2]

d,b1,b2 = states(par1)
U1 = unitaryGate(d,b1,b2,g1_1,g2_1)[2:5,2:5]

d,b1,b2 = states(par2)
U2 = unitaryGate(d,b1,b2,g1_2,g2_2)[2:5,2:5]; 

U = U2 @ U1

res = (1/sp.sqrt(3))*np.array([[1,1,1],[1,sp.exp(2*sp.pi*1j/3),sp.exp(4*sp.pi*1j/3)],[1,sp.exp(4*sp.pi*1j/3),sp.exp(2*sp.pi*1j/3)]],dtype="complex_")

U = U.reshape(9,1)[:,0]; #U = sp.simplify(U)
res = res.reshape(9,1)[:,0]

print(U)

EQ1 = sp.Eq(U[0],res[0])
print(sp.nonlinsolve(EQ1, t1))
