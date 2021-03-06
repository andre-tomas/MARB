import numpy as np
import scipy as sp

# Tomas Andre 2022-05-07
# Main document for dynamicaly simulating a qudit        

class Qudit:
    def __init__(self,dim,eta,T=1.0,N=100):
        # Dimension of the qudit
        self.dim = dim

        # Coupling strength (could be multiple values)
        self.eta = eta

        # Period time
        self.T = T
        
        # Time vector
        self.t = np.linspace(0,T,N)

        # Dark-bright basis kets.
        self.ketDB = None
        
        # Parameters for unitary 
        self.par = np.array([])
        
        # Solution vector of state evolution is saved here 
        self.sol = None
        # self.stats = Some way to store stats for analysis, maybe new class?
        

        """ 
        basis kets, the first dim-1 kets are excited states, the following dim kets 
        are the orignal basis, and the last ket is the auxilliary state

        ket[0] = |e_1>, ket[1] = |e_2>, ..., ket[dim-2] = |e_(dim-1)>,
        ket[dim-1] = |1>, ket[dim] = |2>, ket[dim + 1] = |3>,..., ket[2dim-1] = |dim>
        ket[dim] = |aux>
        """
        kets = np.identity(2*self.dim)
        '''
        Exited kets. ketsE[0]=|e_1>, ...,ketsE[dim-1]=|e_(dim-1)> 
        Computational kets. ketsC[0] = |1>, ..., ketsC[2dim-1] = |dim>
        Auxillary state ket. ketA = |a>
        '''
        self.ketsE = kets[:self.dim-1]
        self.ketsC = kets[self.dim-1:2*self.dim-1]
        self.ketsA = kets[-1]
        

    def setParameters(self, par):
        self.par = par
            
            
        
        #  
    def genDBBasis(self):
        phi = self.par[:self.dim-1]
        theta = self.par[self.dim-1:]


        # Generate state coeffcients 
        c = np.zeros(self.dim, dtype = "complex_")
        c[0] = np.cos(phi[0]) 
        for k in range(1,self.dim-1):
            temp = 1
            for l in range(k):
                temp = temp*np.sin(phi[l])
            c[k] = np.exp(1j*theta[k])*np.cos(phi[k-1])*temp

            temp = 1
            for phis in phi:
                temp = temp*np.sin(phis)
        c[-1] = temp*np.exp(1j*theta[-1])

        
        # Generate lambda coeffcient
        lambdaCoef = np.zeros(self.dim, dtype="complex_"); lambdaCoef[0] = 0; lambdaCoef[1] = 0;
        lambdaCoef[2:] = [np.vdot(c[:k-1],c[:k-1])*(-1/np.conj(c[k])) for k in range(2,self.dim)]
        
        
        # Generate Normalization factors
        N = np.zeros(self.dim-1, dtype="complex_")
        N = [np.reciprocal(np.sqrt(np.vdot(c[:k+1],c[:k+1]) + np.abs(lambdaCoef[k+1])**2))
             for k in range(self.dim-1)]

        # Construct new basis states
        DB = np.zeros([self.dim,self.dim],dtype="complex_")

        # Handle edge cases
        DB[0,:] = c*self.ketsC # The dark state
        DB[1,:] = N[0]*(-np.conj(c[1])*self.ketsC[0] + np.conj(c[0])*self.ketsC[1]) # state b1
        
        for k in range(2,self.dim-1):
            for l in range(k):
                DB[k,:] = N[k-1]*(c[:k]*self.ketsC[:k] + c[k+1]*lambdaCoef[k+1])

                
        print(DB)

        
        """
        Implement this
        Generate the dark bright basis from the kets
        """
        return DB
        
    
    def getDim(self):
        return self.dim
    
    # Simulates the dynamics of the qudit 
    def simulate(self, initial_state, par):
        
        """
        Implement this
        """
        self.sol = None

    def getResults(self, printToFile = False):
        """
        Shows plots and prints all data to files if 
        Implement this
        """

        # For debugging 
    def printKets(self):
        print(f"E-kets: \n {self.ketsE}")
        print(f"C-kets: \n {self.ketsC}")
        print(f"A-ket: \n {self.ketsA}")
       

dim = 3
eta = 4.0
T = 1.0
N = 100
par = (np.pi/3, np.pi/3, np.pi/3, np.pi/3)
        
qu3 = Qudit(dim,eta,T,N)
print(qu3)
qu3.printKets()
qu3.setParameters(par)
qu3.genDBBasis()

        
