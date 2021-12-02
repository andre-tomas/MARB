import numpy as np
import matplotlib.pyplot as plt
import math


def isPrime(num):
    if num < 1:
        return False
    
    for i in range(2, int(math.sqrt(float(num))+1)):
        if (num % i) == 0:
            return False
    return True
N_u = 2
N = N_u + 100
par = []
su = []
K = []
X = []
perfectX = []
perfectY = []
primes = 0
for n in range(N_u,N):
    if isPrime(n):
        primes = primes + 1
        temp1 = n**2 -1
        temp2 = 3*(n-1)
        temp3 = float(temp1)/float(temp2)

        
        su.append(temp1)
        par.append(temp2)
        K.append(math.ceil(temp3))
        X.append(n)
        if temp3.is_integer():
            #print(f"{n} is perfect!")
            print(f"{n} | {temp2} | {temp1} | {math.ceil(temp3)}")
            perfectY.append(temp3)
            perfectX.append(n)
print(f"Number of primes in {N_u} to {N}: {primes}")

plt.plot(X,K,'--*', )
plt.plot(perfectX,perfectY,'o',label = "Perfect match")
plt.xlabel("n")
plt.ylabel("#loops")
plt.legend()
plt.grid()
plt.show()



    
