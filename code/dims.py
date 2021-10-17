import numpy as np
import matplotlib.pyplot as plt
import math


N = 20
par = []
su = []
K = []
X = []
perfectX = []
perfectY = []
for n in range(2,N):
    temp1 = n**2 -1
    temp2 = 3*(n-1)
    temp3 = float(temp1)/float(temp2)

    print(f"{n} | {temp2} | {temp1} | {math.ceil(temp3)}")
    su.append(temp1)
    par.append(temp2)
    K.append(math.ceil(temp3))
    X.append(n)
    if temp3.is_integer():
        #print(f"{n} is perfect!")
        perfectY.append(temp3)
        perfectX.append(n)


plt.plot(X,K,'--', )
plt.plot(perfectX,perfectY,'o',label = "Perfect match")
plt.xlabel("n")
plt.ylabel("#loops")
plt.legend()
plt.grid()
plt.show()



    
