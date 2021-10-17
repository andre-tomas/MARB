import numpy as np
import math
import matplotlib.pyplot as plt

N = 8
M = 10

X = range(1,10)

info = []
for n in range(2,N):
    Y = [math.log(n**x,2)for x in X]
    info.append(Y)

plt.plot(X,np.transpose(info),'--*')
plt.grid()
plt.show()
