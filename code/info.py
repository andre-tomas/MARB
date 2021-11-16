import numpy as np
import math
import matplotlib.pyplot as plt

max = 20



Dims = [d for d in range(2,max)]
Y = [1.0/math.log(d,2) for d in Dims]


plt.plot(Dims,Y,'--*')
plt.grid()
plt.show()
