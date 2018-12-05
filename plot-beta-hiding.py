import matplotlib.pyplot as plt
import math
import numpy as np

# Data for plotting
X = np.arange(0.0, 1.0, 0.01)
y = [(math.exp(7*x) - 1) for x in X]
fig, ax = plt.subplots()
ax.scatter(X,y)

ax.set_xlabel('bid/payment',fontsize=16)
ax.set_ylabel('number of instances',fontsize=16)
ax.grid()

# fig.savefig("test.png")
plt.show()



