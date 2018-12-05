import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 1.0, 0.01)
s = (1 + t)/(1 + 3*t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set_xlabel('p',fontsize=16)
ax.set_ylabel('equilibrium bid',fontsize=16)
ax.grid()

# fig.savefig("test.png")
plt.show()

