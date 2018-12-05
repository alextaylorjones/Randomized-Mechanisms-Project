import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 1.0, 0.01)
s = (1./3)*(1. + t)**2/(1. + 3*t)

print t
print s

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set_xlabel('p',fontsize=18)
ax.set_ylabel('expected revenue',fontsize=18)
plt.setp(ax.get_xticklabels(), rotation='horizontal', fontsize=18)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=18)
ax.grid()

plt.show()



