import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi, 101, endpoint=True)
f = np.sin(x)
simps(f, x)
plt.plot(x, f, '.-')

x = np.linspace(0, 4, 100, endpoint=False)
f = x * x
f = np.append(f, 0)
x = np.append(x, 4)
simps(f, x)
plt.plot(x, f, '.-')
