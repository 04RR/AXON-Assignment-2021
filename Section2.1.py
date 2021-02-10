import numpy as np
import matplotlib.pyplot as plt

R = 2
# I_A = [0 for i in range(0, 100)]
# I_A[25:65] = 1.5
I_B = [0.75 for _ in range(0, 100)]
I_C = [5*np.sin(75*t) for t in range(0, 100)]
I_D = [-2*np.sin(30*t) + 3*np.sin(45*t) + np.cos(60*t) for t in range(0, 100)]
tau = 4
ur = 0
threshold = 10
t, t1 = 0, 0
U = []
i = 0

while(True):
    u = ur
    while(t < threshold):
        # Change I here to I_A, I_B, I_C and I_D
        u = R*I_D[t]*(1-np.exp((t1-t)/tau))
        t += 1
        U.append(u)
    if len(U) >= 100:
        break
    t1 = t
    t = 0

plt.plot(U)
plt.show()
