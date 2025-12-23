import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy.linalg as LA

def TN_linear_system(t, f, R, S, T, P, w_I, w_G, Lambda):
    f0, f1, f2 = f

    G0 = P
    G1 = (T + S) / 2
    G2 = R

    avgG = G0 * f0 + G1 * f1 + G2 * f2

    df0 = (1 + w_I * T) * f1 \
          + Lambda * w_G * f0 * (P - avgG)

    df1 = -(2 + w_I * (T + S)) * f1 \
          + Lambda * w_G * f1 * (G1 - avgG)

    df2 = (1 + w_I * S) * f1 \
          + Lambda * w_G * f2 * (R - avgG)

    return np.array([df0, df1, df2])
w_I, w_G, Lambda = 0.1, 0.1, 10
f0_init = np.array([0.33, 0.33, 0.34])
t_span = (0, 200)
dt = 0.001
# HD = dict(R=6, S=5, T=12, P=3)
PD = dict(R=7, S=3, T=8, P=6) #P>
# PD = dict(R=6, S=3, T=7, P=5) #R>
# PD = dict(R=5, S=3, T=8, P=6) 
def solve(system, y0, params):
    t_vals = np.arange(t_span[0], t_span[1], dt)
    y_vals = np.zeros((len(t_vals), 3))
    y = y0.copy()

    for i, t in enumerate(t_vals):
        y_vals[i] = y
        y = rk4_step(lambda tt, yy: system(tt, yy, **params,
                                            w_I=w_I, w_G=w_G, Lambda=Lambda),
                      t, y, dt)
        y = y / np.sum(y) 

    return t_vals, y_vals
# t_HD, res_HD = solve(TN_linear_system, f0_init, HD)
t_PD, res_PD = solve(TN_linear_system, f0_init, PD)

# plt.figure(figsize=(7,5))
# plt.plot(t_HD, res_HD[:,0], label=r"$f_0$ (Defect)", color="red")
# plt.plot(t_HD, res_HD[:,1], label=r"$f_1$ (Hybrid)", color="orange")
# plt.plot(t_HD, res_HD[:,2], label=r"$f_2$ (Cooperate)", color="green")

# plt.title("Linear Model (HD game)")
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 1)
# plt.show()


plt.figure(figsize=(7,5))
plt.plot(t_PD, res_PD[:,0], label=r"$f_0$ (Defect)", color="red")
plt.plot(t_PD, res_PD[:,1], label=r"$f_1$ (Hybrid)", color="orange")
plt.plot(t_PD, res_PD[:,2], label=r"$f_2$ (Cooperate)", color="green")

plt.title("Linear Model (PD game)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
