import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy.linalg as LA

HD_params = dict(R=6, S=5, T=12, P=3)
PD_params = dict(R=6, S=3, T=7, P=5)
w_I,w_G,Lambda = 0.01, 0.01, 100   

def make_TN_system(R, S, T, P, w_I, w_G, Lambda):
    G_1 = (T + S) / 2.0

    def TN_system(t, f):
        f0, f1, f2 = f

        term0 = f0 * np.exp(w_G * P)
        term1 = f1 * np.exp(w_G * G_1)
        term2 = f2 * np.exp(w_G * R)
        D = term0 + term1 + term2
        if D == 0:
            D = 1e-10

        df0 = (0.5 * (np.exp(w_I * T) / (np.exp(w_I * T) + np.exp(w_I * S))) * f1
               + Lambda * f0 * ((np.exp(w_G * P) / D) - 1))

        df1 = (-0.5 * f1
               + Lambda * f1 * ((np.exp(w_G * G_1) / D) - 1))

        df2 = (0.5 * (np.exp(w_I * S) / (np.exp(w_I * T) + np.exp(w_I * S))) * f1
               + Lambda * f2 * ((np.exp(w_G * R) / D) - 1))

        return np.array([df0, df1, df2])

    return TN_system

def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = func(t + dt, y + dt*k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def solve_ode(system_func, y0, t_span, dt):
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_current = np.array(y0)
    for i, t in enumerate(t_values):
        y_values[i] = y_current
        y_next = rk4_step(system_func, t, y_current, dt)
        y_current = y_next / np.sum(y_next) 
    return t_values, y_values


f0_init = [0.33, 0.33, 0.34] 
t_span = (0, 10)
dt = 0.0001

TN_HD = make_TN_system(**HD_params, w_I=w_I, w_G=w_G, Lambda=Lambda)
TN_PD = make_TN_system(**PD_params, w_I=w_I, w_G=w_G, Lambda=Lambda)

t_HD, res_HD = solve_ode(TN_HD, f0_init, t_span, dt)
t_PD, res_PD = solve_ode(TN_PD, f0_init, t_span, dt)


plt.figure(figsize=(7,5))
plt.plot(t_HD, res_HD[:, 0], label='$f_0$ (Defect)', color='red')
plt.plot(t_HD, res_HD[:, 1], label='$f_1$ (Hybrid)', color='orange')
plt.plot(t_HD, res_HD[:, 2], label='$f_2$ (Coop)', color='green')

plt.title("TN Dynamics (HD Game)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(t_PD, res_PD[:, 0], label='$f_0$ (Defect)', color='red')
plt.plot(t_PD, res_PD[:, 1], label='$f_1$ (Hybrid)', color='orange')
plt.plot(t_PD, res_PD[:, 2], label='$f_2$ (Coop)', color='green')

plt.title("TN Dynamics (PD Game)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
