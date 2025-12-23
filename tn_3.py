import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy.linalg as LA
import numpy as np

R, S, T, P = 6, 5, 12, 3 #HD
# R, S, T, P = 6, 3, 12, 5 #PD
omega_I = 0.1
omega_G = 15
Lambda = 0.1

def pi_C(x):
    return x*R + (1-x)*S

def pi_D(x):
    return x*T + (1-x)*P

def G(x):
    return x*pi_C(x) + (1-x)*pi_D(x)

W1 = np.exp(omega_I * (R+2*S)/3) + 2*np.exp(omega_I * (T+2*P)/3)
W2 = 2*np.exp(omega_I * (2*R+S)/3) + np.exp(omega_I * (2*T+P)/3)
def dfdt(f):
    f0, f1, f2, f3 = f

    D = (
        f0*np.exp(omega_G*P)
        + f1*np.exp(omega_G*(R+2*S+2*T+4*P)/9)
        + f2*np.exp(omega_G*(4*R+2*S+2*T+P)/9)
        + f3*np.exp(omega_G*R)
    )

    df0 = (
        f1 * (2*np.exp(omega_I*pi_D(1/3)) / W1)
        + Lambda * f0 * (np.exp(omega_G*P)/D - 1)
    )
    df1 = (
        -f1*(2*np.exp(omega_I*(T+2*P)/3)/W1)
        -f1*(2*np.exp(omega_I*(R+2*S)/3)/W1)
        + f2*(2*np.exp(omega_I*(2*T+P)/3)/W2)
        + Lambda*f1*(np.exp(omega_G*(R+2*S+2*T+4*P)/9)/D - 1)
    )

    df2 = (
        -f2*(2*np.exp(omega_I*(2*T+P)/3)/W2)
        -f2*(2*np.exp(omega_I*(2*R+S)/3)/W2)
        + f1*(2*np.exp(omega_I*(R+2*S)/3)/W1)
        + Lambda*f2*(np.exp(omega_G*(4*R+2*S+2*T+P)/9)/D - 1)
    )

    # df3
    df3 = (
        f2*(2*np.exp(omega_I*(2*R+S)/3)/W2)
        + Lambda*f3*(np.exp(omega_G*R)/D - 1)
    )

    return np.array([df0, df1, df2, df3])

def rk4_step(f, dt):
    k1 = dfdt(f)
    k2 = dfdt(f + dt*k1/2)
    k3 = dfdt(f + dt*k2/2)
    k4 = dfdt(f + dt*k3)
    f_new = f + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return f_new / np.sum(f_new)


Tmax = 100
dt = 0.001
steps = int(Tmax/dt)
f = np.array([0.25, 0.25, 0.25, 0.25])
history = np.zeros((steps, 4))
time = np.linspace(0, Tmax, steps)

for i in range(steps):
    history[i] = f
    f = rk4_step(f, dt)


plt.figure(figsize=(8,5))
plt.plot(time, history[:,0], label="f0")
plt.plot(time, history[:,1], label="f1")
plt.plot(time, history[:,2], label="f2")
plt.plot(time, history[:,3], label="f3")
plt.xlabel("Time")
plt.ylabel("f_k(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#eq points
def equilibrium_equations(f):
    f0, f1, f2, f3 = f
    S = f0 + f1 + f2 + f3
    if S == 0:
        return np.array([1,1,1,1]) 
    f = f / S
    return dfdt(f)
f_guess = np.array([0.25, 0.25, 0.25, 0.25])
eq_f, info, ier, msg = fsolve(equilibrium_equations, f_guess, full_output=True)
eq_f = eq_f / np.sum(eq_f)
print("Equilibrium point:")
print(eq_f, "\n")

#stability
def jacobian(f, eps=1e-6):
    n = len(f)
    J = np.zeros((n,n))
    F0 = dfdt(f)
    
    for j in range(n):
        f_perturb = f.copy()
        f_perturb[j] += eps
        f_perturb /= np.sum(f_perturb)
        F1 = dfdt(f_perturb)
        J[:, j] = (F1 - F0) / eps

    return J
J_eq = jacobian(eq_f)
# print("Jacobian at equilibrium:")
# print(J_eq, "\n")
eigvals, eigvecs = LA.eig(J_eq)
print("Eigenvalues of Jacobian:")
print(eigvals, "\n")

if np.all(np.real(eigvals) < 0):
    print("=> Equilibrium is STABLE.")
elif np.any(np.real(eigvals) > 0):
    print("=> Equilibrium is UNSTABLE.")
else:
    print("=> Equilibrium is NON-HYPERBOLIC (zero real parts).")


f0, f1, f2, f3 = eq_f 
D_eq = (
      f0 * np.exp(omega_G * P)
    + f1 * np.exp(omega_G * (R + 2*S + 2*T + 4*P) / 9)
    + f2 * np.exp(omega_G * (4*R + 2*S + 2*T + P) / 9)
    + f3 * np.exp(omega_G * R)
)
print(D_eq)
print(np.exp(omega_G * R))
print(np.exp(omega_G * P))

