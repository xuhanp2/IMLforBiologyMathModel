import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

plt.rc('text', usetex=False)


# Parameters
N = 800
wi = 1
wg = 0.5
Lambda = 0.01
R, S, T, P = 6, 3, 8, 5  # PD  game
# wi = 0.3
# wg = 1
# Lambda = 0.1
# R, S, T, P = 6, 5, 12, 3  # HD  game
dt = 0.0005
time_steps = 10000

# fj = np.ones(N, dtype=float)
# # normalize so that ∫ f dx ≈ 1
# fj /= (np.sum(fj) / N)

def theta_init(j,N,theta):
	return N ** (1.0 - theta) * (((N - j) ** theta) - ((N - j - 1.0) ** theta) )
theta_vec = np.vectorize(theta_init)

index_holder = np.zeros(N)
for j in range(N):
	index_holder[j] = j
	
fj = np.ones(N)
fj = theta_vec(index_holder,N,1.0)


x_bdry = np.arange(N + 1) / N                  # cell boundaries (for flux)

def pi_C(x):
    return R * x + S * (1 - x)

def pi_D(x):
    return T * x + P * (1 - x)

def Pi_C(x):
    a = wi * pi_C(x)
    b = wi * pi_D(x)
    ea = np.exp(a)
    eb = np.exp(b)
    denom = x * ea + (1 - x) * eb
    return ea / denom

def Pi_D(x):
    a = wi * pi_C(x)
    b = wi * pi_D(x)
    ea = np.exp(a)
    eb = np.exp(b)
    denom = x * ea + (1 - x) * eb
    return eb / denom

def G(x):
    return x * pi_C(x) + (1 - x) * pi_D(x)

def curly_G(fj):
    numer = np.exp(wg * G(x_centers))
    dy = 1.0 / N
    denom = np.sum(fj * numer) * dy
    if denom == 0:
        denom = 1e-20
    return numer / denom

'''
np.arange(N) creates array [0, 1, 2, ..., N-1]
+ 0.5 shifts to cell centers
/ N scales to [0, 1]
'''
x_centers = (np.arange(N) + 0.5) / N 

def velocities(x):
    return x * (1 - x) * (Pi_C(x) - Pi_D(x))

'''
vb gives the velocities at cell boundaries
We use upwinding based on the sign of vb.
fj represents the cell-averaged density in each cell.
f_R shifts the fj values to the right boundaries of each cell. like x1(idx = 1 in f_R) has f1(idx = 1 in fj), x2 has f2, ...
f_L shifts the fj values to the left boundaries of each cell. like x1(idx = 1 in f_L) has f0(idx = 0 in fj), x2 has f1, ...
We then choose f_upwind based on the sign of vb.
'''
def within_grp_flux(fj):
    vb = velocities(x_bdry) # size N+1

    f_R = np.zeros(N + 1)
    f_L = np.zeros(N + 1)

    f_R[:-1] = fj  #f(t,x_(j+1)) = f_(j+1)
    f_R[-1] = 0.0  #value does not matter

    f_L[1:] = fj  #f(t,x_(j+1)) = f_j
    f_L[0] = 0.0  #value does not matter

    # upwind: if v>=0 use left, else use right
    fb = np.where(vb >= 0, f_L, f_R)
    fluxes = vb * fb   # size N+1

    return (-fluxes[1:] + fluxes[:-1]) * N  

def between_group_term(fj):
    return (Lambda / N) * fj * (curly_G(fj) - 1)
    # return (Lambda / N) * fj * (curly_G(fj) - np.sum(curly_G(fj) * fj)) #Question: should there be a /N (dx) here?


for time in range(time_steps):
    rhs = within_grp_flux(fj) + between_group_term(fj)
    fj = fj + dt * rhs

    if time % 1000 == 0:
        plt.plot(
            x_centers, fj,
            color=cmap.jet((float(time) / time_steps) ** 0.25),
            lw=3.0
        )

idx = np.argmax(fj)
x_peak_grid = x_centers[idx]
f_peak_grid = fj[idx]
print("Peak at x = ", x_peak_grid, " with f = ", f_peak_grid)

plt.xlabel('Fraction of Cooperators, $x$', fontsize=16)
plt.ylabel('Population Density, $f(t,x)$', fontsize=16)
plt.tight_layout()
plt.show()

