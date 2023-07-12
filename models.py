from scipy.integrate import solve_ivp
import numpy as np

def SEIHR_forward(
    t_array,
    y0,
    N,
    beta,
    b,
    epsilon,
    r1,
    r2,
    sigma,
    gamma,
    eta,
    omega,
    mu1,
    mu2
):

    def func(t, y):
        S, E, I, H, R = y
        dS_dt = -beta * S * I * (1-b)  - epsilon * beta * S * E * (1-b)
        dE_dt = beta * S * I * (1-b) + epsilon * beta * S * E * (1-b) - sigma * E + r1 * R
        dI_dt = sigma * E - gamma * I - eta * I - mu1 * I + r2 * H
        dH_dt = eta * I - omega * H - mu2 * H - r2 * H
        dR_dt = gamma * I + omega * H - r1 * R

        return np.array([dS_dt, dE_dt, dI_dt, dH_dt, dR_dt])

    t_span = (t_array[0], t_array[-1])
    sol = solve_ivp(func, t_span, np.array(y0)/N, dense_output=True)
    return sol.sol(t_array).T