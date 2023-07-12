import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.integrate import solve_ivp

sns.set_theme(style="whitegrid")

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


def SEIHTR_forward(t_array,
    y0,
    N,
    beta,
    b,
    epsilon,
    sigma,
    r1,
    gamma,
    eta,
    alpha,
    mu1,
    r2,
    r3,
    omega,
    mu2,
    delta
):

    def func(x,y):
        s, e, i, h, t, r = y
        dSdt = - beta * s * i * (1-b) - epsilon * beta * s * e * (1-b)
        dEdt = beta * s * i * (1-b) + epsilon * beta * s * e * (1-b) - sigma * e + r1 * r
        dIdt = sigma * e - gamma * i - eta * i - alpha * i - mu1 * i + r2 * h + r3 * t
        dHdt = eta * i - r2 * h - omega * h - mu2 * h
        dTdt = alpha * i - r3 * t - delta * t
        dRdt = delta * t + gamma * i + omega * h - r1 * r
        return np.array([dSdt, dEdt, dIdt, dHdt, dTdt, dRdt])

    t_span = (t_array[0], t_array[-1])
    sol = solve_ivp(func, t_span, np.array(y0)/N, dense_output=True)
    return sol.sol(t_array).T

st.set_page_config(
    page_title = "Patient Addiction Journey Models",
    page_icon="🧬",
    layout="wide",
)

st.title("Patient Addiction Journey Models")

tab_SEIHR, tab_SEIHTR = st.tabs(["SEIHR", "SEIHTR"])

with tab_SEIHR:
    st.latex(r"""
    \begin{align}
        \dfrac{dS}{dt} &= -\frac{\beta S I}{N}  (1-b) - \frac{\epsilon \beta S E}{N} (1-b)\\
        \dfrac{dE}{dt} &= \frac{\beta S I}{N} (1-b) + \frac{\epsilon \beta S E}{N}(1-b) -\sigma E + r_{1} R\\
        \dfrac{dI}{dt} &= \sigma E - \gamma I - \eta I - \mu_{1} I + r_{2} H  \\
        \dfrac{dH}{dt} &= \eta I - \omega H - \mu_{2} H - r_2{H} \\
        \dfrac{dR}{dt} &= \gamma I + \omega H - r_{1} R \\
    \end{align} 
    """
    )

    col11, col12, col13, col14, col15, col16, col17, col18, col19, col110, col111, col112, col113 = st.columns(13)

    N = col11.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        help="Population",
    )

    n_days = col12.number_input(
        "Number of days",
        min_value=30,
        max_value=366,
        value=60,
        step=1,
        format=None,
        help="Number of days for simulation",
    )

    beta = col13.number_input(
        "Beta",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format=None,
        help="Transmission rate"
    )

    b = col14.number_input(
        "b",
        min_value=0.001,
        max_value=1.0,
        value=0.0,
        step=0.01,
        format=None,
        help="Measure of human behavior (1 is good, 0 is bad)"
    )

    epsilon = col15.number_input(
        "Epsilon",
        min_value=0.001,
        max_value=1.0,
        value=0.7,
        step=0.01,
        format=None,
        help="Infection rate between a susceptible and exposed individual"
    )

    r1 = col16.number_input(
        st.latex(r"$\r_{1}$"),
        min_value=0.001,
        max_value=1.0,
        value=0.4,
        step=0.01,
        format=None,
        help="Relapse rate for recovered individuals"
    )

    r2 = col17.number_input(
        st.latex(r"$\r_{2}$"),
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format=None,
        help="Relapse rate for hospitalized individuals"
    )

    sigma = col16.number_input(
        st.latex("Sigma"),
        min_value=0.001,
        max_value=1.0,
        value=0.4,
        step=0.01,
        format=None,
        help="Relapse rate for recovered individuals"
    )






