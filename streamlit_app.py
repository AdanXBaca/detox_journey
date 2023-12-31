import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from models import *

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title = "Patient Addiction Journey Models",
    page_icon="🧬",
    layout="wide",
)

mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''

st.markdown(mystyle, unsafe_allow_html=True)

st.title("Patient Addiction Journey Models")

tab_SEIHR, tab_SEIHTR, tab_contact = st.tabs(["SEIHR", "SEIHTR", "Contact Info"])

with tab_SEIHR:
    st.markdown('''The following model represents the jouney of addiction
    patients, as a population. Each compartment represents a state in the
    journey. S represents susceptible, which could be anyone. E is an 
    exposed state, and I is infected. H represents hospitalization,
    which could be the result of overdose. Finally, R represents recovery
    from an addiction.
    ''')
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

    col11, col12, col13, col14, col15, col16 = st.columns(6)

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
        "Number of Days",
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
        min_value=0.0,
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
        # st.latex(r"$\r_{1}$"),
        "r1",
        min_value=0.001,
        max_value=1.0,
        value=0.4,
        step=0.01,
        format=None,
        help="Relapse rate for recovered individuals"
    )

    col21, col22, col23, col24, col25, col26, col27 = st.columns(7)

    r2 = col21.number_input(
        # st.latex(r"$\r_{2}$"),
        "r2",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format=None,
        help="Relapse rate for hospitalized individuals"
    )

    sigma = col22.number_input(
        "Sigma",
        min_value=0.001,
        max_value=1.0,
        value=0.9,
        step=0.01,
        format=None,
        help="Infection rate for exposed"
    )

    gamma = col23.number_input(
        "Gamma",
        min_value=0.001,
        max_value=1.0,
        value=1/14,
        step=0.01,
        format=None,
        help="Rate of infected individuals who recover"
    )

    eta = col24.number_input(
        "Eta",
        min_value=0.001,
        max_value=1.0,
        value=1/21,
        step=0.01,
        format=None,
        help="Rate of infected individuals who become hospitalized"
    )

    omega = col25.number_input(
        "Omega",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format=None,
        help="Rate of hospitalized patients who recover"
    )

    mu1 = col26.number_input(
        # st.latex(r"$\mu_{1}"),
        "mu1",
        min_value=0.001,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format=None,
        help="Death rate for infected"
    )

    mu2 = col27.number_input(
        # st.latex(r"$\mu_{2}"),
        "mu2",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.01,
        format=None,
        help="Death rate for hospitalized"
    )

    if st.button("Run forward simulation"):
        with st.spinner("Wait for it..."):
            t_train = np.arange(0, n_days, 1)
            parameters = {
                "beta": beta,
                "b":b,
                "epsilon":epsilon,
                "r1":r1,
                "r2":r2,
                "sigma":sigma,
                "gamma":gamma,
                "eta":eta,
                "omega":omega,
                "mu1":mu1,
                "mu2":mu2
            }
            S_0 = N - 51
            E_0 = 50
            I_0 = 1
            H_0 = 0
            R_0 = 0
            y0 = [S_0, E_0, I_0, H_0, R_0]
            y_sol = SEIHR_forward(
                t_train,
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
            )

            model_name = "SEIHR"
            population_names = list(model_name)
            data_real = (
                pd.DataFrame(y_sol, 
                columns=population_names).assign(time=t_train).melt(
                    id_vars="time", var_name="status", value_name="population")
                )
            
            fig, ax = plt.subplots(figsize = (10,4))
            sns.lineplot(
                data = data_real,
                x = "time",
                y = "population",
                hue = "status",
                legend = True,
                linestyle = "dashed",
                ax=ax
            )
            ax.set_title(f"{model_name} model - Forward Simulation")
            st.pyplot(fig)


with tab_contact:
    st.markdown("Made by Adan Baca, Diego Gonzales, Alonso Ogueda, and Padhu Seshaiyer.")
    st.markdown("Contact Info: \n Adan Baca: adanbaca@arizona.edu \n Diego Gonzalez: diego.gonzalez@laverne.edu")
            








