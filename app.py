import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from app_helpers import plot_mean_budget_inflation, plot_movies_budget_slider

# Set page configuration
st.set_page_config(page_title="Analyse des Budgets des Films", layout="wide")

# Introduction Section
st.title("Analyse des Budgets des Films")
st.header("Introduction")
st.write("Ce tableau de bord interactif vous permet d'explorer l'évolution des budgets des films au fil du temps ainsi que d'interagir avec des visualisations avancées.")

# Evolution du budget des films au cours du temps
st.header("Evolution du budget des films au cours du temps")

# Load the figure
fig = plot_mean_budget_inflation()

# Display plot in Streamlit
st.pyplot(fig)

# Plots interactifs (plotly)
st.header("Plots interactifs (Plotly)")

# Load the slider plots
budg_slider = plot_movies_budget_slider(mode='budget')      # Budget mode (default)
relat_slider = plot_movies_budget_slider(mode='relative')    # Relative mode
perc_slider = plot_movies_budget_slider(mode='percentage')  # Percentage mode

# Display interactive plot

st.subheader("Budget mode")
st.plotly_chart(budg_slider)

st.subheader("Relative mode")
st.plotly_chart(relat_slider)

st.subheader("Percentage mode")
st.plotly_chart(perc_slider)