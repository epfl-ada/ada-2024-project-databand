import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# Set up paths
root_dir = Path(__file__).parent.parent.parent  # Go up two levels from the current file
sys.path.append(str(root_dir))

# Import custom modules
from src.data.process_data import create_tmdb_dataset
from src.utils.load_data import load_raw_data
from src.utils.data_utils import *
from src.utils.plot_utils import *
from src.models.lda_model import *
from src.scripts.clean_data import clean_raw_data

# Clean raw data and load the processed dataset
df = create_tmdb_dataset('data/processed/TMDB_clean.csv')
df_filtered = df[df['revenue'] > 0]

# Prepare data: ensure positive values and proper types
plot_data = df_filtered.copy()
plot_data = plot_data[plot_data['revenue'] > 0]  # Remove zero or negative values
plot_data['revenue'] = plot_data['revenue'].astype(float)
plot_data['log_revenue'] = np.log10(plot_data['revenue'])

# Page configuration
st.set_page_config(
    page_title="Film Industry Evolution",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and introduction using Streamlit's native components
st.title("The Evolution of the Film Industry: The Impact of DVDs and Beyond")

# Add content and layout
st.write("""
    The film industry has undergone seismic shifts over the past few decades, driven largely by changes in 
    distribution models. Among the most notable innovations was the emergence of DVDs in the 1990s. DVDs 
    revolutionized the accessibility of movies, providing a new revenue stream for production companies and 
    reshaping both the business and creative landscapes of filmmaking.

    This narrative explores these transformations through an analysis of key differences observed in data 
    from various eras of the film industry, focusing on revenue, budgets, production dynamics, and genre evolution.
""")

# Layout with columns for plot and explanatory text
col1, col2 = st.columns([1, 2])

with col2:
    fig = px.histogram(
        plot_data,
        x='log_revenue',
        color='dvd_era',
        nbins=50,
        title='Revenue Distribution Across DVD Eras',
        labels={'log_revenue': 'Revenue (log scale)', 'dvd_era': 'DVD Era'},
        color_discrete_sequence=['#2E86C1', '#28B463', '#E74C3C'],
        opacity=0.6
    )
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        xaxis_title='Revenue (log scale)',
        yaxis_title='Density',
        bargap=0.1,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(t=50, b=50, l=50, r=50),
        height=500
    )

    # Add gridlines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    st.plotly_chart(fig, use_container_width=True)

with col1:
    st.write("### The Role of DVDs in the Film Industry")
    st.write("""
    The introduction of DVDs provided a secondary revenue stream for production companies. Many movies that 
    struggled at the box office found profitability in home entertainment. This diversification of revenue sources allowed for:
    
    - **Recovery Opportunities**: Underperforming films could achieve financial success through strong DVD sales.
    - **Broader Accessibility**: DVDs expanded the reach of films, particularly benefiting family-friendly and niche genres.
    """)

st.header('Budget Dynamics: Leveling the Playing Field')



