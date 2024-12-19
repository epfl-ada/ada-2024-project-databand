import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
import json
import geopandas as gpd

# Set up paths
root_dir = Path(__file__).parent.parent.parent  # Go up two levels from the current file
sys.path.append(str(root_dir))

# Import custom modules
from src.data.process_data import create_tmdb_dataset
from src.utils.load_data import load_raw_data
from src.utils.data_utils import *
from src.utils.plot_utils import *

# Clean raw data and load the processed dataset
df = create_tmdb_dataset('data/processed/TMDB_clean.csv')
df_filtered = df[df['revenue'] > 0]

# Prepare data: ensure positive values and proper types
plot_data = df_filtered.copy()
plot_data['revenue'] = plot_data['revenue'].astype(float)
plot_data['log_revenue'] = np.log10(plot_data['revenue'])

# Page configuration
st.set_page_config(
    page_title="The Evolution of the Film Industry",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and Introduction
st.title("The Evolution of the Film Industry: The Impact of DVDs and Beyond")

st.markdown("""
The film industry has undergone seismic shifts over the past few decades, driven largely by changes in distribution models. Among the most notable innovations was the emergence of **DVDs in the 1990s**. DVDs revolutionized the accessibility of movies, providing a new revenue stream for production companies and reshaping both the business and creative landscapes of filmmaking. 

This narrative explores these transformations through an analysis of key differences observed in data from various eras of the film industry, focusing on:
- Revenue
- Budgets
- Production dynamics
- Genre evolution
""")

# Revenue Section
st.header("Revenue: A Changing Landscape")

# Subsection: DVDs as a Revenue Catalyst
st.subheader("DVDs as a Revenue Catalyst")
st.markdown("""
The introduction of DVDs provided a secondary revenue stream for production companies. Many movies that struggled at the box office found profitability in home entertainment. This diversification of revenue sources allowed for:
- **Recovery Opportunities**: Underperforming films could achieve financial success through strong DVD sales.
- **Broader Accessibility**: DVDs expanded the reach of films, particularly benefiting family-friendly and niche genres.
""")

# Layout with columns for explanatory text and plot
col1, col2 = st.columns([1, 2])

# Column 1: Explanatory text
with col1:
    st.subheader("Transition to Streaming")
    st.markdown("""
    The decline of DVD sales in the post-2013 era shifted revenue reliance back to theatrical releases and emerging digital platforms. Streaming services brought their own dynamics:
    - **Licensing Revenue**: Studios increasingly depended on deals with platforms like Netflix and Amazon Prime.
    - **Global Appeal**: Revenue strategies prioritized mainstream, blockbuster genres with universal themes.
    """)

# Column 2: Revenue distribution plot
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

    # Update layout for better visualization
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

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Budget Section
st.header("Budget Dynamics: Leveling the Playing Field")
st.subheader("Empowerment Through DVDs")
st.markdown("""
By providing an alternate revenue stream, the rise of DVDs at their peak allowed smaller production companies to enter the market.
Indeed, small studios could directly distribute their movies as DVDs, without the hassle and expense of going for a theatrical release. 
""")
st.markdown("""
To find evidence of this trend, we define 4 types of production companies using the mean budget as reference:
- **Independent** (<0.1x mean)
- **Small** (<1x mean)
- **Big** (>1x mean)
- **Super** (>5x mean)

Let’s take a look at how these productions evolve over time:
""")
prop_rolling_avg = budget_rolling_averages(df_filtered, window=3)

# Create individual plots for each category
def create_category_plot(data, category, label, color):
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[category],
            name=label,
            line=dict(width=2, color=color),
            hovertemplate="Year: %{x}<br>" +
                        f"Proportion: %{{y:.1%}}<br>" +
                        "<extra></extra>"
        )
    )
    
    fig.update_layout(
        title={
            'text': f'Proportion of {category.lower()} productions over the years (3-year rolling average)',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Release year',
        yaxis_title='Proportion',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(t=60, b=50, l=50, r=50)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        dtick=5
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    return fig

# Define categories and their properties
categories = [
    ('Independent', 'Independent productions (<0.1x mean budget)', '#9B59B6'),
    ('Small', 'Small productions (<1x mean budget)', '#E74C3C'),
    ('Big', 'Big productions (>1x mean budget)', '#28B463'),
    ('Super', 'Super productions (>5x mean budget)', '#2E86C1')
]

# Create and display each plot
for category, label, color in categories:
    fig = create_category_plot(prop_rolling_avg, category, label, color)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a small space between plots
    st.write("")
st.markdown("---")
st.write("""
Indeed! We can see a clear rise in movies from Independent production companies after DVDs hit the scene, 
with small productions maintaining a strong presence right up to their golden era. On the flip side, 
big production movie releases have been steadily tapering off since the DVD revolution.
""")
st.subheader("Post-DVD Challenges")
st.write("""
However, the fall of DVDs reshaped budget allocations once again… 
While Small production companies remained consistently prevalent during the DVD era, their popularity took a slight dark turn when DVD sales started declining. It seems that, without a viable secondary market, these projects struggled to find footing. 
In contrast, the peak DVD era was also marked with a slight boost in Super production companies, which maintained their influence in the movie industry after DVDs’ downfall. Indeed, some studios focused on large-scale productions with global earning potential.
Let’s summarize all this in a plot: if we compare pre- and post-DVD eras, we can clearly observe that the rise of DVDs was related to the emergence of many more Independent production movies at the cost of Big production movies, and with a small increase in Super production movies.   

Let’s summarize all this in a plot: if we compare pre- and post-DVD eras, we can clearly observe that the rise of DVDs was related to the emergence of many more Independent production movies at the cost of Big production movies, and with a small increase in Super production movies.    
""")
mean_budgets = df[df.budget > 0].groupby('release_year').agg(mean_budget = ('budget', 'mean'))
df['prod_type'] = df.apply(categorize_production, axis=1, args=(mean_budgets,))
df['dvd_era'] = pd.Categorical(df['dvd_era'], categories=['pre', 'during', 'post'], ordered=True)
df['prod_type'] = pd.Categorical(df['prod_type'], categories=['Independent', 'Small', 'Big', 'Super'], ordered=True)


# Filter data
df_filtered = df[df.budget > 0]

# Calculate proportions
props = (df_filtered.groupby('dvd_era')['prod_type']
         .value_counts(normalize=True)
         .unstack())

# Create stacked bar plot
fig = go.Figure()

# Color palette similar to tab20
colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
    '#2ca02c', '#98df8a', '#d62728', '#ff9896',
    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
    '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]

# Add bars for each production type
for i, prod_type in enumerate(props.columns):
    fig.add_trace(go.Bar(
        name=prod_type,
        x=props.index,
        y=props[prod_type],
        marker_color=colors[i % len(colors)]
    ))

# Update layout
fig.update_layout(
    barmode='relative',
    title={
        'text': 'Production Type Proportions Across DVD Eras',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='DVD Era',
    yaxis_title='Proportion',
    yaxis_tickformat='.0%',
    legend_title='Production Types',
    legend=dict(
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    ),
    template='plotly_white',
    height=600,
    margin=dict(r=150)  # Add right margin for legend
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

st.subheader("Impact globally: ")
st.write("""But was the movie industry impacted the same way globally? 
Let's focus on the production countries of our movies. Since there are many, we group them into main global regions:  
""")

def plot_world_map(countries_regions):
    try:
        # Updated file path to match your structure
        world = gpd.read_file('data/website_data/ne_110m_admin_0_countries.shp', encoding='utf-8')
        world['SOVEREIGNT'] = world['SOVEREIGNT'].str.lower()
        
        # Add region column
        world['region'] = world['SOVEREIGNT'].map(countries_regions)
        
        # Create Plotly choropleth map
        fig = px.choropleth(
            world.dropna(subset=['region']),
            geojson=world.geometry,
            locations=world.index,
            color='region',
            hover_name='SOVEREIGNT',
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        
        # Update layout
        fig.update_geos(
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            showframe=False,
            projection_type="equirectangular"
        )
        
        fig.update_layout(
            title={
                'text': 'Production Countries Map Colored by Region',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=600,
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            legend_title_text='World Regions',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            template='plotly_white'  # Match your website's theme
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading the map: {str(e)}")
        st.write("Please ensure the shapefile is in the correct location: data/ne_110m_admin_0_countries.shp")

# Load and display the map
try:
    # Updated file path to match your structure
    with open('data/website_data/countries_to_region.json', 'r') as file:
        countries_regions = json.load(file)
    plot_world_map(countries_regions)
except FileNotFoundError:
    st.error("Could not find the countries_to_region.json file. Please ensure it's in the data/website_data directory.")

st.write("""Interestingly, Eastern Asia and Europe show opposite trends in movie releases around the DVD era: 
Eastern Asian releases dipped slightly during this time, while European releases climbed. Meanwhile, 
North American movies, which have dominated since the mid-80s, hit their golden era in the late 1990s—just 
as DVDs emerged—and have seen a small but steady decline ever since.
""")