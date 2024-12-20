import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from itertools import combinations
import numpy as np
import geopandas as gpd
from plotly.subplots import make_subplots
import pandas as pd

from src.utils.data_utils import categorize_budget


from plotly.subplots import make_subplots
import plotly.graph_objects as go

## Color Palette (Viridis)

palette_seq = px.colors.sequential.Viridis
palette_for_empath = plt.cm.viridis

def get_color_palette(categories):
    """Generate a consistent color mapping using viridis."""
    palette = sns.color_palette("viridis", len(categories))
    return {category: f"rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, 1)" 
            for category, color in zip(categories, palette)}


def return_movies_prop_per_region(region_props):
    """Plot proportion of movies released over time per region."""
    # Get consistent colors for regions
    unique_regions = region_props['region'].unique()
    colors = get_color_palette(unique_regions)
    
    fig = px.line(
        region_props[region_props.prop > 0.01],
        x='release_year',
        y='prop',
        color='region',
        color_discrete_map=colors,
        title='Proportion of movies released over time per region',
        labels={'release_year': 'Release Year', 'prop': 'Proportion'}
    )
    fig.update_layout(
        legend_title='Region',
        xaxis_title='Release Year',
        yaxis_title='Proportion',
        legend=dict(x=1.05, y=0.5, orientation='v', traceorder='normal'),
        margin=dict(r=200)  # Add space for legend
    )
    return fig


def return_prod_type_prop_per_region(selected_regions, df_countries_filtered):
    """Plot production type proportions for selected regions."""
    # Define production types and consistent colors
    prod_types = ['Independent', 'Small', 'Big', 'Super']
    colors = get_color_palette(prod_types)

    # Initialize the figure
    fig = make_subplots(
        rows=1,
        cols=len(selected_regions),
        shared_yaxes=True,
        subplot_titles=selected_regions
    )

    # Add a bar trace for each production type in each region
    for col_idx, region in enumerate(selected_regions, start=1):
        df_region = df_countries_filtered[df_countries_filtered['region'] == region]
        for prod_type in prod_types:
            df_prod = df_region[df_region['prod_type'] == prod_type]
            proportions = df_prod.groupby('dvd_era').size() / len(df_region)
            
            # Add trace
            fig.add_trace(
                go.Bar(
                    x=proportions.index,
                    y=proportions.values,
                    name=prod_type,  # Proper label for legend
                    marker_color=colors[prod_type],
                    showlegend=(col_idx == 1)  # Show legend only for the first column
                ),
                row=1,
                col=col_idx
            )

    # Update layout
    fig.update_layout(
        title="Production Type Proportions for Major Regions",
        barmode='stack',
        height=600,
        width=1200,
        legend_title="Production Type",
        legend=dict(x=1.05, y=0.5, orientation='v', traceorder='normal'),  # Move legend outside
        showlegend=True,
        margin=dict(r=200)  # Add space for legend
    )

    # Adjust subplot titles and spacing
    fig.update_annotations(font_size=12)
    fig.update_xaxes(title_text="DVD Era")
    fig.update_yaxes(title_text="Proportion", col=1)

    return fig


def return_genre_prop_per_region(selected_regions, countries_genres_props):
    """Plot genre proportions per region."""
    print("Creating genre proportions per region plot...")
    
    # Get consistent colors for genres
    unique_genres = countries_genres_props['genres'].unique()
    colors = get_color_palette(unique_genres)

    # Create subplots with one row and columns equal to the number of regions
    fig = make_subplots(
        rows=1, cols=len(selected_regions),
        shared_yaxes=True,
        subplot_titles=selected_regions
    )

    # Loop over selected regions to create a line plot for each
    for i, region in enumerate(selected_regions):
        subset = countries_genres_props[countries_genres_props['region'] == region]
        
        # Create a line plot for each genre in the subset
        for genre in subset['genres'].unique():
            genre_subset = subset[subset['genres'] == genre]
            
            # Add a trace for the genre within the current region's subplot
            fig.add_trace(
                go.Scatter(
                    x=genre_subset['dvd_era'],
                    y=genre_subset['prop'],
                    mode='lines+markers',
                    name=genre,
                    line=dict(color=colors[genre], shape='linear'),
                    marker=dict(symbol='circle'),
                    showlegend=(i == 0)  # Show legend only for the first subplot
                ),
                row=1, col=i+1
            )

        # Set x-axis label for each subplot
        fig.update_xaxes(title_text="DVD Era", row=1, col=i+1)

    # Update the shared y-axis label
    fig.update_yaxes(title_text="Proportion", row=1, col=1)

    # Update layout for titles, appearance, and legend
    fig.update_layout(
        title='Genre Proportions per Region',
        height=600,
        width=1200,
        showlegend=True,
        legend_title='Genres',
        legend=dict(
            x=1.05, y=0.5, traceorder='normal',
            title='Genres', xanchor='left', yanchor='middle'
        ),
        margin=dict(r=200)  # Add space for the legend
    )

    return fig


def return_genre_prop_by_prod_type(genre_proportions):
    """Plot genre proportions by production type."""
    # Group and calculate proportions
    grouped_genres = genre_proportions.groupby(['prod_type', 'genres'], observed=False).sum('count').reset_index()
    grouped_genres['proportion'] = grouped_genres['count'] / grouped_genres['total']

    # Create a pivot table to get proportions by production type and genre
    pivot_data = grouped_genres.pivot_table(
        index='prod_type', columns='genres', values='proportion', fill_value=0, observed=False
    )

    # Get consistent colors for genres
    unique_genres = pivot_data.columns
    colors = get_color_palette(unique_genres)

    # Create a stacked bar plot
    fig = go.Figure()

    # Add a bar trace for each genre
    for genre in unique_genres:
        fig.add_trace(go.Bar(
            x=pivot_data.index,
            y=pivot_data[genre],
            name=genre,
            marker_color=colors[genre]
        ))

    # Update layout with titles and legends
    fig.update_layout(
        title="Genre Proportions by Production Type",
        xaxis_title="Production Type",
        yaxis_title="Proportion",
        barmode='stack',
        legend_title="Genres",
        legend=dict(
            x=1.05, y=0.5, traceorder='normal',
            title='Genres', xanchor='left', yanchor='middle'
        ),
        height=600,
        width=1200,
        margin=dict(r=200)  # Add space for the legend
    )

    return fig

def return_genre_trends_by_prod_type(genre_proportions):
    """Plot genre trends by production type using viridis colors."""
    prod_types = genre_proportions['prod_type'].unique()
    colors = get_color_palette(genre_proportions['genres'].unique())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=prod_types,
        shared_yaxes=True
    )

    for i, prod_type in enumerate(prod_types):
        subset = genre_proportions[genre_proportions['prod_type'] == prod_type]
        for genre in subset['genres'].unique():
            genre_subset = subset[subset['genres'] == genre]
            fig.add_trace(
                go.Scatter(
                    x=genre_subset['dvd_era'],
                    y=genre_subset['prop'],
                    mode='lines+markers',
                    name=genre,
                    line=dict(color=colors[genre], shape='linear'),
                    marker=dict(symbol='circle'),
                    showlegend=(i == 0)  # Show legend only in the first subplot
                ),
                row=(i // 2) + 1, col=(i % 2) + 1
            )

    fig.update_layout(
        title="Genre Proportions Across DVD Eras",
        height=800,
        width=1000,
        legend=dict(
            x=1.05, y=0.5,
            title="Genres",
            xanchor="left", yanchor="middle"
        ),
        margin=dict(r=200)
    )
    fig.update_xaxes(title_text="DVD Era")
    fig.update_yaxes(title_text="Proportion")
    return fig

def create_edges_list(df):
    edges = []
    for companies in df['production_companies']:
        if len(companies) > 1:
            edges.extend(list(combinations(companies, 2)))
    return edges

def return_collaborations(df):
    """Create a collaboration network plot."""
    before_edges = create_edges_list(df)
    G_before = nx.Graph()
    G_before.add_edges_from(before_edges)
    pos = nx.spring_layout(G_before)

    edge_x, edge_y = [], []
    for edge in G_before.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y = zip(*pos.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G_before.nodes),
        textposition='bottom center',
        marker=dict(
            size=15,
            color='lightblue',
            line=dict(width=1, color='black')
        )
    ))

    fig.update_layout(
        title="Production Company Collaboration Network",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white',
        height=600,
        width=800
    )
    return fig


def return_avg_num_prod_companies(yearly_avg_companies):
    """Plot average number of production companies and a trend line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_avg_companies['release_year'],
        y=yearly_avg_companies['production_companies'],
        mode='lines+markers',
        name='Average Production Companies',
        line=dict(width=2, color=sns.color_palette("viridis", 1).as_hex()[0])
    ))

    z = np.polyfit(yearly_avg_companies['release_year'], yearly_avg_companies['production_companies'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=yearly_avg_companies['release_year'],
        y=p(yearly_avg_companies['release_year']),
        mode='lines',
        name=f'Trend line (slope: {z[0]:.3f})',
        line=dict(dash='dash', color='red')
    ))

    fig.update_layout(
        title="Average Number of Production Companies Over Time",
        xaxis_title="Release Year",
        yaxis_title="Average Number of Production Companies",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(
            x=1.05, y=0.5,
            title="Legend",
            xanchor="left", yanchor="middle"
        ),
        margin=dict(r=200)
    )

    corr_matrix = yearly_avg_companies.corr()
    heatmap_fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='viridis',
        title='Correlation Matrix',
        labels={'x': 'Features', 'y': 'Features'}
    )
    heatmap_fig.update_layout(height=500)
    return fig, heatmap_fig


def return_num_companies_per_prod_type(df):
    fig = go.Figure()

    # Define production types
    production_types = ['Super', 'Big', 'Small', 'Independent']
    
    # Get the color palette for the production types
    colors = get_color_palette(production_types)

    df_graph = df[df['production_companies'].str.len() > 0]
    
    for prod_type in production_types:
        yearly_data = []
        years = sorted(df_graph['release_year'].unique())

        for year in years:
            df_filtered = df_graph[
                (df_graph['release_year'] == year) & 
                (df_graph['prod_type'] == prod_type)
            ]
            total_movies = len(df_filtered)
            if total_movies > 0:
                edges = create_edges_list(df_filtered)
                G = nx.Graph()
                G.add_edges_from(edges)
                avg_collaborations = G.number_of_edges() / total_movies if G.number_of_nodes() > 0 else 0
                yearly_data.append((year, avg_collaborations))
        
        if yearly_data:
            years, avg_collabs = zip(*yearly_data)
            fig.add_trace(go.Scatter(
                x=years, y=avg_collabs, mode='lines+markers', 
                name=prod_type, line=dict(color=colors[prod_type], width=2)
            ))

    # Customize plot appearance
    fig.update_layout(
        title='Average Number of Collaborations per Movie Over Time by Production Type',
        xaxis_title='Year',
        yaxis_title='Average Collaborations per Movie',
        title_x=0.5,
        template='plotly_white',
        showlegend=True,
        legend_title='Production Type',
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', 
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', 
            zeroline=False
        ),
        annotations=[  # Add annotations for different eras
            dict(x=1997, y=1, xref="x", yref="y", text="Pre-DVD", showarrow=False, font=dict(size=12, color="gray")),
            dict(x=2001.5, y=1, xref="x", yref="y", text="DVD Era", showarrow=False, font=dict(size=12, color="gray")),
            dict(x=2015, y=1, xref="x", yref="y", text="Post-DVD", showarrow=False, font=dict(size=12, color="gray")),
        ],
        margin=dict(r=150),  # Add margin for legend to the right
    )

    # Add vertical lines for DVD era boundaries
    fig.add_vline(x=1997, line=dict(color='gray', dash='dash', width=2))
    fig.add_vline(x=2006, line=dict(color='gray', dash='dash', width=2))

    # Move the legend to the right of the plot
    fig.update_layout(
        legend=dict(
            title='Production Type', 
            x=1.05, y=1,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.5)', borderwidth=2
        )
    )
    
    return fig

# Utility function: Create edges list (you can define the logic of this function)
def create_edges_list(df_filtered):
    # Assuming the function creates a list of edges for the network graph based on production companies
    edges = []
    # Placeholder logic for creating edges (adjust as needed)
    for _, row in df_filtered.iterrows():
        companies = row['production_companies']
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                edges.append((companies[i], companies[j]))
    return edges


# Function: Create category plot (with rolling average and consistent color palette)
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
            x=1.05
        ),
        margin=dict(t=60, b=50, l=50, r=150)  # Adding extra margin for legend
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

def create_histogram(data, x_col, color_col, title, labels, color_palette="viridis", nbins=50):
    # Use color_discrete_sequence for discrete color palettes
    # Define the three extreme colors from the Viridis palette
    viridis_colors = [palette_seq[0], palette_seq[len(palette_seq)//2], palette_seq[-1]]

    fig = px.histogram(
        data,
        x=x_col,
        color=color_col,
        nbins=nbins,
        title=title,
        labels=labels,
        color_discrete_sequence=viridis_colors,  # Use the three extreme colors from the Viridis palette
        opacity=0.6
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title=labels.get(x_col, x_col),
        yaxis_title="Count",
        title_x=0.5,
        title_font_size=20,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.05),
        margin=dict(t=50, b=50, l=50, r=150),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    
    return fig

# Function: Create line plot with consistent color palette
def create_line_plot(data, x_col, y_col, color_col, title, labels, color_palette):
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        labels=labels,
        color_discrete_sequence=color_palette,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=labels.get(x_col, x_col),
        yaxis_title=labels.get(y_col, y_col),
        title_x=0.5,
        title_font_size=20,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.05),
        margin=dict(t=50, b=50, l=50, r=150),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    return fig



# Create stacked bar plot with Viridis color palette
def create_stacked_bar(props, title='Production Type Proportions Across DVD Eras', 
                       x_title='DVD Era', y_title='Proportion', legend_title='Production Types'):
    fig = go.Figure()

    # Use the Viridis color palette
    colors = get_color_palette(props.columns)

    # Add bars for each production type
    for i, prod_type in enumerate(props.columns):
        fig.add_trace(go.Bar(
            name=prod_type,
            x=props.index,
            y=props[prod_type],
            marker_color=colors[prod_type]  # Use color from the Viridis palette
        ))

    fig.update_layout(
        barmode='stack',  # Stacked bar mode
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_tickformat='.0%',
        legend_title=legend_title,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05  # Adjust position to the right of the plot
        ),
        template='plotly_white',
        height=600,
        margin=dict(r=150)  # Add right margin for legend
    )

    return fig

def create_world_map(countries_regions, shapefile_path):
    try:
        # Read the shapefile
        world = gpd.read_file(shapefile_path, encoding='utf-8')
        world['SOVEREIGNT'] = world['SOVEREIGNT'].str.lower()

        # Add region column
        world['region'] = world['SOVEREIGNT'].map(countries_regions)

        # Filter out NaN values and reset index
        world_filtered = world.dropna(subset=['region']).reset_index(drop=True)

        # Create Plotly choropleth map with a continuous viridis color scale
        fig = px.choropleth(
            world_filtered,
            geojson=world_filtered.geometry,
            locations=world_filtered.index,
            color='region',
            hover_name='SOVEREIGNT',
            color_continuous_scale='viridis'  # Use continuous viridis color scale
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
            height=500,
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            legend_title_text='World Regions',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            template='plotly_white'
        )

        return fig, None

    except Exception as e:
        return None, f"Error loading the map: {str(e)}"

# Utility function: Filter data for proportions
def calculate_proportions(df, base_vars, target_var):
    counts = df.groupby(base_vars)[target_var].value_counts(normalize=True).unstack()
    return counts.reset_index()

def create_regional_distribution_plot(region_prop, min_prop=0.01):
    # Filter regions by minimum proportion
    filtered_data = region_prop[region_prop.prop > min_prop]
    
    # Generate color palette
    regions = filtered_data['region'].unique()
    color_map = get_color_palette(regions)
    
    # Create the line plot
    fig = px.line(
        filtered_data, 
        x='release_year', 
        y='prop',
        color='region',
        title='Regional Distribution of Movie Production Over Time',
        labels={
            'release_year': 'Release Year',
            'prop': 'Proportion of Movies',
            'region': 'Region'
        },
        color_discrete_map=color_map  # Use consistent color map
    )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Regional Distribution of Movie Production Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Release Year',
        yaxis_title='Proportion of Movies',
        yaxis_tickformat='.0%',
        legend_title='Region',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.02  # Move legend to the right
        ),
        template='plotly_white',
        height=500,
        margin=dict(t=50, b=50, l=50, r=150)  # Add margin for legend
    )

    # Add gridlines
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

def create_regional_production_subplots(df_countries_filtered, selected_regions):
    """
    Create subplots showing production type proportions by region across DVD eras.
    
    Parameters:
    -----------
    df_countries_filtered : pandas.DataFrame
        Filtered DataFrame containing columns: 'region', 'dvd_era', 'prod_type'
    selected_regions : list
        List of regions to create subplots for
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure object with subplots
    """
    
    # Generate color palette for production types
    prod_types = ['Independent', 'Small', 'Big', 'Super']
    color_map = get_color_palette(prod_types)

    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=len(selected_regions),
        subplot_titles=selected_regions,
        shared_yaxes=True
    )

    # Create a plot for each region
    for i, region in enumerate(selected_regions):
        region_data = df_countries_filtered[df_countries_filtered['region'] == region]
        
        # Calculate proportions for each DVD era and production type
        props = (region_data.groupby('dvd_era')['prod_type']
                 .value_counts(normalize=True)
                 .unstack()
                 .fillna(0))
        
        # Ensure all production types are present
        for prod_type in prod_types:
            if prod_type not in props.columns:
                props[prod_type] = 0
        
        # Reorder columns
        props = props[prod_types]
        
        # Add bars for each production type
        for j, prod_type in enumerate(prod_types):
            fig.add_trace(
                go.Bar(
                    name=prod_type,
                    x=props.index,
                    y=props[prod_type],
                    legendgroup=prod_type,
                    showlegend=(i == 0),  # Show legend only for first region
                    marker_color=color_map[prod_type]
                ),
                row=1, 
                col=i+1
            )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Production Type Proportions by Region Across DVD Eras',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        barmode='relative',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.02,
            title="Production Type"
        ),
        template='plotly_white',
        margin=dict(t=100, r=150)  # Add right margin for legend
    )

    # Update axes
    for i in range(len(selected_regions)):
        fig.update_xaxes(title_text="DVD Era", row=1, col=i+1)
        if i == 0:  # Only add y-axis title to first subplot
            fig.update_yaxes(title_text="Proportion", row=1, col=1)

    # Update y-axes format
    fig.update_yaxes(tickformat='.0%')
    
    return fig

def budget_rolling_averages(df, window):
    budget_stats = df.groupby('release_year')['budget'].agg(mean_budget='mean').reset_index()
    df.loc[:, 'budget_category'] = df.apply(categorize_budget, args=(budget_stats,), axis=1)

    # Count the number of each budget category per year
    budget_category_counts = df.groupby(['release_year', 'budget_category']).size().unstack(fill_value=0)

    # Calculate the proportion of each budget category per year
    budget_category_proportions = budget_category_counts.div(budget_category_counts.sum(axis=1), axis=0)

    # Calculate the 3-year rolling average for each budget category
    proportion_rolling = budget_category_proportions.rolling(window=window, center=True).mean()
    return proportion_rolling


import plotly.graph_objects as go

def create_combined_plot(data, categories):
    """
    Create a single plot combining all production type trends.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the rolling averages
    categories : list
        List of tuples containing (category, label, color)
    """
    # Get a consistent color palette using viridis
    color_map = get_color_palette([category for category, _, _ in categories])
    
    fig = go.Figure()
    
    # Add a trace for each category
    for category, label, _ in categories:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[category],
                name=label,
                line=dict(width=2, color=color_map[category]),  # Color taken from viridis
                hovertemplate="Year: %{x}<br>" +
                            f"Proportion: %{{y:.1%}}<br>" +
                            "<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Production Type Proportions Over Time (3-year rolling average)',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Release Year',
        yaxis_title='Proportion',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=600,  # Made taller for better visibility
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.02,  # Move legend to the right of the plot
            bgcolor="rgba(255, 255, 255, 0.8)"  # Semi-transparent background for the legend
        ),
        margin=dict(t=80, b=50, l=50, r=150)  # Adjust right margin for legend space
    )
    
    # Add gridlines
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