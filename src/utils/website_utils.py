import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from itertools import combinations
import numpy as np

def return_genre_prop_per_region(selected_regions, countries_genres_props):
    print("Creating genre proportions per region plot...")
    # Create subplots with one row and len(selected_regions) columns
    fig = make_subplots(
        rows=1, cols=len(selected_regions),
        shared_yaxes=True,  # All subplots will share the same y-axis
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
                    name=genre,  # Name for the legend
                    line=dict(shape='linear'),
                    marker=dict(symbol='circle')
                ),
                row=1, col=i+1
            )

        # Set x-axis label for each subplot
        fig.update_xaxes(
            title_text="DVD Era",
            row=1, col=i+1
        )

    # Update the shared y-axis label
    fig.update_yaxes(
        title_text="Proportion",
        row=1, col=1  # Only update the first subplot's y-axis
    )

    # Update layout for titles, appearance, and legend
    fig.update_layout(
        title='Production Type Proportions for Major Regions',
        height=600,
        width=1000,
        showlegend=True,  # Ensure legend is shown
        legend_title='Genres',  # Title of the legend
        legend=dict(
            x=1.05, y=1, traceorder='normal',  # Position of the legend
            title='Genres', xanchor='left', yanchor='top'
        )
    )

    # Adjust the layout of the subplot titles
    fig.update_layout(
        title_x=0.5,
        title_y=0.95
    )

    return fig

def return_genre_prop_by_prod_type(genre_proportions):
    # Group data
    grouped_genres = genre_proportions.groupby(['prod_type', 'genres'], observed=False).sum('count').reset_index()
    grouped_genres['proportion'] = grouped_genres['count'] / grouped_genres['total']

    # Create a pivot table to get proportions by production type and genre
    pivot_data = grouped_genres.pivot_table(index='prod_type', columns='genres', values='proportion', fill_value=0, observed=False)
    
    # Create a bar plot using Plotly
    fig = go.Figure()

    # Add a bar trace for each genre
    for genre in pivot_data.columns:
        fig.add_trace(go.Bar(
            x=pivot_data.index,
            y=pivot_data[genre],
            name=genre
        ))

    # Update layout with titles and legends
    fig.update_layout(
        title="Genre Proportions by Production Type",
        xaxis_title="Production Type",
        yaxis_title="Proportion",
        barmode='stack',
        legend_title="Genres",
        height=600,
        width=1000
    )

    return fig

def return_genre_trends_by_prod_type(genre_proportions):
    # Create subplots
    prod_types = genre_proportions['prod_type'].unique()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=prod_types,
        shared_yaxes=True
    )

    # Loop over production types and add traces
    for i, prod_type in enumerate(prod_types):
        subset = genre_proportions[genre_proportions['prod_type'] == prod_type]
        
        # Add a line trace for each genre
        for genre in subset['genres'].unique():
            genre_subset = subset[subset['genres'] == genre]
            fig.add_trace(
                go.Scatter(
                    x=genre_subset['dvd_era'],
                    y=genre_subset['prop'],
                    mode='lines+markers',
                    name=genre,
                    line=dict(shape='linear'),
                    marker=dict(symbol='circle')
                ),
                row=(i//2)+1, col=(i%2)+1
            )

    # Update layout for titles, axis labels, and legend
    fig.update_layout(
        title="Genre Proportions Across DVD Eras",
        height=800,
        width=1000,
        showlegend=True,
        legend_title="Genres",
    )

    # Update x and y axis labels for all subplots
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
    # Create edges list from the dataframe
    before_edges = create_edges_list(df)
    G_before = nx.Graph()
    G_before.add_edges_from(before_edges)
    
    # Get positions for nodes using spring layout
    pos = nx.spring_layout(G_before)
    
    # Extract node and edge data
    edge_x = []
    edge_y = []
    for edge in G_before.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
    
    # Node data
    node_x = []
    node_y = []
    for node in G_before.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Create Plotly figure
    fig = go.Figure()

    # Add edges to the plot
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes to the plot
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G_before.nodes),
        textposition='bottom center',
        marker=dict(
            showscale=False,  # Remove the colorbar
            colorscale='Blues',
            size=15,  # Increase the node size
            color='lightblue',
            line_width=1
        ),
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        title="Production Company Collaboration Network",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white',
        height=600,
        width=800
    )

    return fig


def return_avg_num_prod_companies(yearly_avg_companies):
    # Create the figure
    fig = go.Figure()

    # Plot mean number of production companies per year
    fig.add_trace(go.Scatter(x=yearly_avg_companies['release_year'],
                             y=yearly_avg_companies['production_companies'],
                             mode='lines+markers',
                             name='Average Production Companies',
                             line=dict(width=2, color='#2ecc71')))

    # Add linear fit to the plot
    z = np.polyfit(yearly_avg_companies['release_year'],
                   yearly_avg_companies['production_companies'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=yearly_avg_companies['release_year'],
                             y=p(yearly_avg_companies['release_year']),
                             mode='lines',
                             name=f'Trend line (slope: {z[0]:.3f})',
                             line=dict(dash='dash', color='red')))

    # Update layout for the first plot
    fig.update_layout(
        title='Average Number of Production Companies Over Time',
        xaxis_title='Release Year',
        yaxis_title='Average Number of Production Companies',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    # Create a heatmap of correlations
    corr_matrix = yearly_avg_companies.corr()
    heatmap_fig = px.imshow(corr_matrix, 
                            text_auto=True, 
                            color_continuous_scale='RdBu', 
                            title='Correlation Matrix', 
                            labels={'x': 'Year/Production Companies', 'y': 'Year/Production Companies'})
    heatmap_fig.update_layout(height=500)

    return fig, heatmap_fig


def return_num_companies_per_prod_type(df):
    # Create a figure
    fig = go.Figure()

    # Define production types and colors for better visualization
    production_types = ['Super', 'Big', 'Small', 'Independent']
    colors = ['red', 'blue', 'green', 'purple']
    
    df_graph = df[df['production_companies'].str.len() > 0]
    
    # For each production type
    for prod_type, color in zip(production_types, colors):
        # Group by year and calculate average collaborations
        yearly_data = []
        years = sorted(df_graph['release_year'].unique())

        for year in years:
            # Filter data for this year and production type
            df_filtered = df_graph[
                (df_graph['release_year'] == year) &
                (df_graph['prod_type'] == prod_type)
            ]

            # Get total number of movies for normalization
            total_movies = len(df_filtered)

            if total_movies > 0:  # Only process if there are movies
                # Create network
                edges = create_edges_list(df_filtered)
                G = nx.Graph()
                G.add_edges_from(edges)

                # Calculate average collaborations normalized by number of movies
                if G.number_of_nodes() > 0:
                    avg_collaborations = G.number_of_edges() / total_movies
                else:
                    avg_collaborations = 0

                yearly_data.append((year, avg_collaborations))

        # Convert to arrays for plotting
        years, avg_collabs = zip(*yearly_data)

        # Plot line for the production type
        fig.add_trace(go.Scatter(
            x=years, y=avg_collabs, mode='lines+markers', name=prod_type, 
            line=dict(color=color, width=2)
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
        annotations=[
            dict(
                x=1997, y=1, xref="x", yref="y", text="Pre-DVD", showarrow=False,
                font=dict(size=12, color="gray")
            ),
            dict(
                x=2001.5, y=1, xref="x", yref="y", text="DVD Era", showarrow=False,
                font=dict(size=12, color="gray")
            ),
            dict(
                x=2015, y=1, xref="x", yref="y", text="Post-DVD", showarrow=False,
                font=dict(size=12, color="gray")
            ),
        ],
    )
    
    # Add vertical lines for DVD era boundaries
    fig.add_vline(x=1997, line=dict(color='gray', dash='dash', width=2))
    fig.add_vline(x=2006, line=dict(color='gray', dash='dash', width=2))

    return fig
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
# Utility function: Create histogram
def create_histogram(data, x_col, color_col, title, labels, color_palette, nbins=50):
    fig = px.histogram(
        data,
        x=x_col,
        color=color_col,
        nbins=nbins,
        title=title,
        labels=labels,
        color_discrete_sequence=color_palette,
        opacity=0.6
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=labels.get(x_col, x_col),
        yaxis_title="Count",
        title_x=0.5,
        title_font_size=20,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(t=50, b=50, l=50, r=50),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    return fig


# Utility function: Create line plot
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
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(t=50, b=50, l=50, r=50),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.2)")
    return fig


# Utility function: Create stacked bar plot
def create_stacked_bar(data, x_col, y_cols, title, color_palette):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(
            go.Bar(
                x=data[x_col],
                y=data[col],
                name=col,
                marker_color=color_palette[i % len(color_palette)],
            )
        )
    fig.update_layout(
        barmode="relative",
        template="plotly_white",
        title=title,
        xaxis_title=x_col,
        yaxis_title="Proportion",
        legend_title="Categories",
        title_x=0.5,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# Utility function: Create choropleth map
def create_choropleth_map(geo_data, region_mapping, color_palette):
    geo_data["region"] = geo_data["SOVEREIGNT"].map(region_mapping)
    geo_data = geo_data.dropna(subset=["region"])
    fig = px.choropleth(
        geo_data,
        geojson=geo_data.geometry,
        locations=geo_data.index,
        color="region",
        hover_name="SOVEREIGNT",
        color_discrete_sequence=color_palette,
    )
    fig.update_geos(
        showcoastlines=True, coastlinecolor="Black", showland=True, showframe=False
    )
    fig.update_layout(
        template="plotly_white",
        title="Production Countries by Region",
        title_x=0.5,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )
    return fig


# Utility function: Filter data for proportions
def calculate_proportions(df, base_vars, target_var):
    counts = df.groupby(base_vars)[target_var].value_counts(normalize=True).unstack()
    return counts.reset_index()