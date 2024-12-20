import colorcet as cc
import seaborn as sns
import numpy as np
from src.utils.data_utils import adjust_inflation
import geopandas as gpd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import networkx as nx
from itertools import combinations


def plot_movies_budget_slider(df, mode='budget'):
    # Filter out movies with budget under 1000
    df = df[df['budget'] > 1000]

    # Get unique years in the dataset
    years = df['release_year'].unique()
    years = list(range(min(years), max(years) + 1))

    # Get the min and max budget from the dataset
    min_budget = df['budget'].min()
    max_budget = df['budget'].max()

    # Define variables that will change based on the mode
    if mode == 'budget':
        title = 'Number of films exceeding a fixed budget over time'
        xlabel = 'Year'
        ylabel = 'Number of films'
        threshold_name = 'Budget'
        slider_prefix = 'Budget threshold: '
        y_func = lambda threshold: get_movies_above_budget(threshold, mode)
        thresholds = np.logspace(np.log10(min_budget), np.log10(max_budget), 100)

    elif mode == 'relative':
        title = 'Percentage of films exceeding a fixed budget over time'
        xlabel = 'Year'
        ylabel = 'Percentage of films'
        threshold_name = 'Budget'
        slider_prefix = 'Budget threshold: '
        y_func = lambda threshold: get_movies_above_budget(threshold, mode)
        thresholds = np.logspace(np.log10(min_budget), np.log10(max_budget), 100)

    elif mode == 'percentage':
        title = 'Percentage of films exceeding a relative-to-mean budget over time'
        xlabel = 'Year'
        ylabel = 'Percentage of films'
        threshold_name = 'Percentage of mean budget'
        slider_prefix = 'Percentage threshold: '
        y_func = lambda threshold: get_movies_above_budget(threshold, mode)
        thresholds = np.linspace(0, 1000, 100)  # 0% to 100% for percentage mode

    # Function to get the count of movies with a budget above a certain threshold for each year
    def get_movies_above_budget(threshold, mode='budget'):
        counts = []
        for year in years:
            df_year = df[df['release_year'] == year]
            total_movies = len(df_year)
            
            if mode == 'budget':
                count = len(df_year[df_year['budget'] > threshold])
                counts.append(count)
                
            elif mode == 'relative':
                count = len(df_year[df_year['budget'] > threshold]) / total_movies * 100
                counts.append(count)
                
            elif mode == 'percentage':
                mean_budget = df_year['budget'].mean()
                threshold_value = mean_budget * (threshold / 100)
                count = len(df_year[df_year['budget'] > threshold_value]) / total_movies * 100
                counts.append(count)
                
        return counts

    # Create the figure
    fig = go.Figure()

    # Create steps for the slider
    steps = []

    for i, threshold in enumerate(thresholds):
        # Add the trace for each threshold
        fig.add_trace(go.Scatter(
            visible=(i == 0),  # Only the first row is visible initially
            y=y_func(threshold),
            mode='lines+markers',
            name=f'{threshold:.0f}' if mode == 'budget' or mode == 'relative' else f'{threshold:.0f}%'
        ))

        # Add steps to update the graph based on the slider value
        step = dict(
            method="update",
            args=[  # Update the data of the plot for the selected threshold
                {"visible": [j == i for j in range(len(thresholds))]},  # Show only the current threshold's trace
                {"title": f"{threshold_name} exceeding {threshold:.0f}$" if mode != 'percentage' else f"{threshold_name} exceeding {threshold:.0f}%"}
            ],
            label=f"{threshold:.0f}"  # Label for the current threshold
        )
        steps.append(step)

    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": slider_prefix},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        sliders=sliders,
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(tickmode='array', tickvals=years),  # Display year labels on the x-axis
        template="plotly_dark"  # Optional: Set a dark theme for the chart
    )

    fig.show()





def plot_top_proportions_per_era(top_k_df, column, k):
    pivot_data = top_k_df.pivot_table(index='dvd_era', columns=column, values='proportion', aggfunc='sum',
                                      observed=True).fillna(0)
    palette = sns.color_palette(cc.glasbey, n_colors=len(pivot_data.columns))

    # use stacked bar plot
    pivot_data.plot(kind='bar', stacked=True, figsize=(8, 6), color=palette)

    plt.title('Proportions of top ' + str(k) + " " + column + ' by DVD Era', fontsize=16)
    plt.xlabel('DVD Era', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend(title=column, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_wordclouds_per_genre(wordclouds, genre):
    f, axes = plt.subplots(1, 3, figsize=(20, 20))
    eras = ['pre', 'during', 'post']
    for i in range(len(wordclouds)):
        axes[i].imshow(wordclouds[i], interpolation='bilinear')
        axes[i].axis('off')
        axes[i].title.set_text(f'{genre} movies - {eras[i]} DVD era')
    plt.show()

def plot_loghist(x, bins, xlabel, ylabel):
    """
    Code adapted from: https://stackoverflow.com/questions/7694298/how-to-make-a-log-log-histogram-in-python
    """
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel + ' distribution by ' + xlabel)
    plt.show()

def plot_revenue_per_era(df):
    # Filter out movies with missing or zero revenue data
    df_filtered = df[df['revenue'] > 0]

    # Plot revenue distributions for each DVD era
    plt.figure(figsize=(14, 6))
    sns.histplot(data=df_filtered, x='revenue', hue='dvd_era', kde=True, log_scale=True, palette='viridis')
    plt.title('Distribution of Revenue by DVD Era')
    plt.xlabel('Revenue (log scale)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def style_plot(title='', xlabel='', ylabel='', legend=False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    sns.despine()
    if legend:
        plt.legend()
    plt.show()


def plot_movie_freq_per_production_company(df):
    companies = df['production_companies'].explode().value_counts()
    plt.figure(figsize=(12, 6))

    plt.plot(range(len(companies)),
             companies.values,
             linewidth=2,
             marker='o',
             markersize=4,
             color='#2ecc71')
    plt.yscale('log')
    plt.xscale('log')

    for i in range(5):  # Annotate top 5
        plt.annotate(f'{companies.index[i]}',
                     xy=(i, companies.values[i]),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    style_plot('Number of Movies per Production Company', 'Production Company', 'Number of Movies', False)

def plot_world_map(countries_regions):
    # Load a world map
    world = gpd.read_file('./data/ne_110m_admin_0_countries.shp', encoding='utf-8')
    world['SOVEREIGNT'] = world['SOVEREIGNT'].str.lower()

    # Add a region column by mapping from the dictionary
    world['region'] = world['SOVEREIGNT'].map(countries_regions)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.dropna(subset=['region']).plot(column='region', legend=True, ax=ax, cmap='tab20',
                                         legend_kwds={'title': 'World Regions', 'bbox_to_anchor': (1.05, 1),
                                                      'loc': 'upper left'})
    plt.title("Production Countries Map Colored by Region", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_genre_prop_by_prod_type(genre_proportions):
    grouped_genres = genre_proportions.groupby(['prod_type', 'genres'], observed=False).sum('count').reset_index()
    grouped_genres['proportion'] = grouped_genres['count'] / grouped_genres['total']

    (grouped_genres.pivot_table(index='prod_type', columns='genres', values='proportion', fill_value=0, observed=False)
     .plot(kind='bar', stacked=True, figsize=(8, 6), colormap='tab20'))
    plt.tight_layout()
    plt.legend(title="Genres", bbox_to_anchor=(1.2, 1), loc='upper right')
    style_plot("Genre Proportions by Production Type", "Production Type", "Proportion")

def plot_genre_trends_by_prod_type(genre_proportions):
    sns.set(style="whitegrid")
    f, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    for i, prod_type in enumerate(genre_proportions['prod_type'].unique()):
        subset = genre_proportions[genre_proportions['prod_type'] == prod_type]

        ax = axs.flatten()[i]
        sns.lineplot(data=subset, x='dvd_era', y='prop', hue='genres', marker='o', ax=ax, palette='tab20')
        ax.set_title(f"{prod_type} Production Type")
        ax.set_xlabel('DVD Era')
        ax.set_ylabel('Proportion')

        ax.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

    plt.suptitle('Genre Proportions Across DVD Eras', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_movies_prop_per_region(region_props):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=region_props[region_props.prop > 0.01], x='release_year', y='prop', hue='region', palette='tab20')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), frameon=False)
    style_plot('Proportion of movies released over time per region', 'Release Year', 'Proportion')
    plt.show()

def plot_prod_type_prop_per_region(selected_regions, df_countries_filtered):
    f, axs = plt.subplots(1, len(selected_regions), figsize=(16, 6), sharey=True)

    for i, region in enumerate(selected_regions):
        ax = axs[i]
        legend = False if (i < len(selected_regions) - 1) else 'full'
        sns.histplot(data=df_countries_filtered[(df_countries_filtered['region'] == region)], x='dvd_era',
                     hue='prod_type',
                     multiple='fill', legend=legend, ax=ax, hue_order=['Independent', 'Small', 'Big', 'Super'],
                     palette='tab20')
        ax.set_title(region)
        ax.set_ylabel('Proportion')
        ax.set_xlabel('DVD Era')
    sns.move_legend(axs[len(selected_regions) - 1], loc='upper right', bbox_to_anchor=(2.25, 1),
                    title='Production Type')
    f.suptitle('Production type proportions for major regions', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_genre_prop_per_region(selected_regions, countries_genres_props):
    sns.set(style="whitegrid")
    f, axs = plt.subplots(1, len(selected_regions), figsize=(20, 6), sharey=True)
    for i, region in enumerate(selected_regions):
        subset = countries_genres_props[countries_genres_props['region'] == region]
        ax = axs[i]
        legend = False if (i < len(selected_regions) - 1) else 'full'
        sns.lineplot(data=subset, x='dvd_era', y='prop', hue='genres', marker='o', ax=ax, legend=legend,
                     palette='tab20')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('DVD Era')
        ax.set_title(region)

    sns.move_legend(axs[len(selected_regions) - 1], loc='upper right', bbox_to_anchor=(2.25, 1), title='Genres')
    f.suptitle('Production type proportions for major regions', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_rolling_averages(proportion_rolling):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=proportion_rolling.index, y=proportion_rolling['Independent'], 
                             mode='lines', name='Independent Productions (<0.1x mean budget)', 
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=proportion_rolling.index, y=proportion_rolling['Small'], 
                             mode='lines', name='Small Productions (<1x mean budget)', 
                             line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=proportion_rolling.index, y=proportion_rolling['Big'], 
                             mode='lines', name='Big Productions (>1x mean budget)', 
                             line=dict(color='green')))
    fig.add_trace(go.Scatter(x=proportion_rolling.index, y=proportion_rolling['Super'], 
                             mode='lines', name='Super Productions (>5x mean budget)', 
                             line=dict(color='purple')))

    fig.update_layout(
        title='Proportions of Different Types of Productions Over the Years (3-year Rolling Average)',
        xaxis_title='Release Year',
        yaxis_title='Proportion',
        template='plotly_dark',
        showlegend=True,
        height=600
    )

    # Show the plot
    fig.show()

def plot_budget_histograms(df, eras, colors, labels, title):
    fig, axes = plt.subplots(len(eras), 1, figsize=(12, 6*len(eras)), sharex=True)

    if len(eras) == 1:
        era1 = eras[0][0]
        era2 = eras[0][1]
        sns.histplot(df[df['dvd_era'] == era1]['budget'], bins=50, color=colors[0][0], label=labels[0][0],
                     kde=True, stat="density")
        sns.histplot(df[df['dvd_era'] == era2]['budget'], bins=50, color=colors[0][1], label=labels[0][1],
                     kde=True, stat="density")
        plt.xscale('log')
        plt.xlabel('Budget')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'{title} ({labels[0][0]} vs {labels[0][1]})')
    else:
        for i, (era1, era2) in enumerate(eras):
            ax = axes[i]
            sns.histplot(df[df['dvd_era'] == era1]['budget'], bins=50, color=colors[i][0], label=labels[i][0], ax=ax, kde=True, stat="density")
            sns.histplot(df[df['dvd_era'] == era2]['budget'], bins=50, color=colors[i][1], label=labels[i][1], ax=ax, kde=True, stat="density")
            ax.set_xscale('log')
            ax.set_xlabel('Budget')
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title(f'{title} ({labels[i][0]} vs {labels[i][1]})')

    plt.tight_layout()
    plt.show()


def plot_revenue_inflation(df):
    
    # Adjust the inflation for the revenue statistics
    revenue_inflation = adjust_inflation(df, old_col='revenue', new_col='revenue_inflation')

    # Filter out movies with missing or zero revenue data
    revenue_inflation = revenue_inflation[revenue_inflation['revenue_inflation'] > 0]

    # Plot revenue distributions for each DVD era
    plt.figure(figsize=(14, 6))
    sns.histplot(data=revenue_inflation, x='revenue_inflation', hue='dvd_era', kde=True, log_scale=True, palette='viridis')
    plt.title('Distribution of Revenue by DVD Era - adjusted for inflation')
    plt.xlabel('Revenue (log scale)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def plot_mean_budget_inflation(df, save_path=None):
    # Filter out movies with budget under 1000
    budget_stats = df[df.budget > 0].groupby('release_year')['budget'].agg(mean_budget='mean').reset_index()

    # Adjust the inflation for the budget statistics
    budget_stats_inflation = adjust_inflation(budget_stats, old_col='mean_budget', new_col='mean_budget_inflation')

    # Calculate mean budgets for each period
    pre_dvd_mean = budget_stats_inflation[budget_stats_inflation['release_year'] < 1997]['mean_budget_inflation'].mean()
    dvd_mean = budget_stats_inflation[(budget_stats_inflation['release_year'] >= 1997) & (budget_stats_inflation['release_year'] <= 2013)]['mean_budget_inflation'].mean()
    post_dvd_mean = budget_stats_inflation[budget_stats_inflation['release_year'] > 2013]['mean_budget_inflation'].mean()

    # Create the figure
    fig = go.Figure()

    # Plot the adjusted inflation budget data
    fig.add_trace(go.Scatter(x=budget_stats_inflation['release_year'], 
                             y=budget_stats_inflation['mean_budget_inflation'], 
                             mode='lines+markers', name='Mean Budget (Inflation Adjusted)', 
                             marker=dict(color='blue')))

    # Add vertical lines for the DVD era start and end
    fig.add_vline(x=1997, line=dict(color='green', dash='dot'), annotation_text="Start DVD era", annotation_position="top left")
    fig.add_vline(x=2013, line=dict(color='red', dash='dot'), annotation_text="End DVD era", annotation_position="top left")

    # Add horizontal lines for mean budgets
    fig.add_hline(y=pre_dvd_mean, line=dict(color='blue', dash='dash'), annotation_text="Pre-DVD Mean", annotation_position="bottom left")
    fig.add_hline(y=dvd_mean, line=dict(color='orange', dash='dash'), annotation_text="DVD Era Mean", annotation_position="bottom left")
    fig.add_hline(y=post_dvd_mean, line=dict(color='purple', dash='dash'), annotation_text="Post-DVD Mean", annotation_position="bottom left")

    # Customize layout
    fig.update_layout(
        title='Year by Year Mean Film Budget - Adjusted for Inflation',
        xaxis_title='Release Year',
        yaxis_title='Budget (Inflation Adjusted)',
        template='plotly_dark',
        showlegend=True,
        height=600
    )

    # Save the plot if save_path is provided
    if save_path:
        fig.write_image(save_path)

    # Show the plot
    fig.show()

def plot_avg_num_prod_companies(yearly_avg_companies):
    # Create 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot mean number of production companies per year
    ax1.plot(yearly_avg_companies['release_year'],
             yearly_avg_companies['production_companies'],
             linewidth=2,
             color='#2ecc71')

    # Add linear fit to the plot
    z = np.polyfit(yearly_avg_companies['release_year'],
                   yearly_avg_companies['production_companies'], 1)
    p = np.poly1d(z)
    ax1.plot(yearly_avg_companies['release_year'],
             p(yearly_avg_companies['release_year']),
             "r--",
             alpha=0.8,
             label=f'Trend line (slope: {z[0]:.3f})')

    ax1.set_title('Average Number of Production Companies\nOver Time')
    ax1.set_xlabel('Release Year')
    ax1.set_ylabel('Average Number of Production Companies')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    correlation = yearly_avg_companies['release_year'].corr(yearly_avg_companies['production_companies'])
    covariance = yearly_avg_companies['release_year'].cov(yearly_avg_companies['production_companies'])

    # Plot correlation between number of production companies and release year
    sns.heatmap(yearly_avg_companies.corr(),
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.3f',
                ax=ax2)
    plt.show()

def create_edges_list(df):
    edges = []
    for companies in df['production_companies']:
        if len(companies) > 1:
            edges.extend(list(combinations(companies, 2)))
    return edges

def plot_num_companies_per_prod_type(df):
    # Create figure
    plt.figure(figsize=(15, 8))

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

        # Plot line
        plt.plot(years, avg_collabs, label=prod_type, color=color, linewidth=2, alpha=0.7)

    # Customize plot
    plt.title('Average Number of Collaborations per Movie Over Time by Production Type',
              pad=20, fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Collaborations per Movie', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Production Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add vertical lines for DVD era boundaries
    plt.axvline(x=1997, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=2006, color='gray', linestyle='--', alpha=0.5)

    # Add era labels
    plt.text(1985, plt.ylim()[1], 'Pre-DVD', ha='center', va='bottom')
    plt.text(2001.5, plt.ylim()[1], 'DVD Era', ha='center', va='bottom')
    plt.text(2015, plt.ylim()[1], 'Post-DVD', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_collaborations(df):
    before_edges = create_edges_list(df)
    G_before = nx.Graph()

    G_before.add_edges_from(before_edges)
    plt.figure(figsize=(12, 10))
    nx.draw(
        G_before,
        with_labels=True,
        node_size=500,
        font_size=10,
        node_color='lightblue',  # Add color to nodes
        edge_color='gray',  # Add color to edges
        alpha=0.7,  # Add transparency
        width=1,  # Edge width
        pos=nx.spring_layout(G_before)  # Use spring layout for better node positioning
    )
    plt.title("Production Company Collaboration Network")
    plt.show()


def create_genre_proportions_heatmap(genre_proportions):
    plt.figure(figsize=(14, 8))

    sns.heatmap(
        genre_proportions,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={'label': 'Proportion'},
        linewidths=0.5,
        linecolor='white'
    )

    plt.title("Heatmap of Genre Proportions Over DVD Eras", fontsize=20, weight='bold', pad=20, color='#2E4057')
    plt.xlabel("DVD Era", fontsize=14, color='#2E4057')
    plt.ylabel("Genre", fontsize=14, color='#2E4057')

    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()


def create_genre_contributions_plot(genre_mean_revenues_pivot):
    genres = genre_mean_revenues_pivot.columns[1:]
    pre_values = genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'pre'].iloc[0, 1:].values
    during_values = genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'during'].iloc[0, 1:].values
    post_values = genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'post'].iloc[0, 1:].values

    pre_values_billion = pre_values / 1e6
    during_values_billion = during_values / 1e6
    post_values_billion = post_values / 1e6

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.25

    index = np.arange(len(genres))

    ax.bar(index - bar_width, pre_values_billion, width=bar_width, color='#1f77b4', label='Pre Era')

    ax.bar(index, during_values_billion, width=bar_width, color='#ff7f0e', label='During Era')

    ax.bar(index + bar_width, post_values_billion, width=bar_width, color='#2ca02c', label='Post Era')

    ax.set_title('Genre Revenues by DVD Era', fontsize=16, fontweight='bold')
    ax.set_xlabel('Genres', fontsize=14, fontweight='bold')
    ax.set_ylabel('Revenue (Millions USD)', fontsize=14, fontweight='bold')
    ax.legend(title='DVD Era', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(index, genres, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()

# %run src/utils/plot_utils.py