import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data_utils import adjust_inflation

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

import plotly.graph_objects as go
import numpy as np

def plot_movies_budget_slider(df, mode='budget'):
    # Filter out movies with budget under 1000
    df = df[df['budget'] > 1000]

    # Get unique years in the dataset
    years = df['release_year'].unique()
    years = list(range(min(years), max(years) + 1))

    # Get the min and max budget from the dataset
    min_budget = df['budget'].min()
    max_budget = df['budget'].max()

    # Function to get the count of movies with a budget above a certain threshold for each year
    def get_movies_above_budget(threshold, mode='budget'):
        counts = []
        for year in years:
            df_year = df[df['release_year'] == year]
            total_movies = len(df_year)
            
            if mode == 'budget':
                # Count movies above a specific budget threshold
                count = len(df_year[df_year['budget'] > threshold])
                counts.append(count)
                
            elif mode == 'relative':
                # Calculate percentage of movies above the threshold for this year
                count = len(df_year[df_year['budget'] > threshold]) / total_movies * 100
                counts.append(count)
                
            elif mode == 'percentage':
                # Calculate threshold as percentage of mean budget
                mean_budget = df_year['budget'].mean()
                threshold_value = mean_budget * (threshold / 100)
                count = len(df_year[df_year['budget'] > threshold_value]) / total_movies * 100
                counts.append(count)
                
        return counts

    # Create the figure
    fig = go.Figure()

    # Determine slider range based on mode
    if mode == 'budget':
        thresholds = np.logspace(np.log10(min_budget), np.log10(max_budget), 100)
    elif mode == 'relative':
        thresholds = np.logspace(np.log10(min_budget), np.log10(max_budget), 100)
    elif mode == 'percentage':
        thresholds = np.linspace(0, 1000, 100)  # 0% to 1000% for percentage mode

    # Create steps for the slider
    steps = []

    for i, threshold in enumerate(thresholds):
        # Add the trace for each threshold
        fig.add_trace(go.Scatter(
            visible=(i == 0),  # Only the first row is visible initially
            y=get_movies_above_budget(threshold, mode),
            mode='lines+markers',
            name=f'{threshold:.0f}' if mode == 'budget' or mode == "relative" else f'{threshold:.0f}%'
        ))

        # Add steps to update the graph based on the slider value
        step = dict(
            method="update",
            args=[  # Update the data of the plot for the selected threshold
                {"visible": [j == i for j in range(len(thresholds))]},  # Show only the current threshold's trace
                {"title": f"Number of films exceeding threshold of {threshold:.0f}$"}
            ],
            label=f"{threshold:.0f}"  # Label for the current threshold
        )
        steps.append(step)

    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{mode.capitalize()} Threshold: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        sliders=sliders,
        title=f"Number of Films by Budget Over Time ({mode.capitalize()} Mode)",
        xaxis_title="Year",
        yaxis_title="Percentage of films" if mode in ['relative', 'percentage'] else "Number of films",
        xaxis=dict(tickmode='array', tickvals=years)  # Display year labels on the x-axis
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


def plot_mean_budget_inflation(df, save_path=None):
    budget_stats = df[df.budget > 0].groupby('release_year')['budget'].agg(mean_budget='mean').reset_index()

    # Adjust the inflation for the budget statistics
    budget_stats_inflation = adjust_inflation(budget_stats, old_col='mean_budget', new_col='mean_budget_inflation')

    # Calculate mean budgets for each period
    pre_dvd_mean = budget_stats_inflation[budget_stats_inflation['release_year'] < 1997]['mean_budget_inflation'].mean()
    dvd_mean = budget_stats_inflation[(budget_stats_inflation['release_year'] >= 1997) & (budget_stats_inflation['release_year'] <= 2013)]['mean_budget_inflation'].mean()
    post_dvd_mean = budget_stats_inflation[budget_stats_inflation['release_year'] > 2013]['mean_budget_inflation'].mean()

    # Set the style and context for the plot
    sns.set(style="whitegrid", context="talk")


    # Plot the statistics
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=budget_stats_inflation, x='release_year', y='mean_budget_inflation', marker='o')
    plt.axvline(x=1997, label='Start DVD era', color='green', linestyle='dotted')
    plt.axvline(x=2013, label='End DVD era', color='red', linestyle='dotted')

    # Add horizontal lines for mean budgets
    plt.hlines(pre_dvd_mean, xmin=budget_stats_inflation['release_year'].min(), xmax=1997, colors='blue', linestyles='dashed', label='Pre-DVD Mean Budget')
    plt.hlines(dvd_mean, xmin=1997, xmax=2013, colors='orange', linestyles='dashed', label='DVD Era Mean Budget')
    plt.hlines(post_dvd_mean, xmin=2013, xmax=budget_stats_inflation['release_year'].max(), colors='purple', linestyles='dashed', label='Post-DVD Mean Budget')

    # Customize the plot
    plt.title('Year by year mean film budget - adjusted for inflation', fontsize=16)
    plt.xlabel('Release yearrrrrrrrrrrrrrrrrrrr')
    plt.ylabel('Budget (inflation adjusted)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()

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

def plot_rolling_averages(proportion_rolling):
    # Set the style and context for the plot
    sns.set(style="whitegrid", context="talk")

    # Create a 4x1 grid of plots
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

    # Plot the proportion of independent productions over the years
    sns.lineplot(data=proportion_rolling['Independent'], ax=axes[0],
                 label='Independent productions (<0.1x mean budget)')
    axes[0].set_title('Proportion of independent productions over the years (3-year rolling average)')
    axes[0].set_xlabel('Release year')
    axes[0].set_ylabel('Proportion')
    axes[0].legend()

    # Plot the proportion of small productions over the years
    sns.lineplot(data=proportion_rolling['Small'], ax=axes[1], label='Small productions (<1x mean budget)')
    axes[1].set_title('Proportion of small productions over the years (3-year rolling average)')
    axes[1].set_xlabel('Release year')
    axes[1].set_ylabel('Proportion')
    axes[1].legend()

    # Plot the proportion of big productions over the years
    sns.lineplot(data=proportion_rolling['Big'], ax=axes[2], label='Big productions (>1x mean budget)')
    axes[2].set_title('Proportion of big productions over the years (3-year rolling average)')
    axes[2].set_xlabel('Release year')
    axes[2].set_ylabel('Proportion')
    axes[2].legend()

    # Plot the proportion of super productions over the years
    sns.lineplot(data=proportion_rolling['Super'], ax=axes[3], label='Super productions (>5x mean budget)')
    axes[3].set_title('Proportion of super productions over the years (3-year rolling average)')
    axes[3].set_xlabel('Release year')
    axes[3].set_ylabel('Proportion')
    axes[3].legend()

    plt.tight_layout()
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

