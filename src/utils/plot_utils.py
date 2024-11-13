import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data_utils import adjust_inflation

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


def plot_mean_budget_inflation(df):
    budget_stats = df[df.budget > 0].groupby('release_year')['budget'].agg(mean_budget='mean').reset_index()

    # Adjust the inflation for the budget statistics
    budget_stats_inflation = adjust_inflation(budget_stats, old_col='mean_budget', new_col='mean_budget_inflation')

    # Set the style and context for the plot
    sns.set(style="whitegrid", context="talk")

    # Plot the statistics
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=budget_stats_inflation, x='release_year', y='mean_budget_inflation', marker='o')

    # Customize the plot
    plt.title('Year by year mean film budget - adjusted for inflation', fontsize=16)
    plt.xlabel('Release year')
    plt.ylabel('Budget (inflation adjusted)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_budget_histograms(df, eras, colors, labels, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

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