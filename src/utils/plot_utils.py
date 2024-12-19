import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data_utils import adjust_inflation
import geopandas as gpd

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
    plt.axvline(x=1997, label='Start DVD era', color='green', linestyle='dotted')
    plt.axvline(x=2013, label='End DVD era', color='red', linestyle='dotted')

    # Customize the plot
    plt.title('Year by year mean film budget - adjusted for inflation', fontsize=16)
    plt.xlabel('Release year')
    plt.ylabel('Budget (inflation adjusted)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Show the plot
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

# %run src/utils/plot_utils.py