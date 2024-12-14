# import libraries 
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import seaborn as sns

# Add 'src' to the system path
sys.path.append(str(Path().resolve() / 'src'))
from src.data.process_data import create_tmdb_dataset
from src.utils.load_data import load_raw_data
from src.utils.data_utils import *
from src.utils.plot_utils import *
from src.models.lda_model import *

df = create_tmdb_dataset('data/processed/TMDB_clean.csv')
df_filtered = df[(df['budget'] > 0)]

def plot_mean_budget_inflation(df = df_filtered, save_path=None):
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
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=budget_stats_inflation, x='release_year', y='mean_budget_inflation', marker='o', ax=ax)
    ax.axvline(x=1997, label='Start DVD era', color='green', linestyle='dotted')
    ax.axvline(x=2013, label='End DVD era', color='red', linestyle='dotted')

    # Add horizontal lines for mean budgets
    ax.hlines(pre_dvd_mean, xmin=budget_stats_inflation['release_year'].min(), xmax=1997, colors='blue', linestyles='dashed', label='Pre-DVD Mean Budget')
    ax.hlines(dvd_mean, xmin=1997, xmax=2013, colors='orange', linestyles='dashed', label='DVD Era Mean Budget')
    ax.hlines(post_dvd_mean, xmin=2013, xmax=budget_stats_inflation['release_year'].max(), colors='purple', linestyles='dashed', label='Post-DVD Mean Budget')

    # Customize the plot
    ax.set_title('Year by year mean film budget - adjusted for inflation', fontsize=16)
    ax.set_xlabel('Release year')
    ax.set_ylabel('Budget (inflation adjusted)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    # Return the figure object
    return fig

def plot_movies_budget_slider(df = df_filtered, mode='budget'):
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
                {"title": f"Number of films exceeding threshold of {threshold:.0f}$" }
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

    # Return the figure object
    return fig