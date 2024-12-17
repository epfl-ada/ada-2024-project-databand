import pandas as pd
import matplotlib.pyplot as plt
import os 
import math
import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import plotly.graph_objects as go


sys.path.append(str(Path().resolve() / 'src'))
from src.data.process_data import *
from src.data.clean_data import *
from src.utils.data_utils import *
from src.utils.plot_utils import *


def create_genre_proportions_plot(df_tmdb):
   

    unique_genres = set(genre for genres_list in df_tmdb['genres'] for genre in genres_list.split('|'))


    genre_proportions = pd.DataFrame()

   
    for genre in unique_genres:
        genre_counts = df_tmdb[df_tmdb['genres'].apply(lambda x: genre in x)].groupby('dvd_era').size()
        movies_per_era = df_tmdb.groupby('dvd_era').size()
        genre_proportion = genre_counts / movies_per_era
        genre_proportions[genre] = genre_proportion

    
    genre_proportions = genre_proportions.T

  
    genre_proportions = genre_proportions.fillna(0)

    plt.figure(figsize=(12, 8))


    for genre in genre_proportions.index:
        plt.plot(genre_proportions.columns, genre_proportions.loc[genre], label=genre)
    
    plt.title("Proportion of Genres Over DVD Eras", fontsize=16)
    plt.xlabel("DVD Era", fontsize=14)
    plt.ylabel("Proportion", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.show()



def create_genre_proportions_heatmap(df_tmdb):
   
    
    unique_genres = set(genre for genres_list in df_tmdb['genres'] for genre in genres_list.split('|'))

    
    genre_proportions = pd.DataFrame()

    
    for genre in unique_genres:
        genre_counts = df_tmdb[df_tmdb['genres'].apply(lambda x: genre in x)].groupby('dvd_era').size()
        movies_per_era = df_tmdb.groupby('dvd_era').size()
        genre_proportion = genre_counts / movies_per_era
        genre_proportions[genre] = genre_proportion

    
    genre_proportions = genre_proportions.T

    
    genre_proportions = genre_proportions.fillna(0)

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



def get_genre_mean_revenues_pivot(df_tmdb):
   
    sorted_data = df_tmdb.sort_values('revenue', ascending=False)
    
    
    top_10_percent_index = int(len(sorted_data) * 0.10)
    top_10_percent_data = sorted_data.head(top_10_percent_index)
    
    
    top_10_percent_data['genres'] = top_10_percent_data['genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    top_10_percent_exploded = top_10_percent_data.explode('genres')
    
    
    genre_mean_revenues = (
        top_10_percent_exploded
        .groupby(['dvd_era', 'genres'])['revenue']
        .mean()
        .reset_index()
        .rename(columns={'genres': 'genre', 'revenue': 'mean_revenue'})
    )
    
    
    genre_mean_revenues_pivot = genre_mean_revenues.pivot_table(
        index='dvd_era', 
        columns='genre', 
        values='mean_revenue', 
        aggfunc='mean'
    ).reset_index()
    
    
    adjustment_factors = {
        'during_to_pre': 0.91,  
        'post_to_pre': 0.73     
    }

    
    for genre in genre_mean_revenues_pivot.columns[1:]:  
        genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'during', genre] *= adjustment_factors['during_to_pre']
        genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'post', genre] *= adjustment_factors['post_to_pre']
    
    return genre_mean_revenues_pivot



def create_genre_contributions_plot(genre_mean_revenues_pivot):
    
    genres = genre_mean_revenues_pivot.columns[1:]  # Exclude the 'dvd_era' column
    pre_values = genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'pre'].iloc[0, 1:]
    during_values = genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'during'].iloc[0, 1:]
    post_values = genre_mean_revenues_pivot.loc[genre_mean_revenues_pivot['dvd_era'] == 'post'].iloc[0, 1:]

    # Create the figure
    fig = go.Figure()

    # Add bar traces for each era
    fig.add_trace(
        go.Bar(
            x=genres,
            y=pre_values,
            name='Pre Era',
            marker=dict(color='blue')
        )
    )
    fig.add_trace(
        go.Bar(
            x=genres,
            y=during_values,
            name='During Era',
            marker=dict(color='orange')
        )
    )
    fig.add_trace(
        go.Bar(
            x=genres,
            y=post_values,
            name='Post Era',
            marker=dict(color='green')
        )
    )

    # Update layout for grouped bar chart
    fig.update_layout(
        title="Genre Revenues by Era",
        xaxis_title="Genres",
        yaxis_title="Revenue",
        barmode='group',  # Grouped bars for comparison
        legend=dict(title="DVD Era"),
    )

    return fig
