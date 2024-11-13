# %%
# import libraries 
import pandas as pd
import matplotlib.pyplot as plt
import os 
import math
import sys
from pathlib import Path
import seaborn as sns

# Add 'src' to the system path
sys.path.append(str(Path().resolve() / 'src'))
from src.data.process_data import *
from src.data.clean_data import *

# %% [markdown]
# IMPORTANT: these scripts/functions assume you have the following files in the data/raw directory:
# - From the CMU dataset: 
#     - movie.metadata.tsv
#     - plot_summaries.txt
# - From the TMDB dataset: 
#     - TMDB_movie_dataset_v11.csv
# 
# AND have data/processed folder created
# 
# Note: download CMU dataset here: https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
# and TMDB dataset here (Download button): https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

# %%
# from raw files, creates clean datafiles
%run src/data/clean_data.py

# %%
# from clean data files, creates a dataframe with CMU + plots & TMDB movies 
# df_combined = create_cmu_tmdb_dataset('data/processed/movies.csv','data/processed/plot_summaries.csv', 'data/processed/TMDB_clean.csv', 'inner')

# %%
# df_combined.info()

# %%
# df_combined.head()

# %%
df_tmdb = create_tmdb_dataset('data/processed/TMDB_clean.csv')

df_tmdb.info()

# %%
company_counts = df_tmdb['production_companies'].str.len().value_counts().sort_index()
df_tmdb['production_companies'].str.len().value_counts().sort_index()

#There are 230141 with no production companies 

# %%
company_counts = df_tmdb['production_companies'].str.len().value_counts().sort_index()
plt.bar(company_counts.index, 
        company_counts.values,
        color='#2ecc71',
        alpha=0.7,
        edgecolor='white')

plt.title('Distribution of Number of Production Companies per Movie', 
          fontsize=16, 
          pad=20)
plt.xlabel('Number of Production Companies', fontsize=12)
plt.ylabel('Number of Movies (log scale)', fontsize=12)

plt.yscale('log')

plt.grid(True, axis='y', linestyle='--', alpha=0.3)

sns.despine()

plt.show()

print("\nDetailed Statistics:")
print(f"Movies with no production companies: {company_counts.get(0, 0):,} ({company_counts.get(0, 0)/company_counts.sum()*100:.1f}%)")
print(f"Movies with single production company: {company_counts.get(1, 0):,} ({company_counts.get(1, 0)/company_counts.sum()*100:.1f}%)")
print(f"Movies with multiple production companies: {company_counts[company_counts.index > 1].sum():,} ({company_counts[company_counts.index > 1].sum()/company_counts.sum()*100:.1f}%)")

# %%
#Do no count the ones with no production company 

df_tmdb_prod = df_tmdb[df_tmdb['production_companies'].str.len() > 0]

yearly_avg_companies = (df_tmdb_prod.groupby('release_year')
                       .agg({'production_companies': lambda x: x.str.len().mean()})
                       .reset_index())

plt.figure(figsize=(15, 8))

plt.plot(yearly_avg_companies['release_year'], 
        yearly_avg_companies['production_companies'],
        linewidth=2,
        color='#2ecc71')

plt.title('Average Number of Production Companies per Movie Over Time', 
          fontsize=16, 
          pad=20)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Average Number of Production Companies', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)


z = np.polyfit(yearly_avg_companies['release_year'], 
               yearly_avg_companies['production_companies'], 1)
p = np.poly1d(z)
plt.plot(yearly_avg_companies['release_year'], 
         p(yearly_avg_companies['release_year']), 
         "r--", 
         alpha=0.8,
         label=f'Trend line (slope: {z[0]:.3f})')

plt.legend()
sns.despine()
plt.show()

# %%
correlation = yearly_avg_companies['release_year'].corr(yearly_avg_companies['production_companies'])
covariance = yearly_avg_companies['release_year'].cov(yearly_avg_companies['production_companies'])

print(f"Covariance between year and avg number of production companies: {covariance:.3f}")
print(f"Correlation between year and avg number of production companies: {correlation:.3f}")

plt.figure(figsize=(8, 6))
sns.heatmap(yearly_avg_companies.corr(), 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.3f')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# This shows there is a strong correlation between the number of production companies and the release year. 

# %%
companies = df_tmdb['production_companies'].explode().value_counts()
plt.figure(figsize=(15, 8))

plt.plot(range(len(companies)), 
         companies.values, 
         linewidth=2, 
         marker='o',
         markersize=4,
         color='#2ecc71')
plt.yscale('log')
plt.xscale('log')

    
plt.title('Distribution of Production Companies in Movies', 
          fontsize=16, 
          pad=20)
plt.xlabel('Production Company', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

sns.despine()

for i in range(7):  # Annotate top 10
    plt.annotate(f'{companies.index[i]}',
                xy=(i, companies.values[i]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
print(df_combined['production_companies'].explode().value_counts().describe())


# %%
top_100_companies = df_tmdb['production_companies'].explode().value_counts().head(100)
plt.figure(figsize=(15, 20))

ax = sns.barplot(
    y=top_100_companies.index,
    x=top_100_companies.values,
    palette='viridis'  
)

plt.title('Top 100 Production Companies by Number of Movies', 
          fontsize=16, 
          pad=20)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Production Company', fontsize=12)

for i in ax.containers:
    ax.bar_label(i, padding=5)
ax.grid(axis='x', linestyle='--', alpha=0.7)
sns.despine()



# %%
company_revenue = (df_tmdb.explode('production_companies')
                  .groupby('production_companies')['revenue']
                  .agg(['sum', 'count'])
                  .sort_values('sum', ascending=False))

company_revenue['revenue_millions'] = company_revenue['sum'] / 1_000_000

# Show top 20 companies by revenue
print("Top 20 Production Companies by Total Revenue:")
print(company_revenue.head(20))

plt.figure(figsize=(15, 8))
sns.barplot(data=company_revenue.head(20).reset_index(), 
            x='revenue_millions', 
            y='production_companies')
plt.title('Top 20 Production Companies by Total Revenue')
plt.xlabel('Total Revenue (Millions USD)')
plt.ylabel('Production Company')
sns.despine()

# %%
top_companies = (df_tmdb.explode('production_companies')
                .groupby('production_companies')['revenue']
                .sum()
                .nlargest(50)
                .index)

# Prepare data for plotting
company_revenue_time = (df_combined.explode('production_companies')
                       .groupby(['production_companies', 'release_year'])['revenue']
                       .sum()
                       .reset_index())

# Create the plot
plt.figure(figsize=(15, 8))

# Plot a line for each top company
for company in top_companies:
    data = company_revenue_time[company_revenue_time['production_companies'] == company]
    plt.plot(data['release_year'], 
            data['revenue'] / 1_000_000,  # Convert to millions
            linewidth=2,
            label=company)

plt.title('Revenue Over Time for Top 5 Production Companies', 
          fontsize=16, 
          pad=20)

# plt.yscale('log')
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Revenue (Millions USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), 
          loc='upper left', 
          borderaxespad=0.)

sns.despine()
plt.show()

# %%
remakes = (df_tmdb.groupby('dvd_era'))
remakes.head(40)


# %%
remakes = (df_tmdb.groupby('title')
           .agg({'release_year': ['count']})
           .reset_index())
remakes.head()
# # Filter for titles that appear multiple times
# remakes = remakes[remakes[('release_year', 'count')] > 1]

# %%
remakes = (df_tmdb.groupby('title')
           .agg({'release_year': ['count']})
           .reset_index())

# Filter for titles that appear multiple times
remakes = remakes[remakes[('release_year', 'count')] > 1]

# Create a plot
plt.figure(figsize=(15, 8))

# Count remakes per year
yearly_remakes = (df_tmdb[df_tmdb['title'].isin(remakes['title'])]
                 .groupby('release_year')
                 .size())

plt.plot(yearly_remakes.index, yearly_remakes.values, 
         marker='o', 
         linewidth=2)

plt.title('Number of Potential Remakes by Year', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
sns.despine()

# %%
# Create a function to identify potential remakes
def identify_remakes(group):
    if len(group) <= 1:
        return pd.Series({'is_remake': False, 'original_year': None})
    
    # Sort by year
    group = group.sort_values('release_year')
    
    # Mark all but the first as remakes
    is_remake = [False] + [True] * (len(group) - 1)
    original_year = [group['release_year'].iloc[0]] * len(group)
    
    return pd.Series({'is_remake': is_remake, 'original_year': original_year})

# Apply the function
df_with_remakes = (df_combined.groupby('title')
                   .apply(identify_remakes)
                   .reset_index())

# Plot remakes over time
plt.figure(figsize=(15, 8))

yearly_remakes = (df_with_remakes[df_with_remakes['is_remake']]
                 .groupby('release_year')
                 .size())

plt.plot(yearly_remakes.index, 
         yearly_remakes.values, 
         marker='o', 
         linewidth=2)

plt.title('Number of Movie Remakes by Year', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Remakes', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
sns.despine()

# Print some statistics
print(f"Total number of remakes: {df_with_remakes['is_remake'].sum()}")
print("\nSome example remakes:")
print(df_with_remakes[df_with_remakes['is_remake']].head())


