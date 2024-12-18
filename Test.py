#To install the necessary packages, please run this cell:
!pip install -r pip_requirements.txt


# %%
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

# %%
# First, run the clean_data script to clean the TMDB dataset
%run src/scripts/clean_data.py

# %%
# from clean data files, creates a dataframe with TMDB movies 
df = create_tmdb_dataset('data/processed/TMDB_clean.csv')
df.info()

# %% [markdown]
# # DVD releases over time

# %%
df_dvd_releases = load_raw_data('data/processed/dvd_releases.csv')
df_dvd_releases['dvd_release_date'] = pd.to_datetime(df_dvd_releases['dvd_release_date'])
df_grouped = df_dvd_releases.resample('ME', on='dvd_release_date').size()

# %%
# Revenue per era
plot_revenue_per_era(df)

# %% [markdown]
# # Budget overview
# We then consider budget distributions.

# %%
df_filtered = df[(df['budget'] > 0)]

print('Summary statistics of budget for each DVD era:')
df_filtered.groupby('dvd_era')['budget'].describe().reindex(['pre', 'during', 'post']).transpose()

# %% [markdown]
# We can see that most budget statistics (quantiles, median and mean) are lower in the post-DVD era, but the maximum budget is higher. 
# 
# We examine the budget trends in more details using plots. 

# %%
# Plot mean budget across time, accounting for inflation
plot_mean_budget_inflation(df)

# %% [markdown]
# 

# %% [markdown]
# We can use histograms to compare film budgets before, during and after the DVD era. We use a log-scale to represent movies with both small and high budgets on the same graph. This leads to the results of pre vs. post DVD era to be unreadable. 
# 
# 
# We can nevertheless interpret the results for adjacent eras :
# - Pre vs. during: before DVDs, only high budgets films were produced, with a mono-modal distribution, around 10^7 dollars budget. During the DVD era, the distribution widened, with smaller budget films being produced.
# - During vs. post: after the DVD era, we see the distribution becoming more bimodal with another density maximum around 10^4 dollars.
# 
# Interpreting this is tricky : maybe more movies can become available in streaming services, pushing for smaller-budget productions.

# %%

# Define eras, colors, and labels for the plots
eras = [("pre", "post"), ("post", "during"), ("during", "pre")]
colors = [('green', 'blue'), ('blue', 'red'), ('red', 'green')]
labels = [('Pre DVD Era Budgets', 'Post DVD Era Budgets'), 
          ('Post DVD Era Budgets', 'During DVD Era Budgets'), 
          ('During DVD Era Budgets', 'Pre DVD Era Budgets')]

# Plot histograms
plot_budget_histograms(df_filtered, eras, colors, labels, 'Histogram of Budgets')

# %% [markdown]
# ## Production types
# 
# We categorize the movies in different types according to their budgets (compared to the mean) :
# - Independent movies: less than 1/10th of the mean budget.
# - Small productions: Between 1/10th and 1 of the mean budget.
# - Big productions: Between 1 and 5 times the mean budget.
# - Super productions More than 5 times the mean budget.
# 
# We then plot the proportion of those movies (over the total) using a 3 years rolling average.
# 
# The most interesting finding is that the DVD era seems to correspond to a loss of interest for really expensive movies. This can be explained by the fact that :
# - Before DVDs, going to the cinema was exceptional, but also the only way to consume movies. So only big franchises with high production budgets could really make a lot of profit.
# - After DVDs, the streaming platforms want to differentiate from each other by giving access to exceptional movies, that are really costly.

# %%
prop_rolling_avg = budget_rolling_averages(df_filtered, window=3)
plot_rolling_averages(prop_rolling_avg)

# %% [markdown]
# # Production Companies

# %% [markdown]
# We then take a look at production companies

# %%
df['budget']


# %%
def calculate_roi(df):
    if df.revenue > 0 and df.budget > 0:
        return (df.revenue - df.budget) / df.budget * 100
    else: 
        return 0
    
df['roi'] = df.apply(calculate_roi, axis=1)

# %%
mean_budgets = df[df.budget > 0].groupby('release_year').agg(mean_budget = ('budget', 'mean'))

def categorize_production(row, means):
    mean_budget = means.loc[row.release_year, 'mean_budget']
    if row['budget'] < 0.1 * mean_budget:
        return 'Independent'
    elif row['budget'] < mean_budget:
        return 'Small'
    elif row['budget'] < 5 * mean_budget:
        return 'Big'
    else:
        return 'Super'
    
df['prod_type'] = df.apply(categorize_production, axis=1, args=(mean_budgets,))

# %%
sns.lineplot(
    df[(df.roi > 0) & (df.roi <1e4)].groupby(['release_year', 'prod_type']).agg(mean_roi=('roi', 'mean')).reset_index(),
    x='release_year', y='mean_roi', hue='prod_type')

# %%
df.head()

# %%
from itertools import combinations
import networkx as nx

# %%
def create_edges_list(df):
    edges = []
    for companies in df['production_companies']:
        if len(companies)>1:
            edges.extend(list(combinations(companies,2)))
    # edges = list(set(edges))
    return edges


# %%
df_graph = df[df['production_companies'].str.len() > 0]

before_DVD_era_super = df_graph[(df_graph['dvd_era'] == 'pre') & (df_graph['prod_type'] == 'Super')]
# during_DVD_era = df_graph[df_graph['dvd_era'] == 'during']
# after_DVD_era = df_graph[df_graph['dvd_era'] == 'post']

print(before_DVD_era_super.shape)
# print(during_DVD_era.shape)
# print(after_DVD_era.shape)



# %%
before_edges = create_edges_list(before_DVD_era_super)
# during_edges = create_edges_list(during_DVD_era)
# after_edges = create_edges_list(after_DVD_era)

# %%
before_edges[0]


# %%

# Create the graph
G_before = nx.Graph()
# G_during = nx.Graph()
# G_after = nx.Graph()

G_before.add_edges_from(before_edges)
# G_during.add_edges_from(during_edges)
# G_after.add_edges_from(after_edges)


# %%
from ipywidgets import interact, widgets, IntSlider, Layout
import matplotlib.pyplot as plt
import networkx as nx

@interact(
    year=IntSlider(
        min=1976,
        max=2023,
        step=1,
        value=2000,
        description='Year:',
        continuous_update=False,
        layout=Layout(width='50%')
    ),
    prod_type=widgets.Dropdown(
        options=['Super', 'Big', 'Small', 'Independent'],
        value='Super',
        description='Production Type:'
    )
)
def plot_network_by_year(year, prod_type):
    # Clear previous plot
    plt.clf()
    
    # Filter data
    df_filtered = df_graph[
        (df_graph['release_year'].astype(int) == year) & 
        (df_graph['prod_type'] == prod_type)
    ]
    
    # Create edges list
    edges = create_edges_list(df_filtered)
    
    # Create network
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Calculate network layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        node_size=1000,
        font_size=8,
        font_weight='bold',
        edge_color='gray',
        width=0.5,
        alpha=0.7
    )
    
    # Add title with statistics
    plt.title(f"Production Company Network - {year} ({prod_type} Productions)\n" +
              f"Companies: {G.number_of_nodes()}, " +
              f"Collaborations: {G.number_of_edges()}, " +
              f"Avg. Collaborations per Company: {2*G.number_of_edges()/G.number_of_nodes():.2f}" if G.number_of_nodes() > 0 else "No data",
              pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print additional statistics
    if G.number_of_nodes() > 0:
        print("\nTop 5 Companies by Number of Collaborations:")
        degrees = dict(G.degree())
        top_companies = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        for company, degree in top_companies:
            print(f"{company}: {degree} collaborations")

# %%
# Create a figure with subplots for each production type
plt.figure(figsize=(20, 15))

# Define production types
production_types = ['Super', 'Big', 'Small', 'Independent']

# Create subplot for each production type
for idx, prod_type in enumerate(production_types, 1):
    plt.subplot(2, 2, idx)
    
    # Filter data for this production type
    df_filtered = df_graph[df_graph['prod_type'] == prod_type]
    
    # Create edges list
    edges = create_edges_list(df_filtered)
    
    # Create network
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Get degree (number of collaborations) for each company
    degrees = dict(G.degree())
    
    # Sort companies by number of collaborations
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20 companies
    companies, num_collabs = zip(*sorted_degrees) if sorted_degrees else ([], [])
    
    # Create bar plot
    plt.bar(range(len(companies)), num_collabs, color='lightblue')
    plt.xticks(range(len(companies)), companies, rotation=45, ha='right')
    
    # Add labels and title
    plt.title(f'{prod_type} Productions\nTop Companies by Number of Collaborations')
    plt.ylabel('Number of Collaborations')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics for each production type
for prod_type in production_types:
    df_filtered = df_graph[df_graph['prod_type'] == prod_type]
    edges = create_edges_list(df_filtered)
    G = nx.Graph()
    G.add_edges_from(edges)
    degrees = dict(G.degree())
    
    print(f"\n{prod_type} Productions Summary:")
    print(f"Total companies: {len(degrees)}")
    print(f"Average collaborations per company: {sum(degrees.values())/len(degrees):.2f}")
    print("\nTop 5 most collaborative companies:")
    top_5 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for company, collabs in top_5:
        print(f"{company}: {collabs} collaborations")

        

# %%
# Create a figure with subplots for each production type
plt.figure(figsize=(20, 15))

# Define eras and production types
eras = ['pre', 'during', 'post']
production_types = ['Super', 'Big', 'Small', 'Independent']
colors = ['green', 'blue', 'red']  # One color for each era

# Create subplot for each production type
for prod_idx, prod_type in enumerate(production_types, 1):
    plt.subplot(2, 2, prod_idx)
    
    # Get collaboration data for each era
    era_data = []
    max_companies = 0
    
    for era in eras:
        # Filter data for this era and production type
        df_filtered = df_graph[
            (df_graph['dvd_era'] == era) & 
            (df_graph['prod_type'] == prod_type)
        ]
        
        # Create edges list and network
        edges = create_edges_list(df_filtered)
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Get and sort collaborations
        degrees = dict(G.degree())
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20
        era_data.append(sorted_degrees)
        max_companies = max(max_companies, len(sorted_degrees))
    
    # Plot grouped bars
    bar_width = 0.25
    r1 = np.arange(max_companies)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars for each era
    for idx, (era_degrees, r, color) in enumerate(zip(era_data, [r1, r2, r3], colors)):
        companies, collabs = zip(*era_degrees) if era_degrees else ([], [])
        plt.bar(r[:len(collabs)], collabs, 
               width=bar_width, 
               label=f'{eras[idx].upper()} DVD Era',
               color=color,
               alpha=0.7)
    
    # Add labels and customize plot
    plt.title(f'{prod_type} Productions\nTop Companies by Number of Collaborations')
    plt.xlabel('Company Rank')
    plt.ylabel('Number of Collaborations')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Adjust x-axis
    plt.xticks([r + bar_width for r in range(max_companies)], 
               [f'#{i+1}' for i in range(max_companies)],
               rotation=45)

plt.tight_layout()
plt.show()

# Print summary statistics
for prod_type in production_types:
    print(f"\n{prod_type} Productions Summary:")
    for era in eras:
        df_filtered = df_graph[
            (df_graph['dvd_era'] == era) & 
            (df_graph['prod_type'] == prod_type)
        ]
        edges = create_edges_list(df_filtered)
        G = nx.Graph()
        G.add_edges_from(edges)
        degrees = dict(G.degree())
        
        if len(degrees) > 0:
            print(f"\n{era.upper()} DVD Era:")
            print(f"Total companies: {len(degrees)}")
            print(f"Average collaborations: {sum(degrees.values())/len(degrees):.2f}")
            print("Top 3 companies:")
            top_3 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
            for company, collabs in top_3:
                print(f"  {company}: {collabs} collaborations")

# %%
# Create a figure with subplots for each production type
plt.figure(figsize=(20, 15))

# Define eras and production types
eras = ['pre', 'during', 'post']
production_types = ['Super', 'Big', 'Small', 'Independent']
colors = ['green', 'blue', 'red']  # One color for each era

# Create subplot for each production type
for prod_idx, prod_type in enumerate(production_types, 1):
    plt.subplot(2, 2, prod_idx)
    
    # Get collaboration data for each era
    era_data = []
    max_companies = 0
    
    for era in eras:
        # Filter data for this era and production type
        df_filtered = df_graph[
            (df_graph['dvd_era'] == era) & 
            (df_graph['prod_type'] == prod_type)
        ]
        
        # Get total number of movies for this era and production type
        total_movies = len(df_filtered)
        
        # Create edges list and network
        edges = create_edges_list(df_filtered)
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Get and sort collaborations, normalized by number of movies
        degrees = dict(G.degree())
        normalized_degrees = [(company, degree/total_movies) for company, degree in degrees.items()]
        sorted_degrees = sorted(normalized_degrees, key=lambda x: x[1], reverse=True)[:20]  # Top 20
        era_data.append(sorted_degrees)
        max_companies = max(max_companies, len(sorted_degrees))
    
    # Plot grouped bars
    bar_width = 0.25
    r1 = np.arange(max_companies)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars for each era
    for idx, (era_degrees, r, color) in enumerate(zip(era_data, [r1, r2, r3], colors)):
        companies, collabs = zip(*era_degrees) if era_degrees else ([], [])
        plt.bar(r[:len(collabs)], collabs, 
               width=bar_width, 
               label=f'{eras[idx].upper()} DVD Era',
               color=color,
               alpha=0.7)
    
    # Add labels and customize plot
    plt.title(f'{prod_type} Productions\nTop Companies by Normalized Collaborations')
    plt.xlabel('Company Rank')
    plt.ylabel('Collaborations per Movie')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Adjust x-axis
    plt.xticks([r + bar_width for r in range(max_companies)], 
               [f'#{i+1}' for i in range(max_companies)],
               rotation=45)

plt.tight_layout()
plt.show()

# Print summary statistics
for prod_type in production_types:
    print(f"\n{prod_type} Productions Summary:")
    for era in eras:
        df_filtered = df_graph[
            (df_graph['dvd_era'] == era) & 
            (df_graph['prod_type'] == prod_type)
        ]
        total_movies = len(df_filtered)
        edges = create_edges_list(df_filtered)
        G = nx.Graph()
        G.add_edges_from(edges)
        degrees = dict(G.degree())
        
        if len(degrees) > 0:
            normalized_degrees = {k: v/total_movies for k, v in degrees.items()}
            print(f"\n{era.upper()} DVD Era:")
            print(f"Total movies: {total_movies}")
            print(f"Total companies: {len(degrees)}")
            print(f"Average collaborations per movie: {sum(degrees.values())/total_movies:.2f}")
            print("Top 3 companies (collaborations per movie):")
            top_3 = sorted(normalized_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
            for company, collabs in top_3:
                print(f"  {company}: {collabs:.3f} collaborations per movie")

# %%
# Create figure
plt.figure(figsize=(15, 8))

# Define production types and colors for better visualization
production_types = ['Super', 'Big', 'Small', 'Independent']
colors = ['red', 'blue', 'green', 'purple']

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

# Print summary statistics for each era and production type
for prod_type in production_types:
    print(f"\n{prod_type} Productions Summary:")
    for era in ['pre', 'during', 'post']:
        df_filtered = df_graph[
            (df_graph['dvd_era'] == era) & 
            (df_graph['prod_type'] == prod_type)
        ]
        total_movies = len(df_filtered)
        
        if total_movies > 0:
            edges = create_edges_list(df_filtered)
            G = nx.Graph()
            G.add_edges_from(edges)
            
            avg_collaborations = G.number_of_edges() / total_movies if G.number_of_nodes() > 0 else 0
            
            print(f"\n{era.upper()} DVD Era:")
            print(f"Total movies: {total_movies}")
            print(f"Average collaborations per movie: {avg_collaborations:.2f}")

# %%
import statsmodels.api as sm
from scipy import stats

# Create DataFrame for analysis
analysis_data = []
for year in sorted(df_graph['release_year'].unique()):
    for prod_type in production_types:
        # Filter data
        df_filtered = df_graph[
            (df_graph['release_year'] == year) & 
            (df_graph['prod_type'] == prod_type)
        ]
        
        total_movies = len(df_filtered)
        
        if total_movies > 0:
            # Create network
            edges = create_edges_list(df_filtered)
            G = nx.Graph()
            G.add_edges_from(edges)
            
            # Calculate normalized collaborations
            collaborations = G.number_of_edges() / total_movies if G.number_of_nodes() > 0 else 0
            
            analysis_data.append({
                'year': year,
                'prod_type': prod_type,
                'collaborations': collaborations,
                'total_movies': total_movies
            })

analysis_df = pd.DataFrame(analysis_data)

# 1. Linear Regression Analysis for each production type
print("Linear Regression Analysis:")
for prod_type in production_types:
    data = analysis_df[analysis_df['prod_type'] == prod_type]
    
    # Prepare data for regression
    X = data['year']
    y = data['collaborations']
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X).fit()
    
    print(f"\n{prod_type} Productions:")
    print(f"R-squared: {model.rsquared:.3f}")
    print(f"P-value: {model.pvalues[1]:.3e}")
    print(f"Coefficient (slope): {model.params[1]:.3e}")
    
# 2. One-way ANOVA across DVD eras
print("\nOne-way ANOVA across DVD eras:")
for prod_type in production_types:
    # Group data by DVD era
    pre_data = df_graph[
        (df_graph['dvd_era'] == 'pre') & 
        (df_graph['prod_type'] == prod_type)
    ]
    during_data = df_graph[
        (df_graph['dvd_era'] == 'during') & 
        (df_graph['prod_type'] == prod_type)
    ]
    post_data = df_graph[
        (df_graph['dvd_era'] == 'post') & 
        (df_graph['prod_type'] == prod_type)
    ]
    
    # Calculate collaborations per movie for each era
    def get_collab_per_movie(data):
        if len(data) > 0:
            edges = create_edges_list(data)
            G = nx.Graph()
            G.add_edges_from(edges)
            return G.number_of_edges() / len(data) if G.number_of_nodes() > 0 else 0
        return 0
    
    pre_collabs = get_collab_per_movie(pre_data)
    during_collabs = get_collab_per_movie(during_data)
    post_collabs = get_collab_per_movie(post_data)
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway([pre_collabs], [during_collabs], [post_collabs])
    
    print(f"\n{prod_type} Productions:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.3e}")
    print("Mean collaborations per movie:")
    print(f"  Pre-DVD: {pre_collabs:.3f}")
    print(f"  During DVD: {during_collabs:.3f}")
    print(f"  Post-DVD: {post_collabs:.3f}")

# 3. Correlation Analysis
print("\nSpearman Correlation Analysis:")
for prod_type in production_types:
    data = analysis_df[analysis_df['prod_type'] == prod_type]
    correlation, p_value = stats.spearmanr(data['year'], data['collaborations'])
    
    print(f"\n{prod_type} Productions:")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"P-value: {p_value:.3e}")
