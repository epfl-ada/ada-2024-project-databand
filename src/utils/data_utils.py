import pandas as pd


def top_proportions_per_era(df, column, k, islist=False):
    # get total number movies grouped by era and column
    if islist:
        counts = df.explode(column).groupby(['dvd_era', column]).size().reset_index(name='count')
    else:
        counts = df.groupby(['dvd_era', column]).size().reset_index(name='count')

    # get proportion
    total_counts = counts.groupby('dvd_era')['count'].transform('sum')
    counts['proportion'] = counts['count'] / total_counts

    # get top-k for each era
    top_k = counts.sort_values(by=['proportion'], ascending=[False]).groupby('dvd_era').head(k)
    category_order = ['pre', 'during', 'post']
    top_k['dvd_era'] = pd.Categorical(top_k['dvd_era'], categories=category_order, ordered=True)

    return top_k

def adjust_inflation(df, old_col="budget", new_col="budget_inflation", inflation_file="data/cpi_inflation.csv"):
    """
    Adjusts the budget of movies for inflation.
    """
    # Read the inflation file
    inflation = pd.read_csv(inflation_file)
    
    # Calculate the cumulative inflation factor
    inflation['Cumulative_Inflation'] = (1 + inflation['CPALTT01USM657N'] / 100).cumprod()
    
    # Extract the year from the date
    inflation['Year'] = pd.to_datetime(inflation['DATE']).dt.year
    
    # Create a dictionary for inflation factors
    inflation_dict = inflation.set_index('Year')['Cumulative_Inflation'].to_dict()
    
    # Adjust the budget for inflation
    df[new_col] = df.apply(lambda row: row[old_col] * inflation_dict.get(row['release_year'], 1), axis=1)

    return df
