import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_proportions_per_era(top_k_df, column, k):
    pivot_data = top_k_df.pivot_table(index='dvd_era', columns=column, values='proportion', aggfunc='sum',
                                      observed=True).fillna(0)
    palette = sns.color_palette(cc.glasbey, n_colors=len(pivot_data.columns))

    # use stacked bar plot
    pivot_data.plot(kind='bar', stacked=True, figsize=(8, 6), color=palette)

    plt.title('Proportions of top ' + str(k) + " " + column + ' by DVD Era', fontsize=16)
    plt.xlabel('DVD Era', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.legend(title=column, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()