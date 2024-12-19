import pandas as pd
import statsmodels.api as sm
from data_utils import get_collab_per_movie
from scipy import stats


def lin_reg_prod_companies(production_types, analysis_df):
    results = []
    for prod_type in production_types:
        data = analysis_df[analysis_df['prod_type'] == prod_type]

        # Prepare data for regression
        X = data['year']
        y = data['collaborations']
        X = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X).fit()

        # Store results
        results.append({
            'Production': prod_type,
            'R-Squared': model.rsquared,
            'p_value': model.pvalues.iloc[1],
            'Coefficient (slope)': model.params.iloc[1]
        })

    return pd.DataFrame(results)



def one_way_anova_prod_companies(production_types, df_graph):
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


def spearman_prod_companies(production_types, analysis_df):
    result = []
    for prod_type in production_types:
        data = analysis_df[analysis_df['prod_type'] == prod_type]
        correlation, p_value = stats.spearmanr(data['year'], data['collaborations'])
        result.append({
            'Production': prod_type,
            'Correlation coefficient': correlation,
            'p-value': p_value
        }
        )
    return (pd.DataFrame(result))