
# databand

Website: https://databand-movies.streamlit.app/

## Project Structure
The directory structure of project:

```
├── data                        <- Project data files
│   ├── processed                       <- Clean files in csv
│   ├── raw                             <- Raw data files 
│   ├── website_data                    <- Small data files to create plots for the website
├── src                         <- Source code
│   ├── data                            <- Data directory
│   │   ├── process_data.py                 <- functions extract meaningful information and combine datasets
│   ├── models                          <- Model directory
│   │   ├── lda_model.py                   <- LDA model for topic extraction
│   │   ├── empath_model.py                <- Empath model for topic extraction
│   ├── utils                           <- Utility directory
│   │   ├── data_utils.py                   <- utility functions for data analyses 
│   │   ├── load_data.py                    <- utility functions to load dataframes and save them to csv files 
│   │   ├── plot_utils.py                   <- utility functions to plot data
│   │   ├── statistics_utils.py             <- utility functions for statistic analyses
│   ├── scripts                         <- Shell/Python scripts
│   │   ├── clean_data.py                   <- script to clean all datasets
│
├── data_overview_CMU.ipynb               <- notebook with an initial overview for movie features for CMU movies
├── data_overview_TMDB.ipynb              <- notebook with an initial overview for movie features for TMDB movies 
├── main_results_CMU.ipynb                <- notebook with main results for CMU movies
├── main_results_TMDB.ipynb               <- notebook with main results for TMDB movies, contains the bulk of analyses 
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

Our analyses are primarily focused for the TMDB dataset, thus the main_results_TMDB notebook should be examined first. 
Whenever possible, we perform the same analyses on the CMU dataset, but these provide less information due to the lower 
amount of data and the lack of data after 2012. 

Preliminary analyses on both datasets can be found in the "data_overview" notebooks.

# Abstract: The Rise and Fall of the DVD and it's impact on the movie industry

  The emergence of DVDs in the 1990s had a resounding impact on the film industry, providing a wider accessibility to movies 
and a new revenue stream. The popularity of DVDs meant that even if a movie did not perform well at the box office, studios could still generate revenue through sales of DVDs. Throuhgout the 2000s, however, the industry shifted away from physical media to a digital streaming industry, pushing studios to rely on successful theatrical releases notably, blockbusters.

In this project, we aim to characterize the impact of the DVD phases on the movie industry using the TMDB database. We created three distinct phases (eras) with the help of DVD release data:
1. The pre-DVD era
2. During the DVD era
3. The post-DVD era

Through our initial analysis, we examined shifts in financial aspects like budget and revenue, alongside shifts in the 
types of productions companies making movies. We then analyze how genre and theme preferences evolved across time. Ultimately, we aim to understand how changes in distribution models influenced the business and creative sides of filmmaking.
## Additional datasets
- [TMDB 1M Movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies): This dataset contains information about 1 million movies, such as release date, revenue, budget, genre, runtime, companies and countries of productions,
movie plots, and languages spoken in the movie. It contains movies between 1976 and 2023, which is particularly useful 
as it allows us to do an analysis of movies that were released pre-DVD era (<1997), during the DVD era (1997-2013), and post-DVD era (>2013).
  - Note: Due to the lack of data in the CMU dataset, we use the TMDB dataset as our main data source to perform comparisons between pre, during and post DVD eras.
- DVD rentals: This DVD rentals dataset allows us to approximate the trends of DVD popularity, which we use to define our cut-off dates
for pre-DVD versus post-DVD eras.

## Research questions 
1. Revenue: DVDs provide another revenue stream to production companies, but did they have 
   a noticeable effect on overall movie revenue e.g. does revenue increase with the rise of DVD sales? 

2. Budget: how did DVDs impact small, medium and large production studios?
   - *Subquestion 1* : Did DVDs level the playing field for low-budget movies by providing smaller production companies a way to distribute their movies without relying on costly theatrical releases? 
   - *Subquestion 2* : Did the fall of DVDs impact low and mid-budget movies ?
   - *Subquestion 3* : Did the decline in DVD sales influence production companies to release more "theatrical" i.e., high budget - high revenue, movies?

3. Production: 
   - *Subquestion 1* : Is there a change in dominant production companies during the different eras? 
   - *Subquestion 2* : Are new players entering the market post-DVD era? Is there a significant change in the production companies market share?

4. Genres: How did the rise and fall of the DVD era influence the emergence of new genres and the decline of older ones? 
   - *Subquestion 1*: Did the DVD era encourage new genres due to increased accessibility outside of theaters? 
   - *Subquestion 2*: How did DVD sales impact the genres of low-budget vs high-budget movies? 
   - *Subquestion 3*: are there differences in major themes within genres between the different DVD eras? For instance, does 
   the release of DVDs allow for more niche themes? Does the decline of DVD sales lead to more universal themes? 


## Methods
**Task 1: Revenue analysis**

To observe whether there are differences in revenue between the different eras, we first use visual
representations such as comparing frequency distribution and cumulative revenue distributions between eras.
To determine whether such differences are significant, we use statistical tests such as t-test to qualify the difference
mean or median revenue and chi-square test to evaluate the differences in revenue distributions between DVD eras. 

**Task 2: Budget analysis** 

We first use graphical assessments to observe shifts in budgets between the DVD eras: 
- Plot the mean budget across time, corrected for inflation or not
- Plot histograms to visualize the distribution of budgets across DVD eras ; this is done in a cross-era manner to visualize the shifts in budget distribution. 
- Calculate the rolling discrete derivative of the mean budget with a window of three years, providing insights into the rate of change in movie budgets over time. 
- Display the proportion of tiny, small, big and super-productions (for budgets ranging in [0, 0.2], [0.2, 1], [1, 5] and [5, ..] times the budget average).

Statistical tests are then conducted to confirm the graphical observations :
- Evolution of budget through the years: linear regression is used to assess the importance of release years in explaining 
the budget and ANOVA tests is used to show which part of budget variability can be explained by the release year. 
- Influence of DVD: chi-square tests are used to examine the differences in budget distributions between pre, during and post-dvd era. 
Again, an ANOVA test complements this analysis.

**Task 4: Production analysis**

We analyse production companies collaboration through co-producing movies, considering different production types, over time. 
This is first done through line plots displaying the average number of companies per movies of each production types, then
through statistical analyses such as linear regression and Spearman's correlation tests. 
Then, to examine which companies co-produce movies, we create a collaboration graph where companies that produced together are linked,
allowing us to observe production company clusters.

**Task 5: Genre analysis**

- Genre Emergence and Decline Analysis:
We analyze genre popularity over time by computing the proportion of movies per genre in each DVD era (pre, during, and post). 
A chi-square test is used to assess the statistical significance of shifts in genre distribution.
- High-Revenue Movie Distribution Analysis:
We identify the top 10% of high-revenue films by genre and era, then calculate the mean revenue for each genre, corrected 
for inflation. A bar chart is used to visualize genre contributions to total revenue.

**Task 6: Common themes per genre**
For each of the pre-DVD, during-DVD, and post-DVD eras, movie plots from the top genres are extracted and analyzed. 
First, pre-processing is performed by removing stopwords and special characters. 
Then, leveraging the Empath model, the most common topics are extracted from movie plots for each genre-era.

## Team contributions
- Yann Ravot: revenue analysis 
- Abdul Karim Mouakeh: genre analysis
- Charlotte Meyer: production companies analysis and website creation
- Nicolas Moyne: revenue analysis and website creation
- Charlotte Sacré: topics extraction, production countries analysis, and data story redaction