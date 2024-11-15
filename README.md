
# databand

## Project Structure
The directory structure of project:

```
├── data                        <- Project data files
│   ├── processed                       <- Clean files in csv
│   ├── raw                             <- Raw data files 
├── src                         <- Source code
│   ├── data                            <- Data directory
│   │   ├── process_data.py                 <- functions extract meaningful information and combine datasets
│   ├── models                          <- Model directory
│   │   ├── lda_model.py                   <- LDA model for topic extract 
│   ├── utils                           <- Utility directory
│   │   ├── data_utils.py                   <- utility functions for data analyses 
│   │   ├── load_data.py                    <- utility functions to load dataframes and save them to csv files 
│   │   ├── plot_utils.py                   <- utility functions to plot data
│   │   ├── statistics_utils.py             <- utility functions for statistic analyses
│   ├── scripts                         <- Shell/Python scripts
│   │   ├── clean_data.py                   <- script to clean all datasets
│
├── data_overview_CMU.ipynb               <- notebook with an overview for movie features for CMU movies
├── data_overview_TMDB.ipynb              <- notebook with an overview for movie features for TMDB movies 
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

Our analyses are primarily focused for the TMDB dataset, thus the data_overview_TMDB notebook should be examined first. 
Whenever possible, we perform the same analyses on the CMU datatset, but these provide less information due to the lower 
amount of data and the lack of data after 2012. 

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
1. Revenue: 
   - *Subquestion 1*: DVDs provide another revenue stream to production companies, but did they have 
   a noticeable effect on overall movie revenue e.g. does revenue increase with the rise of DVD sales? 
   - *Subquestion 2*: Are there categories of movies (genres, budget, ...) for which revenue significantly changes from one era to another

2. Budget: how did DVDs impact small, medium and large production studios?
   - *Subquestion 1* : Did DVDs level the playing field for low-budget movies by providing smaller production companies a way to distribute their movies without relying on costly theatrical releases? 
   - *Subquestion 2* : Did the fall of DVDs impact low and mid-budget movies ?
   - *Subquestion 3* : Did the decline in DVD sales influence production companies to release more "theatrical" i.e., high budget - high revenue, movies?
   - *Subquestion 4* : Are movies more expensive today?

3. Production: 
   - *Subquestion 1* : Is there a change in dominant production companies during the different eras? 
   - *Subquestion 2* : Are new players entering the market post-DVD era? Is there a significant change in the production companies market share?

4. Genres: How did the rise and fall of the DVD era influence the emergence of new genres and the decline of older ones? 
   - *Subquestion 1*: Did the DVD era encourage new genres due to increased accessibility outside of theaters? How does this compare to the rise of streaming services? 
   - *Subquestion 2*: How did DVD sales impact the genres of blockbuster movies? Can we observe a shift to typical,
   mainstream genres for such movies after sales decline? 
   - *Subquestion 3*: are there differences in major themes within genres between the different DVD eras? For instance, does 
   the release of DVDs allow for more niche themes? Does the decline of DVD sales lead to more universal themes? 


## Methods
**Task 1: Revenue analysis**

To observe whether there are differences in revenue between the different eras, we first use visual
representations such as comparing frequency distribution and cumulative revenue distributions between eras.
To determine whether such differences are significant, we use statistical tests such as t-test to qualify the difference
mean or median revenue and chi-square test to evaluate the differences in revenue distributions between DVD eras. 

Using similar methods, we also examine the differences in profitability, computed as $profitability = revenue - budget$. 

**Task 2: Budget analysis** 

We first use graphical assessments to observe shifts in budgets between the DVD eras: 
- Plot the mean budget across time, corrected for inflation or not
- Plot histograms to visualize the distribution of budgets across DVD eras ; this is done in a cross-era manner to visualize the shifts in budget distribution. 
- Calculate the rolling discrete derivative of the mean budget with a window of three years, providing insights into the rate of change in movie budgets over time. 
- Display the proportion of tiny, small, big and super-productions (for budgets ranging in [0, 0.2], [0.2, 1], [1, 5] and [5, ..] times the budget average).

Statistical tests will then be conducted to confirm the graphical observations :
- Evolution of budget through the years: linear regression will be used to assess the importance of release years in explaining 
the budget and ANOVA tests will be used to show which part of budget variability can be explained by the release year. 
- Influence of DVD: chi-square tests will be used to examine the differences in budget distributions between pre, during and post-dvd era. 
Again, an ANOVA test will complement this analysis.

**Task 4: Production analysis**

To analyze the evolution of production companies we started with an exploration the data. This was important to understand the top players and how they compare with others (i.e BBC, Warner Bros, etc.). We used bar and line plots for a visual reprensentation. We also found a strong correlation between of the number production companies for one movie and time, suggesting that movies need more investments today with more production companies. The next step to correctly categorize them is using clustering techniques to group the production companies based on their movie production activity and genre they produce. This will better help us observe trends during the different eras. 

**Task 5: Genre analysis**

- Genre Emergence and Decline Analysis:
We will analyze genre popularity over time by calculating the proportion of movies per genre in each DVD era (pre, during, and post), rather than just the count. This will account for the unequal number of movies across eras. Line and area charts will visualize these trends, showing the relative share of each genre in each era. A chi-square test will assess the statistical significance of shifts in genre distribution, and clustering techniques will help identify patterns of genre emergence and decline across eras.
- High-Revenue Movie Distribution Analysis:
We will identify the top 10% of high-revenue films by genre and era, then calculate the mean revenue for each genre, corrected for inflation. Stacked bar charts will visualize genre contributions to total revenue, and percentage change analysis will track how the share of high-revenue films shifted from the pre-DVD to post-DVD era.

**Task 6: Common themes per genre**

For each of the pre-DVD, during-DVD, and post-DVD eras, movie plots from the top genres will 
be extracted and analyzed. First, pre-processing will be performed by removing stopwords and special characters. 
Then, with the Latent Dirichlet Allocation model from the `gensim` library, the most common topics will be extracted 
from movie plots for each genre-era. 

## Timeline & Deadlines
- Milestone 1 (15-29 Nov): Perform all analyses from methods section
- Milestone 2 (29 Nov - 6 Dec): Select most relevant findings for data story
- Milestone 3 (6-13 Dec): 
  - 3a: Compose data story
  - 3b: Create github.io report template with figures 
- Milestone 4 (13-20 Dec): Combine data story and figures in report

20.12.2024: P3 Deadline 

## Team contributions
- Revenue analysis: Yann Ravot
- Budget analysis: Nicolas Moyne
- Production analysis: Charlotte Meyer
- Genre analysis: Abdul Karim Mouakeh
- Theme extractions: Charlotte Sacré
- Data story: Charlotte Sacré, Yann Ravot & Abdul Karim Mouakeh
- Report website: Charlotte Meyer & Nicolas Moyne

## Questions for TA
For the CMU dataset, we perform an inner merge with the TMDB dataset (based on movie title and release year) to obtain 
information such as budget and production companies for CMU movies and perform analyses similar to the TMDB movies. 
However, this significantly reduces the amount of movies to analyse (around 17,000 results movies). As a result, we were 
wondering which of the two options would be preferable for P3:
- Do the inner merge and perform all analyses (e.g. on budget & production companies) on the restricted set of movies 
- Perform a restricted set of analyses using the available features from the CMU dataset on a larger set of movies 

