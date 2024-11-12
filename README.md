
# databand
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.

data: 
- data/raw: raw files given from the CMU movie dataset
- data/processed: relevant files put in csv format, including column headers as specified by dataset README

src:
- src/data: scripts to load and process the data 
  - src/data/load_data.py: functions to load raw files from data/raw and save into correct format into data/processed
  - src/data/process_data.py: functions to pre-process data and extract meaningful information e.g. values from FreebaseID:name tuples

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│   ├── processed                       <- Files in csv format with headers 
│   ├── raw                             <- Raw data files from CMU dataset
├── src                         <- Source code
│   ├── data                            <- Data directory
│   │   ├── load_data.py 
│   │   ├── process_data.py 
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

# <Project title>

The emergence of DVDs in the 1990s had a major impact on the film industry by providing a wider accessibility to movies 
and a shift in revenue streams, reducing the reliance on profitable theatrical releases. Then, the shift away from physical 
media to digital streaming in the late 2000s reshaped the industry again, pushing studios to focus on streaming licenses 
and successful theatrical runs with blockbusters. In this project, we aim to characterize the impact of the rise 
and fall of the DVD on the movie industry using the TMDB database. Considering three distinct phases - pre-DVD, during the 
DVD era, and post-DVD, we first examine shifts in key financial aspects like budget and revenue, alongside shifts in the 
types of productions. We then analyze how genre and theme preferences evolved across time. Ultimately, we aim to understand 
how changes in distribution models influenced the business and creative sides of filmmaking. 

## Additional datasets
- [TMDB 1M Movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies): This dataset 
contains information about 1 million movies, such as release date, revenue, budget, genre, runtime, companies and countries of productions,
movie plots, and languages spoken in the movie. It contains movies between 1976 and 2023, which is particularly useful 
as it allows us to do an analysis of movies that were released pre-DVD era (<1997), during the DVD era (1997-2008), and post-DVD era (>2008).
- DVD sales

## Research questions 

?? Are movied more expensive to make today??
1. Money - budget & revenue: 
2. Genres: How did the rise and fall of the DVD era influence the emergence of new genres and the decline of older ones? Hypothesis 1: The DVD era encouraged new, niche genres due to increased accessibility at home. Hypothesis 2: The shift to streaming favored mainstream genres, causing niche genres to decline.
How did the distribution of high-revenue movies within each genre change from the pre-DVD era to the post-DVD era? Hypothesis 1: The DVD era spread high-revenue success across more genres due to repeat home viewings. Hypothesis 2: The streaming era concentrated high-revenue movies in blockbuster genres like action and superhero films.
4. Themes per genre: are there shifts in major themes within genres between the different DVD eras? For instance, does
the release of DVDs allow for more niche themes? Does the decline of DVD sales lead to more universal themes in blockbuster movies? 
5. Production: Are major franchise production companies more successful in the post-DVD era? Are we seeing more of the same productions companies (for example Marvel Studios) or are new players still entering the market ?
6. How does the DVD rise and downfall influenced the way movies are produced ? *Hypothesis 1* : DVD emergence allowed smaller budget films to gain feasability as people could consume them outside cinemas (which were costly and therefore encouraged people to chose well-known high-budget movies). *Hypothesis 2* : the DVD downfall due to streaming killed both smaller budget movies (that would not be good enough to be bought by streaming services) and superproductions (that would be too costly to be bought by streaming services).

## Methods
1. Budget analysis : this analysis relies heavily on plots (graphical assessment).
- Compute the mean of movie budgets and plot it accross time, corected for inflation or not
- Plot histograms to visualize the distribution of budgets across these eras ; this is done in a cross-era manner to visualize the shifts in budget distribution. 
- Calculate the rolling discrete derivative of the mean budget with a window of three years, providing insights into the rate of change in movie budgets over time. 
- Display the proportion of tiny, small, big and superproductions (for budgets ranging in [0, 0.2], [0.2, 1], [1, 5] and [5, ..] times the average).
2. Revenue analysis 
3. Production analysis
4. Genre analysis
5. Common themes analysis: for each of the pre-DVD, during-DVD, and post-DVD eras, movie plots from the top genres will 
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

## Questions for TA

