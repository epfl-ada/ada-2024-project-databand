
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
as it allows us to do an analysis of movies that were released pre-DVD era (<1990), during the DVD era (1990-2008), and post-DVD era (>2008).
- DVD sales

## Research questions 
1. Money - budget & revenue: 
2. Genres:
3. Themes per genre: are there shifts in major themes within genres between the different DVD eras? For instance, does
the release of DVDs allow for more niche themes? Does the decline of DVD sales lead to more universal themes in blockbuster movies? 
4. Production: 

## Methods
1. Budget analysis
2. Revenue analysis 
3. Production analysis
4. Genre analysis
5. Common themes analysis: for each of the pre-DVD, during-DVD, and post-DVD eras, movie plots from the top genres will 
be extracted and analyzed. First, pre-processing will be performed by removing stopwords and through lemmatization to 
reduce words to their base form. Then, word clouds will be used to display the most common words in movie plots for each
genre-era, giving an indication of common themes. 

## Timeline & Milestones
20.12.2024: P3 Deadline 

## Questions for TA

