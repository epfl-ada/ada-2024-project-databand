
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

