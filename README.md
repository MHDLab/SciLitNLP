# SciLitNLP 

Codes for natural language processing of scientific literature. To see some demos of the results see [here](https://mhdlab.github.io/nlp_reports/).

# Overview

This repo hosts codes to perform natural language processing on scientific literature datasets by generating language models and using those models in static and interactive visualizations. Some of the visualizations can be seen at [this website](https://mhdlab.github.io/projects/). The code is a work in progress designed to be able to generate these visualizations from various topics by searching through the [semantic scholar academic graph](https://api.semanticscholar.org/corpus) or other datasets. 

The main folder are the main steps in the data flow

 - text_data: Management and analysis of raw text data and metadata
    * codes for downloading/ocr to form text databases. Currently two datasets: SEAM pdf collection and the semantic scholar open research corpus
    * analysis of the text databases
    * codes to build citation maps from the semantic scholar metadata
- modeling: Development of NLP models
    * pipelines to generate NLP models: 
        * LDA and CorEx topic models
        * word2vec word vector model
    * analysis of topic model hyperparameters 
- visualization: of generated models 
    * Codes and notebooks to generate plots
    * Codes to generate bokeh-based web visualization html files (static)
- website_gen: code to run and consolidate the above pipeline into a report-style website. 

This repo was formed by putting together different NLP projects/repos formed over the last couple years. The main pipeline described below is under development but should be working, but there are many other scripts and notebooks in the code that still need to be updated with correct paths etc. to work.  

# Installation/Setup

First clone the repo to your machine with 

`git clone --recurse-submodules https://github.com/aspitarl/SciLitNLP/`

then enter the repository with 

`cd SciLitNLP`

There are `environment.yml` files with the conda environment that I've used to run the code. Setup the conda environment with

`conda env create -f environment_(OS).yml`

Where OS is the operating system. Note that this environment was mainly setup with the `conda-forge` channel so you will want to specify `-c conda-forge` when installing furhter packages with conda. 

once the environment is installed activate it with 

`conda activate scilitnlp`

We need to add the `nlp_utils` and `mat2vec` package to your python environment. These packages are submodules that should have been initialized with the `--recursive-submodules` flag. You need to go into the nlp_utils pacakge and installl it. If you use conda 

```
cd nlp_utils
python setup.py develop
cd mat2vec
python setup.py develop
```

Install nltk stopwords for text processing

`python -m nltk.downloader stopwords`

Make a file with the name `.env` in the in the base repodirectory (alongside README.md) for some scripts to work. The .env file defines environment variables (e.g. database path) that are used among various scripts, specifically scripts that use `os.getenv`. Here is a template for the contents file, you don't need to set the paths for databases you won't use. 

```
REPO_DIR = 'path/to/SciLitNLP/repository'
DB_FOLDER = 'path/to/databases/folder'
```



# Running the code

At the moment, **The python scripts in this repo are designed to be run within their respective folders** as they will output files (into gitignored folders) that are used in other scripts.  I use multiple command prompts open for different segments of the data flow pipeline, each open in their respective script's folder. Command line arguments are being added and are not stable but you can find how to run each script with the `-h` flag.

The file `website_gen\run_all.sh` is a shell script that will run through all of the data pipeline to produce a corex topic modeling report like the ones found [here](https://mhdlab.github.io/nlp_reports/). You can look in there to see the basic script pipeline. Note You can run this script on windows with [git bash](https://gitforwindows.org/) and following the commented websites in the script (specificlly '`jq` neeeds to be installed to parse `all_site_info.json`). 

## Setting up a text database
In general the codes are designed to run on data contained within sqlite databse files (extension `.db`). The `text_data` folder contains codes for generating these database files which it looks for in `DB_FOLDER` defined in the `.env` file.  The module `nlp_utils.fileio` has `load_df` functions to read these databases and output a pandas dataframe to a common format expected by the python codes. 

TODO: write a format specification for the output dataframe

### SEAMS

The OCR and database generation codes for the SEAMs have not been updated yet, but there is a `seams.db` file on the SEAMs sharepoint folder (`Publications/SEAMs`). The data is loaded with the `nlp_utils.fileio.load_df_SEAMs` function. You can test that file is loading correctly with the  `text_data/seams/seams_text_eda.py` file. 

### Semantic Open Corpus (soc)

To look through general literature metadata the semantic scholar dataset is the relevant dataset. The codes are looking for a database named `soc.db` which can be formed by following the readme in `text_data/semantic/database_gen/soc`

After setting up the database go into the semantic text data directory (`text_data/semantic`)
Then read the top words contained in the overall literature dataset with 

`python gen_general_lit_words.py`

Because the soc dataset is so large we need to generate a set of paper ids (`ids`) that will be used in other scripts to read the metadata of the relevant papers from the database with the `nlp_utils.fileio.load_df_semantic(db_path, ids)` function. There are two methods for generating a list of ids.

### paper ids from searching for a term

find the indices of papers containing a search term in the title or abstract. These indexed searches are then used in other scripts to read directly from the semantic database. 

`python gen_search gen_search_index_database.py -t "search term"` 

The relevant data is ouput into the `data` folder and used by other scripts. 

### Paper ids from citation network
literature dataset can also be generated by forming citation graphs with the `citation_network\graph_gen_cit_tree.py` script which will generate paper ids based on the formation of a citation network.

TODO: citation networks are not working.

## Modeling

The scripts in the `modeling` folder generate language models which are output into the their `models` folder. The most recently tested model is the corex model. For the `soc` dataset with an indexed search generated as above run. 

`python genmodel_soc.py -d search -t "search term" -tw`

Where search term is the same as searched for above. 

TODO: LDA models are not working

## Visualization

The `visualization` folder contains various static and interactive visualization methods. For an example, run the scripts in the topic nework folder which will generate a graph based visualization of models generated in the `models` folder. 

open the folder `visualization\topic_network` and run  
1. `python prep_data_corex.py -d soc`
2. `python gen_networkplot.py --show` 

the `-sn` flag of `gen_networkplot.py` can change node size.

# Development

command for creating environment (missing some packages)

`conda create -n scilitnlp -c conda-forge python=3.8 pandas numba xarray matplotlib networkx scikit-learn nltk spacy monty unidecode gensim bokeh fa2 appdirs pdfminer.six bs4 lxml DAWG python-crfsuite`
