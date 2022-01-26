Codes for natural language processing of scientific literature

## Overview

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

This repo was formed by putting together different NLP projects/repos formed over the last couple years. Some things were heavily reorganized so I decided to restart the version history for everything. Most of the main pipeline and visualizations are working but many old analysis notebooks need to be updated. 

# Using the code

## Installation

First clone the repo to your machine with 

`git clone --recurse-submodules https://github.com/aspitarl/SciLitNLP/`

then enter the repository with 

`cd SciLitNLP`

There are `environment.yml` files with the conda environment that I've used to run the code. Setup the conda environment with

`conda env create -f environment_nobuild.yml`

Note that this environment was mainly setup with the `conda-forge` channel so you will want to specify `-c conda-forge` when installing furhter packages with conda. 

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

A .env file need to be created in the base repodirectory (alongside README.md) for some scripts to work. The .env file defines environment variables (e.g. database path) that are used among various scripts, specifically scripts that use `os.getenv`. Here is a template for the file, you don't need to set the paths for databases you won't use. 

```
REPO_DIR = 'path/to/SciLitNLP'
SOC_DB_DIR = 'path/to/soc/database'
```



## Running the code

**The python scripts in this repo are designed to be run within their respective folders** as they will output files (into gitignored folders) that are used in other scripts. So open a command promt with the previous conda environment activated, change direct into the folder containing the script (`cd to/a/folder/path/`) and run `python the_script_name.py`. I use VSCode with multiple command prompts open for different segments of the data flow pipeline, each open in their respective script's folder.

This repo is in an initial phase and many file paths are specific to my machine. My intention is to relace all of these explicit filepaths with relative paths once the folder structure is a little more solidified but for now you will have to just replace the filepaths with the filepath on your machines. 

## Setting up and analyzeing the text data 
In general the codes are designed to run on data contained within sqlite databse files. The `text_data` folder contains codes for generating these dabase files. The code was deveoped from a collection of OCRed pdfs (seams) and the semantic scholar open research corpus dataset. 

To look through general literature metadata the semantic scholar dataset is the relevant dataset. The dataset is very large and takes time to download and setup though It should be straightforward to download just a subset of the database though to get started. 

The script to download and setup the database is in `text_data\semantic\database_gen\soc\decompress_gz_and_filter.py`

After setting up the sqlite database the baisc pipeline to get to topic modeling involves two scripts:

`text_data\semantic\gen_general_lit_words.py` finds the top words within the overall literature dataset, which can be used as stopwords later. 
`text_data\semantic\gen_search_index_database.py` find the indices of papers containing a search term in the title or abstract. 

The relevant data is ouput into the `data` folder and used by other scripts. 

literature dataset can also be generated by forming citation graphs with the `text_data\semantic\citation_network\graph_gen_cit_tree.py` script.

## Modeling

The scripts in the `modeling` folder generate language models which are output into the their `models` folder. The most tested model is the `lda` Latent Dirichichlet Allocation topic model. To generate the lda model `modeling\lda\lda_pipeline.py` and change the input dataset (currently looking at indexed search generated above) and other hyperpapermeter like number of topics. 

## Visualization

The `visualization` folder contains various static and interactive visualization methods. For an example, run the scripts in the topic nework folder which will generate a graph based visualization of models generated in the `models` folder. 

First run the script `visualization\topic_network\prep_data_lda.py`
Then `visualization\topic_network\gen_networkplot.py` which will output a html file for visualizaiton 

# Development

command for creating environment (missing some packages)

`conda create -n scilitnlp -c conda-forge python=3.8 pandas numba xarray matplotlib networkx scikit-learn nltk spacy monty unidecode gensim bokeh fa2 appdirs pdfminer.six bs4 lxml DAWG python-crfsuite`