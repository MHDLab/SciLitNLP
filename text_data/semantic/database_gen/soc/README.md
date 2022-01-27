The code in this folder downloads the [Semantic Scholar Academic Graph](https://api.semanticscholar.org/corpus/) and outputs a sqlite databse file `soc.db` used in the code. 

# 1. Download semantic scholar .gz files
Semantic scholar provides data as .gz files and they tell how to download them [here](https://api.semanticscholar.org/corpus/download/). These should go into a directory names `temp_gz`. `download_gz.sh` is a shell script for this purpose. Each .gz file is ~30 Mb and there are approximately 6000 of them, so the entire dataset is very large. There is a `MAX_FILES` variable in the script that allows for only downloading a few of these so the code can be tested with a small dataset before downloading the entire thing. 

# 2. decompress the .gz files and form the database

These steps are performed by the `decompress_gz_and_filter.py` script. 

Then move (or not) the `soc.db` file into the DB_FOLDER path set in the `.env` file.

You can insure the database is being read correctly by running the `text_data/semantic/analysis/SOC_eda.py` script