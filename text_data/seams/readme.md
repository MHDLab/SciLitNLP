


# Installation

navigate to the repository folder in a anacnoda prompt and create the environment for this package with `conda env create -f environment.yml`

Install pytesseract windows binary [here](https://github.com/UB-Mannheim/tesseract/wiki), you may have to change the filepath designation for the tesseract excecutable at the top of `nlputils\ocr.py`

Install the scipsacy 'en_core_sci_lg' as described [here](https://github.com/allenai/scispacy)

# Instructions

## Building the sql database

to generate the database from scratch:

run `dbgen.py` to initialize the sql database

run `perform_ocr.py` to ocr the texts. You need to update the path to the SEAMs pdfs in the script.

run `perform_textprocessing.py` to clean and prepare the ocred text for vectorizaiton

run `perform_vector.py` to vectorize and cluster the documents

## Working with the data

use `text_recomendation.py` with `data\input_text.txt` to get a reccomended text csv file `data\recommendations.csv`.

run `bokeh serve bokehplot --show` to start the clustered document bokehplot. 