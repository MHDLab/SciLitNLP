# NLP-Semantic

Codes for natural language processing on Semantic Scholar datasets. 

s2orc - [The Semantic Scholar Open Research Corpus 2](https://github.com/allenai/s2orc). This dataset is newer and includes full paper OCR texts, but the citation data seems to be sparse, making citation maps difficult.
 
soc - [The Semantic Scholar Open Research Corpus 1](https://api.semanticscholar.org/corpus/). This dataset is a little older and does not have full paper OCRs, but the citation data is more complete. 

## Instructuions

1. Download the relevant dataset which are both hosted online as a series of `.gz` files on the order of ~100 Gb. 
2. Convert the `.gz` files into a sqlite database for easy access with the `decompress_gz_and_filter.py` scripts. These also allow for filtering based on the Microsoft Academic topic group. I in particular filtered out medical papers which are a huge component of the data. 
3. run the scripts in `topic_graph_vis`. corpus_gen -> topic_modeling -> prep_data -> gen_networkplot. 