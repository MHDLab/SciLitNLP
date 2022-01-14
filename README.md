Codes for natural language processing of scientific literature.

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