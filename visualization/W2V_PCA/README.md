## Description:
In natural language processing, words are often embedded into numerical vectors that encode the meanings of the words. The embeddings are learned from the context of surrounding words using machine learning algorithms. We can easily compare how similar words are by comparing how close their word vectors are. Another useful feature of word vectors is that you can add and subtract word vectors to search for relevant words. We've plotted some of the top word vectors on a 2D plane, and provided a tool for users to do vector math with words in our corpus.

## Goal: 
The goal of this visualization is to be able to choose words out of the collection of abstracts and explore their relationships either by seeing how close they are on the two-dimensional visualization, or by experimenting with vector math.

This repo contains code for generating a word2vec model and using it to plot words in the dataset using tSNE.

## Instructions:
1. Activate the nlp_3 environment
2. Make sure that you have run python setup.py develop in the nlp_utils repository 
3. Change file paths used to save and load data for your local computer in all files you run
4. Run word_embed_genmodel.py to generate the model.  The code needs to be pointed at the text data in the 'ES_TextData' repo, and the model should come from the w2v_models subfolder.
5. Create the visualization by running gen_w2vplot.py

## To do parameter testing:
1. Go to the parameter_testing folder
2. Change file paths used to save and load data for your local computer in the parameter_testing.py
3. Run parameter_testing.py.  This may take a few hours.  
 
It will output a file called paraemeter_testing.txt containing sample equations for each parameter setting and a set of plots
