from re import I
from typing import Text
from gensim.models import Word2Vec
import os

'''Word similarity
In a word2vec model each word is represented as a vector. I understand the axes/components of this vector space as abstract 'meanings'

we check most similar words to a given word by finding words that are the closest in the vector space.'''

'''The basic application with word vectors is to be able to quantify the relationships between words. The most often used example is finding the word that has the same relationship to 'man' as 'queen' has to 'woman' (king).

https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/

We might be able to use this to find relatonships between different technologies or physical concepts. I attempt this below without much sucess to try and find the material that is used in Li-ion battery anodes from the material that is used in cathodes. This is inspired by the Tshitoyan 2019 paper.

'''

#PCA word visualization
# We can visualize a set of words together by projecting their vectors into a 2D plane using Principal Components Analysis. 
# PCA is commonly used to visualize higher dimensional datasets.

#import bokeh tools
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, TapTool, CustomJS, Div, Button, MultiChoice, Range1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

output_file("output/W2V_tSNE.html", title="Word2Vec visualized with tSNE")

title_info = Div(text="""<h1>Word Embedding Visualization </h1> <br> 
<h2> Words from the abstracts were embedded to 100 dimensional vectors using the word2vec algorithm
These plots represent the proximity of these word vectors projected to 2D space using tSNE.
Similar words are closer together.<br><br> </h2>""")

#Choose model to visualize
mod = Word2Vec.load(os.path.join(os.getenv('REPO_DIR'), r'modeling\word2vec\models\word2vec_semantic.model'))
#pca= PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=0)
#Generate a vector representation (X) of a collection of words
vocab = list(mod.wv.key_to_index.keys())
# clean up vocab by removing words that are just punctuation (takes off ~400KB)
embedded_vocab = mod.wv[vocab]
nearest_vocab = []
#adding the nearest vocab pushes the file size to 25,685KB which is 686KB too large
for word in vocab:
    s=''
    nearest_vocab.append(s.join((t[0] + ', ') for t in mod.wv.most_similar(word)))

#vocab_2D = pca.fit_transform(embedded_vocab)
vocab_2D = tsne.fit_transform(embedded_vocab)
x_all = vocab_2D[:,0].tolist()
y_all = vocab_2D[:,1].tolist()


p_div = Div(text="""<h3> You can add and remove words to the plot and see a new projection based on selected words </h3>""")
word_math = Div(text="""<h3> You can do math with the word vectors by selecting positive and negative words.  
The result will be the word vector with the closest cosine similarity to the sum of all positive and negative word vectors.<br>
 (Note, cosine similarity does not consider magnitude and is calculated in the original 100 dim space, not the 2D space displayed here.)</h3>""")
#display the similar words when a point is selected
sim_words_text = Div(text="""<h2>Hover over or select a word to see the most similar words in the vocabulary.</h2>""")

top_words = mod.top_words #added this custom attribute in gen model script 

nearest_words = []
x_top = []
y_top = []
for word in top_words:
    idx = vocab.index(word)
    x_top.append(x_all[idx])
    y_top.append(y_all[idx])
    nearest_words.append(nearest_vocab[idx])
#    nearest_words.append(s.join((t[0] + ', ') for t in mod.wv.most_similar(word)))


plot_source = ColumnDataSource(data=dict(
    x=x_top,
    y=y_top,
    plot_words=top_words,
    plot_colors = ['#3270C8']*len(x_top),
    similar_words=nearest_words,
))

eq_out_word = ColumnDataSource(data={'output_word': ['']}) 

callback_args = dict(
    source=plot_source,
    vocab=vocab,
    x_all = x_all,
    y_all = y_all,
    nearest_vocab=nearest_vocab,
    p_div=p_div,
    eq_out_word = eq_out_word
)

callback_add= CustomJS(args=callback_args, code="""
        var data = source.data;
        var words_list;
        if (eq_out_word.data['output_word'][0] == ''){
            words_list = cb_obj.value;
        } else {
            words_list = eq_out_word.data['output_word'];
            eq_out_word.data['output_word'] = [''];
        }
        var new_word;
        
        function find_word(word) {
            return word == new_word;
        }

        for (var i=0; i<data['plot_colors'].length; i++){
            data['plot_colors'][i] = '#3270C8';
        }

        for (var i=0; i<words_list.length; i++) {
            new_word = words_list[i];
            if (data['plot_words'].includes(new_word)) {
                var word_index = data['plot_words'].findIndex(find_word);
                data['plot_colors'][word_index] = "#FF0083";
                p_div.text = '<h3>' + new_word + ' has already been plotted.</h3>';
                p_div.change.emit();
            } else {
                console.log(new_word);
                
                for (var i=0; i<data['plot_colors'].length; i++){
                    data['plot_colors'][i] = '#3270C8';
                }
                //add to plot_words and regenerate plot?
                data['plot_words'].push(new_word);
                var word_index = vocab.findIndex(find_word);
                data['x'].push(x_all[word_index])
                data['y'].push(y_all[word_index])
                data['plot_colors'].push('#FF0083');
                data['similar_words'].push(nearest_vocab[word_index]);
                

            }
        }
        source.change.emit();
        """
    )

callback_remove= CustomJS(args=callback_args, code="""
        var data = source.data;
        var words_list = cb_obj.value;
        var new_word;
        
        function find_word(word) {
            return word == new_word;
        }

        for (var i=0; i<data['plot_colors'].length; i++){
            data['plot_colors'][i] = '#3270C8';
        }

        for (var i=0; i<words_list.length; i++){
            new_word = words_list[i];
            if (data['plot_words'].includes(new_word)) {
                
                var word_index = data['plot_words'].findIndex(find_word);
                data['plot_words'].splice(word_index, 1);
                data['similar_words'].splice(word_index, 1);
                data['x'].pop();
                data['y'].pop();
                data['plot_colors'].pop();

            } else {
                p_div.text = '<h3>' + new_word + ' is not currently plotted. </h3>';
                p_div.change.emit();
            }
        }
        source.change.emit()
        """
    )

callback_rem_all = CustomJS(args=callback_args, code="""
    var data = source.data;
    var num_words = data['x'].length;
    for (var i=0; i<num_words; i++){
        data['x'].pop();
        data['y'].pop();
        data['plot_colors'].pop();
        data['plot_words'].pop();
        data['similar_words'].pop();
    source.change.emit();
    }
""")

tap_args = dict(
    source=plot_source,
    sim_words_text = sim_words_text
)

tap_callback = CustomJS(args=tap_args, code="""
    var data = source.data;
    if (source.selected.indices.length == 0){
        sim_words_text.text = '<h2>Hover over or select a word to see the most similar words in the vocabulary.</h2>';

        for (var i=0; i<data['plot_colors'].length; i++){
            source.data['plot_colors'][i] = '#3270C8';
        }
    } else {
        var ind = source.selected.indices[0];
        var word = source.data['plot_words'][ind];
        var similar = source.data['similar_words'][ind];
        sim_words_text.text = '<h2> Similar words to ' + word + ': ' + similar + '</h2>';
        
        data['plot_colors'][ind] = "#FF0083";
    }
    source.change.emit();
""")

# create a scatter plot of the projection
p = figure(title='Word Vectors Projected to 2D', plot_width=900, plot_height=900) #, x_range=Range1d(-35,35))
p.scatter(source=plot_source, x="x", y="y", size=10, color="plot_colors", alpha=0.9)
p.xaxis[0].axis_label = 'First Component'
p.yaxis[0].axis_label = 'Second Component'
p.x_range = Range1d(-80,80)
p.y_range = Range1d(-80,80)

labels = LabelSet(x='x', y='y', text='plot_words', x_offset=5, y_offset=5, source=plot_source, render_mode='canvas')
node_hover_tool = HoverTool(tooltips=[("word", "@plot_words"), ("similar", "@similar_words")])
p.add_tools(node_hover_tool, TapTool())#(callback=tap_callback))
p.add_layout(labels)
plot_source.selected.js_on_change("indices", tap_callback)

word_add = MultiChoice(title="Add a word", options=vocab)
word_add.js_on_change('value', callback_add)
word_remove= MultiChoice(title="Remove a word", options=vocab)
word_remove.js_on_change('value', callback_remove)
remove_all = Button(label='Remove all')
remove_all.js_on_click(callback_rem_all)

# code for word math
equation_data = {
    'positive': [],
    'negative': [],
    'output': [],
    }
equation = ColumnDataSource(data=equation_data)

output_word = Div(height=200)


callback_pos= CustomJS(args=dict(source=equation), code="""
    var new_words = cb_obj.value;
    source.data['positive'] = new_words;
    source.change.emit();
""")
callback_neg = CustomJS(args=dict(source=equation), code="""
    var new_words = cb_obj.value;
    source.data['negative'] = new_words;
    source.change.emit();
""")

calculate_args = dict(
    source = equation,
    vocab = vocab,
    embedded_vocab=embedded_vocab,
    nearest_vocab=nearest_vocab,
    output_word = output_word,
    eq_out_word = eq_out_word
)

button_callback = CustomJS(args=calculate_args, code="""
    var pos_words= source.data['positive'];
    var neg_words= source.data['negative'];

    function cosinesim(A,B) {
        var dotproduct=0;
        var mA=0;
        var mB=0;
        for (var i=0; i<A.length; i++){
            dotproduct += (A[i]*B[i]);
            mA += (A[i]*A[i]);
            mB += (B[i]*B[i]);
        }
        mA = Math.sqrt(mA);
        mB = Math.sqrt(mB);
        var similarity = (dotproduct) / ((mA)*(mB))
        return similarity;
    }
    
    function addArrays(A,B) {
        var result = [...A];
        for(var i=0; i<A.length; i++){
            result[i] = A[i] + B[i];
        }
        return result;
    }

    function subtractArrays(A,B) {
        var result = [...A];
        for(var i=0; i<A.length; i++){
            result[i] = A[i] - B[i];
        }
        return result;
    }

    // make sure words were entered
    if (pos_words.length == 0 && neg_words.length == 0){
        output_word.text = "<h1>No words were entered.  Please try again.</h1>";
        console.log('No words entered');
    } else {
        //get the word embeddings and do the math
        if (pos_words.length > 0){
            var result = subtractArrays(embedded_vocab[vocab.findIndex(word => word === pos_words[0])], embedded_vocab[vocab.findIndex(word => word === pos_words[0])]); 
        } else {
            var result = subtractArrays(embedded_vocab[vocab.findIndex(word => word === neg_words[0])], embedded_vocab[vocab.findIndex(word => word === neg_words[0])]); 
        }

        //add positive words
        for (var i = 0; i<pos_words.length; i++){
            result = addArrays(result, embedded_vocab[vocab.findIndex(word => word === pos_words[i])]);
        }
        
        //subtract negative words
        for (var i = 0; i<neg_words.length; i++){
            result = subtractArrays(result, embedded_vocab[vocab.findIndex(word => word === neg_words[i])]);
        }

        // find the nearest word to the result using cosine similarity
        //https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.most_similar'i
        var output = '';
        var top_word = '';
        var top_sim = -1;
        var top_10_sim = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
        var top_10_words = ['','','','','','','','','',''];

        for (var i=0; i<vocab.length; i++){
            var sim = cosinesim(result, embedded_vocab[i]);
            if (sim > Math.min(...top_10_sim) && !pos_words.includes(vocab[i]) && !neg_words.includes(vocab[i])){
                var idx = top_10_sim.findIndex(x => x === Math.min(...top_10_sim));
                top_10_sim[idx] = sim;
                top_10_words[idx] = vocab[i];
                if (sim > top_sim){
                    top_sim = sim;
                    top_word = vocab[i];
                }
            }
        }

        //add the output to the W2V plot

        //print the output
        if (top_word == ''){
            output_word.text = '<h1> Error </h1>';
        }else {
            var eq_string = '';
            for (var i=0; i<pos_words.length; i++){
                eq_string += pos_words[i];
                if (i<pos_words.length-1){
                    eq_string += ' + ';
                }
            }
            for (var i=0; i<neg_words.length; i++){
                eq_string += ' - ';
                eq_string += neg_words[i];    
            }
            var top_10_string = '';
            for (var i=0; i<top_10_words.length; i++){
                top_10_string += top_10_words[i];
                if (i<top_10_words.length-1){
                    top_10_string += ', ';
                }
            }
            output_word.text = '<h1>' + eq_string + ' = ' + top_word + '</h1> <br> <h3> Top 10 closest words: ' + top_10_string + '</h3>';
            
            eq_out_word.data['output_word'] = [top_word];
            console.log(eq_out_word.data['output_word']);
            eq_out_word.change.emit();
        }
    }
""")

pos_words = MultiChoice(title="Positive Words", options=vocab)
pos_words.js_on_change('value', callback_pos)
pos_words.js_on_change('value', callback_add)
neg_words = MultiChoice(title="Negative Words", options=vocab)
neg_words.js_on_change('value', callback_neg)
neg_words.js_on_change('value', callback_add)
calculate_button = Button(label='Calculate')
calculate_button.js_on_click(button_callback)
calculate_button.js_on_click(callback_add)

layout = column(title_info, sim_words_text, row(column(p_div, word_add, word_remove, remove_all), p, column(word_math, row(pos_words, neg_words), calculate_button, output_word)))
show(layout)

# website_path = r'C:\Users\byahn\Code\MLEF\MLEF-Energy-Storage.github.io'
# html_path = os.path.join(website_path,'W2V_tSNE.html')
# output_file(html_path, title="Word2Vec visualized with tSNE")
save(layout)