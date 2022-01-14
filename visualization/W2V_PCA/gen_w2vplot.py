from re import I
from typing import Text
from gensim.models import Word2Vec
import os

remove_punctuation = True 



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

output_file("output/W2V_PCA.html", title="Word2Vec visualized with PCA")

title_info = Div(text="""<h1>Word Embedding Visualization </h1> <br> 
<h2> Words from the abstracts were embedded to 200 dimensional vectors using the word2vec algorithm
These plots represent the proximity of these word vectors projected to 2D space using PCA.
Similar words are closer together.<br><br> </h2>""")

#Choose model to visualize

mod = Word2Vec.load(r'C:\Users\aspit\Git\NLP\SciLitNLP\modeling\word2vec\models\word2vec_semantic.model')

#Generate a vector representation (X) of a collection of words
vocab = list(mod.wv.key_to_index.keys())
# clean up vocab by removing words that are just punctuation (takes off ~400KB)
if remove_punctuation:
    punctuation = ["''", "'s", '(ii)', ').', ',', '--', '----_----', '..', '...', '1', '1.', '10', '100', '11', '12', '13', '14',
     '16', '18', '1c', '1d', '2', '2+', '2-d', '2.', '20', '22', '2d', '3', '3-d', '30', '3d', '3d_print', '4', '5', '6', '7', '8',
     '9', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '©', '©_asm', '®', '°', '±', '»', '–', '—', '‘', '’', '†', '•', 
     '…', '€', '℃', '→', '−', '−_cm-1', '∼', '≈', '\uf0b7', 'ii', 'iii', 'iv', 'vii', 'titl', 'appendix', 'page', 'fg-1', 'figur',
     '–: −', 'a.', 'b.', 'c.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'k.', 'l.', 'm.', 'o.', 'p.', 'q.', 'r.', 's.', 't.', 'u.', 'v.',
     'w.', 'x.', 'y.', 'z.']
    for p in punctuation:
        if p in vocab:
            vocab.remove(p)
embedded_vocab = mod.wv[vocab]
nearest_vocab = []
#adding the nearest vocab pushes the file size to 25,685KB which is 686KB too large
for word in vocab:
    s=''
    nearest_vocab.append(s.join((t[0] + ', ') for t in mod.wv.most_similar(word)))

p_div = Div(text="""<h3> You can add and remove words to the plot and see a new projection based on selected words </h3>""")
word_math = Div(text="""<h3> You can do math with the word vectors by selecting positive and negative words.  
The result will be the word vector with the closest cosine similarity to the sum of all positive and negative word vectors.<br>
 (Note, cosine similarity does not consider magnitude and is calculated in the original 200 dim space, not the 2D space displayed here.)</h3>""")
#display the similar words when a point is selected
sim_words_text = Div(text="""<h2>Hover over or select a word to see the most similar words in the vocabulary.</h2>""")

top_words = mod.top_words #added this custom attribute in gen model script 

X= mod.wv[top_words]
nearest_words = []
for word in top_words:
	s=''
	nearest_words.append(s.join((t[0] + ', ') for t in mod.wv.most_similar(word)))

pca= PCA(n_components=2)
result= pca.fit_transform(X)

plot_source = ColumnDataSource(data=dict(
    pc1=result[:,0].tolist(),
    pc2=result[:,1].tolist(),
    plot_words=top_words,
    similar_words=nearest_words,
))

eq_out_word = ColumnDataSource(data={'output_word': ['']}) 

callback_args = dict(
    source=plot_source,
    X = X,
    vocab=vocab,
    embedded_vocab=embedded_vocab,
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
        
        //cite https://github.com/bitanath/pca#:~:text=README.md-,Principal%20Components%20Analysis%20in%20Javascript!,API%20and%20some%20ideas%20%F0%9F%92%A1%20.
        var PCA=function(){function r(r){return i(r,c(u(l(r.length),r),1/r.length))}function n(r){return u(h(r),r)}function t(r,n){return c(r,n?1/(r.length-1):1/r.length)}function e(r){var n=function(r){var n,t=Math.pow(2,-52),e=1e-64/t,o=0,a=0,f=0,u=0,i=0,c=s(r),l=c.length,h=c[0].length;if(l<h)throw"Need more rows than columns";var g=new Array(h),v=new Array(h);for(a=0;a<h;a++)g[a]=v[a]=0;var m=function r(n,t,e){void 0===e&&(e=0);var o,a=n[e],f=Array(a);if(e===n.length-1){for(o=a-2;o>=0;o-=2)f[o+1]=t,f[o]=t;return-1===o&&(f[0]=t),f}for(o=a-1;o>=0;o--)f[o]=r(n,t,e+1);return f}([h,h],0);function d(r,n){return r=Math.abs(r),n=Math.abs(n),r>n?r*Math.sqrt(1+n*n/r/r):0==n?r:n*Math.sqrt(1+r*r/n/n)}var p=0,M=0,w=0,b=0,A=0,D=0,y=0;for(a=0;a<h;a++){for(g[a]=M,y=0,i=a+1,f=a;f<l;f++)y+=c[f][a]*c[f][a];if(y<=e)M=0;else for(p=c[a][a],M=Math.sqrt(y),p>=0&&(M=-M),w=p*M-y,c[a][a]=p-M,f=i;f<h;f++){for(y=0,u=a;u<l;u++)y+=c[u][a]*c[u][f];for(p=y/w,u=a;u<l;u++)c[u][f]+=p*c[u][a]}for(v[a]=M,y=0,f=i;f<h;f++)y+=c[a][f]*c[a][f];if(y<=e)M=0;else{for(p=c[a][a+1],M=Math.sqrt(y),p>=0&&(M=-M),w=p*M-y,c[a][a+1]=p-M,f=i;f<h;f++)g[f]=c[a][f]/w;for(f=i;f<l;f++){for(y=0,u=i;u<h;u++)y+=c[f][u]*c[a][u];for(u=i;u<h;u++)c[f][u]+=y*g[u]}}(A=Math.abs(v[a])+Math.abs(g[a]))>b&&(b=A)}for(a=h-1;-1!=a;a+=-1){if(0!=M){for(w=M*c[a][a+1],f=i;f<h;f++)m[f][a]=c[a][f]/w;for(f=i;f<h;f++){for(y=0,u=i;u<h;u++)y+=c[a][u]*m[u][f];for(u=i;u<h;u++)m[u][f]+=y*m[u][a]}}for(f=i;f<h;f++)m[a][f]=0,m[f][a]=0;m[a][a]=1,M=g[a],i=a}for(a=h-1;-1!=a;a+=-1){for(i=a+1,M=v[a],f=i;f<h;f++)c[a][f]=0;if(0!=M){for(w=c[a][a]*M,f=i;f<h;f++){for(y=0,u=i;u<l;u++)y+=c[u][a]*c[u][f];for(p=y/w,u=a;u<l;u++)c[u][f]+=p*c[u][a]}for(f=a;f<l;f++)c[f][a]=c[f][a]/M}else for(f=a;f<l;f++)c[f][a]=0;c[a][a]+=1}for(t*=b,u=h-1;-1!=u;u+=-1)for(var S=0;S<50;S++){var E=!1;for(i=u;-1!=i;i+=-1){if(Math.abs(g[i])<=t){E=!0;break}if(Math.abs(v[i-1])<=t)break}if(!E){o=0,y=1;var V=i-1;for(a=i;a<u+1&&(p=y*g[a],g[a]=o*g[a],!(Math.abs(p)<=t));a++)for(M=v[a],w=d(p,M),v[a]=w,o=M/w,y=-p/w,f=0;f<l;f++)A=c[f][V],D=c[f][a],c[f][V]=A*o+D*y,c[f][a]=-A*y+D*o}if(D=v[u],i==u){if(D<0)for(v[u]=-D,f=0;f<h;f++)m[f][u]=-m[f][u];break}if(S>=49)throw"Error: no convergence.";for(b=v[i],A=v[u-1],M=g[u-1],w=g[u],M=d(p=((A-D)*(A+D)+(M-w)*(M+w))/(2*w*A),1),p=p<0?((b-D)*(b+D)+w*(A/(p-M)-w))/b:((b-D)*(b+D)+w*(A/(p+M)-w))/b,o=1,y=1,a=i+1;a<u+1;a++){for(M=g[a],A=v[a],w=y*M,M*=o,D=d(p,w),g[a-1]=D,p=b*(o=p/D)+M*(y=w/D),M=-b*y+M*o,w=A*y,A*=o,f=0;f<h;f++)b=m[f][a-1],D=m[f][a],m[f][a-1]=b*o+D*y,m[f][a]=-b*y+D*o;for(D=d(p,w),v[a-1]=D,p=(o=p/D)*M+(y=w/D)*A,b=-y*M+o*A,f=0;f<l;f++)A=c[f][a-1],D=c[f][a],c[f][a-1]=A*o+D*y,c[f][a]=-A*y+D*o}g[i]=0,g[u]=p,v[u]=b}for(a=0;a<v.length;a++)v[a]<t&&(v[a]=0);for(a=0;a<h;a++)for(f=a-1;f>=0;f--)if(v[f]<v[a]){for(o=v[f],v[f]=v[a],v[a]=o,u=0;u<c.length;u++)n=c[u][a],c[u][a]=c[u][f],c[u][f]=n;for(u=0;u<m.length;u++)n=m[u][a],m[u][a]=m[u][f],m[u][f]=n;a=f}return{U:c,S:v,V:m}}(r);console.log(n);var t=n.U;return n.S.map(function(r,n){var e={};return e.eigenvalue=r,e.vector=t.map(function(r,t){return-1*r[n]}),e})}function o(n,...t){var e=t.map(function(r){return r.vector}),o=u(e,h(r(n))),a=c(u(l(n.length),n),-1/n.length);return{adjustedData:o,formattedAdjustedData:f(o,2),avgData:a,selectedVectors:e}}function a(o){return e(t(n(r(o)),!1))}function f(r,n){var t=Math.pow(10,n||2);return r.map(function(r,n){return r.map(function(r){return Math.round(r*t)/t})})}function u(r,n){if(!(r[0]&&n[0]&&r.length&&n.length))throw new Error("Both A and B should be matrices");if(n.length!==r[0].length)throw new Error("Columns in A should be the same as the number of rows in B");for(var t=[],e=0;e<r.length;e++){t[e]=[];for(var o=0;o<n[0].length;o++)for(var a=0;a<r[0].length;a++)t[e][o]=t[e][o]?t[e][o]+r[e][a]*n[a][o]:r[e][a]*n[a][o]}return t}function i(r,n){if(r.length!==n.length||r[0].length!==n[0].length)throw new Error("Both A and B should have the same dimensions");for(var t=[],e=0;e<r.length;e++){t[e]=[];for(var o=0;o<n[0].length;o++)t[e][o]=r[e][o]-n[e][o]}return t}function c(r,n){for(var t=[],e=0;e<r.length;e++){t[e]=[];for(var o=0;o<r[0].length;o++)t[e][o]=r[e][o]*n}return t}function l(r){for(var n=[],t=0;t<r;t++){n[t]=[];for(var e=0;e<r;e++)n[t][e]=1}return n}function h(r){return s(r)[0].map(function(n,t){return r.map(function(r){return r[t]})})}function s(r){var n=JSON.stringify(r);return JSON.parse(n)}return{computeDeviationScores:n,computeDeviationMatrix:r,computeSVD:e,computePercentageExplained:function(r,...n){var t=r.map(function(r){return r.eigenvalue}).reduce(function(r,n){return r+n});return n.map(function(r){return r.eigenvalue}).reduce(function(r,n){return r+n})/t},computeOriginalData:function(r,n,t){var e=i(h(u(h(n),r)),t);return{originalData:e,formattedOriginalData:f(e,2)}},computeVarianceCovariance:t,computeAdjustedData:o,getEigenVectors:a,analyseTopResult:function(r){var n=a(r).sort(function(r,n){return n.eigenvalue-r.eigenvalue});return console.log("Sorted Vectors",n),o(r,n[0].vector)},transpose:h,multiply:u,clone:s,scale:c}}();"undefined"!=typeof module&&(module.exports=PCA);
        
        function find_word(word) {
            return word == new_word;
        }

        for (var i=0; i<words_list.length; i++) {
            new_word = words_list[i];
            if (data['plot_words'].includes(new_word)) {
                p_div.text = '<h3>' + new_word + ' has already been plotted.</h3>';
                p_div.change.emit();
            } else {
                console.log(new_word);
                
                //add to plot_words and regenerate plot?
                var word_index = vocab.findIndex(find_word);
                X.push(embedded_vocab[word_index]);

                var eigen_vectors = PCA.getEigenVectors(X);
                var result = PCA.computeAdjustedData(X, eigen_vectors[0], eigen_vectors[1]).adjustedData;
                for (var i =0; i<data['pc1'].length; i++){
                    data['pc1'][i] = result[0][i];
                    data['pc2'][i] = result[1][i];
                }
                data['pc1'].push(result[0][result[0].length-1]);
                data['pc2'].push(result[1][result[1].length-1]);
                data['plot_words'].push(new_word);
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
        
        //cite https://github.com/bitanath/pca#:~:text=README.md-,Principal%20Components%20Analysis%20in%20Javascript!,API%20and%20some%20ideas%20%F0%9F%92%A1%20.
        var PCA=function(){function r(r){return i(r,c(u(l(r.length),r),1/r.length))}function n(r){return u(h(r),r)}function t(r,n){return c(r,n?1/(r.length-1):1/r.length)}function e(r){var n=function(r){var n,t=Math.pow(2,-52),e=1e-64/t,o=0,a=0,f=0,u=0,i=0,c=s(r),l=c.length,h=c[0].length;if(l<h)throw"Need more rows than columns";var g=new Array(h),v=new Array(h);for(a=0;a<h;a++)g[a]=v[a]=0;var m=function r(n,t,e){void 0===e&&(e=0);var o,a=n[e],f=Array(a);if(e===n.length-1){for(o=a-2;o>=0;o-=2)f[o+1]=t,f[o]=t;return-1===o&&(f[0]=t),f}for(o=a-1;o>=0;o--)f[o]=r(n,t,e+1);return f}([h,h],0);function d(r,n){return r=Math.abs(r),n=Math.abs(n),r>n?r*Math.sqrt(1+n*n/r/r):0==n?r:n*Math.sqrt(1+r*r/n/n)}var p=0,M=0,w=0,b=0,A=0,D=0,y=0;for(a=0;a<h;a++){for(g[a]=M,y=0,i=a+1,f=a;f<l;f++)y+=c[f][a]*c[f][a];if(y<=e)M=0;else for(p=c[a][a],M=Math.sqrt(y),p>=0&&(M=-M),w=p*M-y,c[a][a]=p-M,f=i;f<h;f++){for(y=0,u=a;u<l;u++)y+=c[u][a]*c[u][f];for(p=y/w,u=a;u<l;u++)c[u][f]+=p*c[u][a]}for(v[a]=M,y=0,f=i;f<h;f++)y+=c[a][f]*c[a][f];if(y<=e)M=0;else{for(p=c[a][a+1],M=Math.sqrt(y),p>=0&&(M=-M),w=p*M-y,c[a][a+1]=p-M,f=i;f<h;f++)g[f]=c[a][f]/w;for(f=i;f<l;f++){for(y=0,u=i;u<h;u++)y+=c[f][u]*c[a][u];for(u=i;u<h;u++)c[f][u]+=y*g[u]}}(A=Math.abs(v[a])+Math.abs(g[a]))>b&&(b=A)}for(a=h-1;-1!=a;a+=-1){if(0!=M){for(w=M*c[a][a+1],f=i;f<h;f++)m[f][a]=c[a][f]/w;for(f=i;f<h;f++){for(y=0,u=i;u<h;u++)y+=c[a][u]*m[u][f];for(u=i;u<h;u++)m[u][f]+=y*m[u][a]}}for(f=i;f<h;f++)m[a][f]=0,m[f][a]=0;m[a][a]=1,M=g[a],i=a}for(a=h-1;-1!=a;a+=-1){for(i=a+1,M=v[a],f=i;f<h;f++)c[a][f]=0;if(0!=M){for(w=c[a][a]*M,f=i;f<h;f++){for(y=0,u=i;u<l;u++)y+=c[u][a]*c[u][f];for(p=y/w,u=a;u<l;u++)c[u][f]+=p*c[u][a]}for(f=a;f<l;f++)c[f][a]=c[f][a]/M}else for(f=a;f<l;f++)c[f][a]=0;c[a][a]+=1}for(t*=b,u=h-1;-1!=u;u+=-1)for(var S=0;S<50;S++){var E=!1;for(i=u;-1!=i;i+=-1){if(Math.abs(g[i])<=t){E=!0;break}if(Math.abs(v[i-1])<=t)break}if(!E){o=0,y=1;var V=i-1;for(a=i;a<u+1&&(p=y*g[a],g[a]=o*g[a],!(Math.abs(p)<=t));a++)for(M=v[a],w=d(p,M),v[a]=w,o=M/w,y=-p/w,f=0;f<l;f++)A=c[f][V],D=c[f][a],c[f][V]=A*o+D*y,c[f][a]=-A*y+D*o}if(D=v[u],i==u){if(D<0)for(v[u]=-D,f=0;f<h;f++)m[f][u]=-m[f][u];break}if(S>=49)throw"Error: no convergence.";for(b=v[i],A=v[u-1],M=g[u-1],w=g[u],M=d(p=((A-D)*(A+D)+(M-w)*(M+w))/(2*w*A),1),p=p<0?((b-D)*(b+D)+w*(A/(p-M)-w))/b:((b-D)*(b+D)+w*(A/(p+M)-w))/b,o=1,y=1,a=i+1;a<u+1;a++){for(M=g[a],A=v[a],w=y*M,M*=o,D=d(p,w),g[a-1]=D,p=b*(o=p/D)+M*(y=w/D),M=-b*y+M*o,w=A*y,A*=o,f=0;f<h;f++)b=m[f][a-1],D=m[f][a],m[f][a-1]=b*o+D*y,m[f][a]=-b*y+D*o;for(D=d(p,w),v[a-1]=D,p=(o=p/D)*M+(y=w/D)*A,b=-y*M+o*A,f=0;f<l;f++)A=c[f][a-1],D=c[f][a],c[f][a-1]=A*o+D*y,c[f][a]=-A*y+D*o}g[i]=0,g[u]=p,v[u]=b}for(a=0;a<v.length;a++)v[a]<t&&(v[a]=0);for(a=0;a<h;a++)for(f=a-1;f>=0;f--)if(v[f]<v[a]){for(o=v[f],v[f]=v[a],v[a]=o,u=0;u<c.length;u++)n=c[u][a],c[u][a]=c[u][f],c[u][f]=n;for(u=0;u<m.length;u++)n=m[u][a],m[u][a]=m[u][f],m[u][f]=n;a=f}return{U:c,S:v,V:m}}(r);console.log(n);var t=n.U;return n.S.map(function(r,n){var e={};return e.eigenvalue=r,e.vector=t.map(function(r,t){return-1*r[n]}),e})}function o(n,...t){var e=t.map(function(r){return r.vector}),o=u(e,h(r(n))),a=c(u(l(n.length),n),-1/n.length);return{adjustedData:o,formattedAdjustedData:f(o,2),avgData:a,selectedVectors:e}}function a(o){return e(t(n(r(o)),!1))}function f(r,n){var t=Math.pow(10,n||2);return r.map(function(r,n){return r.map(function(r){return Math.round(r*t)/t})})}function u(r,n){if(!(r[0]&&n[0]&&r.length&&n.length))throw new Error("Both A and B should be matrices");if(n.length!==r[0].length)throw new Error("Columns in A should be the same as the number of rows in B");for(var t=[],e=0;e<r.length;e++){t[e]=[];for(var o=0;o<n[0].length;o++)for(var a=0;a<r[0].length;a++)t[e][o]=t[e][o]?t[e][o]+r[e][a]*n[a][o]:r[e][a]*n[a][o]}return t}function i(r,n){if(r.length!==n.length||r[0].length!==n[0].length)throw new Error("Both A and B should have the same dimensions");for(var t=[],e=0;e<r.length;e++){t[e]=[];for(var o=0;o<n[0].length;o++)t[e][o]=r[e][o]-n[e][o]}return t}function c(r,n){for(var t=[],e=0;e<r.length;e++){t[e]=[];for(var o=0;o<r[0].length;o++)t[e][o]=r[e][o]*n}return t}function l(r){for(var n=[],t=0;t<r;t++){n[t]=[];for(var e=0;e<r;e++)n[t][e]=1}return n}function h(r){return s(r)[0].map(function(n,t){return r.map(function(r){return r[t]})})}function s(r){var n=JSON.stringify(r);return JSON.parse(n)}return{computeDeviationScores:n,computeDeviationMatrix:r,computeSVD:e,computePercentageExplained:function(r,...n){var t=r.map(function(r){return r.eigenvalue}).reduce(function(r,n){return r+n});return n.map(function(r){return r.eigenvalue}).reduce(function(r,n){return r+n})/t},computeOriginalData:function(r,n,t){var e=i(h(u(h(n),r)),t);return{originalData:e,formattedOriginalData:f(e,2)}},computeVarianceCovariance:t,computeAdjustedData:o,getEigenVectors:a,analyseTopResult:function(r){var n=a(r).sort(function(r,n){return n.eigenvalue-r.eigenvalue});return console.log("Sorted Vectors",n),o(r,n[0].vector)},transpose:h,multiply:u,clone:s,scale:c}}();"undefined"!=typeof module&&(module.exports=PCA);
        
        function find_word(word) {
            return word == new_word;
        }

        for (var i=0; i<words_list.length; i++){
            new_word = words_list[i];
            if (data['plot_words'].includes(new_word)) {
                
                var word_index = data['plot_words'].findIndex(find_word);
                X.splice(word_index, 1);
                data['plot_words'].splice(word_index, 1);
                data['similar_words'].splice(word_index, 1);
                
                data['pc1'].pop();
                data['pc2'].pop();

                var eigen_vectors = PCA.getEigenVectors(X);
                var result = PCA.computeAdjustedData(X, eigen_vectors[0], eigen_vectors[1]).adjustedData;
                for (var i =0; i<data['pc1'].length; i++){
                    data['pc1'][i] = result[0][i];
                    data['pc2'][i] = result[1][i];
                }

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
    var num_words = data['pc1'].length;
    for (var i=0; i<num_words; i++){
        data['pc1'].pop();
        data['pc2'].pop();
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
    if (source.selected.indices.length == 0){
        sim_words_text.text = '<h2>Hover over or select a word to see the most similar words in the vocabulary.</h2>';
    } else {
        var ind = source.selected.indices[0];
        var word = source.data['plot_words'][ind];
        var similar = source.data['similar_words'][ind];
        sim_words_text.text = '<h2> Similar words to ' + word + ': ' + similar + '</h2>';
    }
""")

# create a scatter plot of the projection
p = figure(title='Word Vectors Projected to 2D', plot_width=900, plot_height=900) #, x_range=Range1d(-35,35))
p.scatter(source=plot_source, x="pc1", y="pc2", size=10, color="red", alpha=0.8)
p.xaxis[0].axis_label = 'First Principle Component'
p.yaxis[0].axis_label = 'Second Principle Component'
p.x_range = Range1d(-20,20)
p.y_range = Range1d(-20,20)

labels = LabelSet(x='pc1', y='pc2', text='plot_words', x_offset=5, y_offset=5, source=plot_source, render_mode='canvas')
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
        var top_5_sim = [-1,-1,-1,-1,-1];
        var top_5_words = ['','','','',''];

        for (var i=0; i<vocab.length; i++){
            var sim = cosinesim(result, embedded_vocab[i]);
            if (sim > Math.min(...top_5_sim) && !pos_words.includes(vocab[i]) && !neg_words.includes(vocab[i])){
                var idx = top_5_sim.findIndex(x => x === Math.min(...top_5_sim));
                top_5_sim[idx] = sim;
                top_5_words[idx] = vocab[i];
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
            var top_5_string = '';
            for (var i=0; i<top_5_words.length; i++){
                top_5_string += top_5_words[i];
                if (i<top_5_words.length-1){
                    top_5_string += ', ';
                }
            }
            output_word.text = '<h1>' + eq_string + ' = ' + top_word + '</h1> <br> <h3> Top 5 closest words: ' + top_5_string + '</h3>';
            
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
#save(layout)