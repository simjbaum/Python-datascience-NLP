
---

_You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._

---

# Assignment 4 - Document Similarity & Topic Modelling

## Part 1 - Document Similarity

For the first part of this assignment, you will complete the functions `doc_to_synsets` and `similarity_score` which will be used by `document_path_similarity` to find the path similarity between two documents.

The following functions are provided:
* **`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. You will need to use this function in `doc_to_synsets`.
* **`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.

You will need to finish writing the following functions:
* **`doc_to_synsets:`** returns a list of synsets in document. This function should first tokenize and part of speech tag the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it should find each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset match should be used. If there is no match, that token is skipped.
* **`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together and normalize this value by dividing it by the number of largest similarity values found. Be careful with data types, which should be floats. Missing values should be ignored.

Once `doc_to_synsets` and `similarity_score` have been completed, submit to the autograder which will run `test_document_path_similarity` to test that these functions are running correctly. 

*Do not modify the functions `convert_tag`, `document_path_similarity`, and `test_document_path_similarity`.*


```python
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    from nltk.tokenize import word_tokenize

    #tokennize the input
    tokens = word_tokenize(doc)
    
    # add a position tag - like noun or verb 
    pos_tag_tokens = nltk.pos_tag(tokens)
    
    # transform the pos_tag_tokens to the format we need below in wn.synsets argument - list as an output
    pos_tag_tokens_converted = [convert_tag(x[1]) for x in pos_tag_tokens]

    
    # create the synsets for each token and tag. Each token and pos_tag_tokens_converted needs to be input as single
    res = []
    for token, tag in zip(tokens, pos_tag_tokens_converted):
        synset = wn.synsets(token,tag)
        if synset:
            res.append(synset[0])
            
    return res


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    set_values = []
    res = []
    for syn_1 in s1:
        for syn_2 in s2:
            val = syn_1.path_similarity(syn_2)
            if val != None : 
                set_values.append(val) 

        if set_values:
            res.append(max(set_values)) # add the highest values
    
        set_values = [] # reset the list that values wont be recycled

    return sum(res) / len(res)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /home/jovyan/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /home/jovyan/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!


### test_document_path_similarity

Use this function to check if doc_to_synsets and similarity_score are correct.

*This function should return the similarity score as a float.*


```python
def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
```

<br>
___
`paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.

`Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).


```python
# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quality</th>
      <th>D1</th>
      <th>D2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Ms Stewart, the chief executive, was not expec...</td>
      <td>Ms Stewart, 61, its chief executive officer an...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>After more than two years' detention under the...</td>
      <td>After more than two years in detention by the ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>"It still remains to be seen whether the reven...</td>
      <td>"It remains to be seen whether the revenue rec...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>And it's going to be a wild ride," said Allan ...</td>
      <td>Now the rest is just mechanical," said Allan H...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>The cards are issued by Mexico's consulates to...</td>
      <td>The card is issued by Mexico's consulates to i...</td>
    </tr>
  </tbody>
</table>
</div>




```python
maxscore_final = []

for index, row in paraphrases.iterrows():
    
    maxscore_row = []
    
    for index_it1, row_it1 in paraphrases.iterrows():
        if document_path_similarity(row["D1"], row_it1["D2"]):
            if maxscore_row: # check if the list is not empty
                if document_path_similarity(row["D1"], row_it1["D2"]) > maxscore_row[0][2]:
                    maxscore_row = []
                    maxscore_row.append( [row["D1"], row_it1["D2"], document_path_similarity(row["D1"], row_it1["D2"]) ])
            else:
                maxscore_row.append( [row["D1"], row_it1["D2"], document_path_similarity(row["D1"], row_it1["D2"]) ])
    
    
    # if the document_path_similarity score in the new pair is higher in the previous pair, delete previous pair and write new one to it 
    if maxscore_final: # check if the list is not empty
        if maxscore_final[0][2] < maxscore_row[0][2]:
            maxscore_final = []
            maxscore_final.append(maxscore_row[0])
    else: # if the list is still empty write the first entry to it:
        maxscore_final.append(maxscore_row[0])

        
# return maxscore_final[0][0], maxscore_final[0][1], maxscore_final[0][2]          
```


```python
maxscore_final
```


```python

```




    tuple




```python
myl = []
r1 = "Ms Stewart, the chief executive,"
r2 = "After more than two years in detention"
sc = 0.5

r3 = "Ms Stewart, the chief executive,"
r4 = "After more than two years in detention"
sc1 = 0.2

r5 = "Ms Stewart, the chief executive,"
r6 = "After more than two years in detention"
sc2 = 0.8
```


```python
myl.append([r1,r2,sc])
#myl.append([r3,r4,sc1])
#myl.append([r5,r6,sc2])
```


```python
myl = list([r1,r2,sc])
```


```python
myl[0][2]
```




    0.5



___

### most_similar_docs

Using `document_path_similarity`, find the pair of documents in paraphrases which has the maximum similarity score.

*This function should return a tuple `(D1, D2, similarity_score)`*


```python
def most_similar_docs():
    maxscore_final = []

    for index, row in paraphrases.iterrows():

        maxscore_row = []

        for index_it1, row_it1 in paraphrases.iterrows():
            if document_path_similarity(row["D1"], row_it1["D2"]):
                if maxscore_row: # check if the list is not empty
                    if document_path_similarity(row["D1"], row_it1["D2"]) > maxscore_row[0][2]:
                        maxscore_row = []
                        maxscore_row.append( [row["D1"], row_it1["D2"], document_path_similarity(row["D1"], row_it1["D2"]) ])
                else:
                    maxscore_row.append( [row["D1"], row_it1["D2"], document_path_similarity(row["D1"], row_it1["D2"]) ])


        # if the document_path_similarity score in the new pair is higher in the previous pair, delete previous pair and write new one to it 
        if maxscore_final: # check if the list is not empty
            if maxscore_final[0][2] < maxscore_row[0][2]:
                maxscore_final = []
                maxscore_final.append(maxscore_row[0])
        else: # if the list is still empty write the first entry to it:
            maxscore_final.append(maxscore_row[0])


    return maxscore_final[0][0], maxscore_final[0][1], maxscore_final[0][2]          
```


```python
#most_similar_docs()
```




    ('"Indeed, Iran should be put on notice that efforts to try to remake Iraq in their image will be aggressively put down," he said.',
     '"Iran should be on notice that attempts to remake Iraq in Iran\'s image will be aggressively put down," he said.\n',
     0.9753086419753086)



### label_accuracy

Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.

*This function should return a float.*


```python
def label_accuracy():
    from sklearn.metrics import accuracy_score
    maxscore_final = []
    
    for index, row in paraphrases.iterrows():

        maxscore_row = []

        for index_it1, row_it1 in paraphrases.iterrows():
            if document_path_similarity(row["D1"], row_it1["D2"]):
                if maxscore_row: # check if the list is not empty
                    if document_path_similarity(row["D1"], row_it1["D2"]) > maxscore_row[0][2]:
                        maxscore_row = []
                        maxscore_row.append( [row["D1"], row_it1["D2"], document_path_similarity(row["D1"], row_it1["D2"]) ])
                else:
                    maxscore_row.append( [row["D1"], row_it1["D2"], document_path_similarity(row["D1"], row_it1["D2"]) ])


        maxscore_final.append(maxscore_row[0][2])
    
    res = []
    for val in maxscore_final:
        if val > 0.75:
            res.append(1)
        else:
            res.append(0)

    paraphrases["score"] = res


    return accuracy_score(paraphrases["Quality"], paraphrases["score"])
```


```python
label_accuracy()
```




    0.80000000000000004



## Part 2 - Topic Modelling

For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`. You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters on the corpus, and save to the variable `ldamodel`. Extract 10 topics using `corpus` and `id_map`, and with `passes=25` and `random_state=34`.


```python
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

```

### lda_topics

Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as a list of 10 tuples where each tuple takes on the form:

`(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')`

for example.

*This function should return a list of tuples.*


```python
def lda_topics():
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics =10, id2word = id_map)
    
    return ldamodel.print_topics()
```

### topic_distribution

For the new document `new_doc`, find the topic distribution. Remember to use vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.

*This function should return a list of tuples, where each tuple is `(#topic, probability)`*


```python
new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]
```


```python
def topic_distribution():
    
    import gensim

    # Fit and transform
    X = vect.transform(new_doc)

    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)


    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics =10, id2word = id_map)

    return list(ldamodel.get_document_topics(corpus))[0]
```


```python
topic_distribution()
```

    /opt/conda/lib/python3.6/site-packages/gensim/models/ldamodel.py:527: RuntimeWarning: overflow encountered in exp2
      (perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words))





    [(0, 0.02000000031650671),
     (1, 0.020000000317134746),
     (2, 0.020000000317399211),
     (3, 0.020000000317934526),
     (4, 0.020000000317814209),
     (5, 0.020000000317023432),
     (6, 0.020000000316539215),
     (7, 0.81999999714454597),
     (8, 0.020000000317949792),
     (9, 0.020000000317152038)]



### topic_names

From the list of the following given topics, assign topic names to the topics you found. If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.

Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, Society & Lifestyle, Religion, Education.

*This function should return a list of 10 strings.*


```python
def topic_names():
    
    # Your Code Here
    
    return # Your Answer Here
```
