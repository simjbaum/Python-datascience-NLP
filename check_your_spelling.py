import nltk
import pandas as pd
import numpy as np
from nltk.corpus import words
from random import sample

def check_your_spelling(inputList):
    """check your spelling in a list of input words. """

    correct_spellings = words.words()
    
    mylist = []
    myrecommend = []
    
    
    for name in inputList:
        for i in (correct_spellings):
            if i.startswith(name[0]):
                word_test = set(nltk.ngrams(i, n=4))
                to_test = set(nltk.ngrams(name, n=4))
                dist = (nltk.jaccard_distance(to_test, word_test))

                # if list is empty get a word in:
                if not mylist:
                    #print("first")
                    mylist.append([i, dist])

                if mylist[0][1] >= dist:
                    #print("now")
                    mylist[0] = [i, dist]

        myrecommend.append(mylist[0][0])
        mylist = []
    
    return myrecommend
    
check_your_spelling(["recommenter"])
