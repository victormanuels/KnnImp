#!/usr/bin/env python
# coding: utf-8

# In[3]:


stopWords=['i',
'me',
'my',
'myself',
'we',
'our',
'ours',
'ourselves',
'you',
'your',
'yours',
'yourself',
'yourselves',
'he',
'him',
'his',
'himself',
'she',
'her',
'hers',
'herself',
'it',
'its',
'itself',
'they',
'them',
'their',
'theirs',
'themselves',
'what',
'which',
'who',
'whom',
'this',
'that',
'these',
'those',
'am',
'is',
'are',
'was',
'were',
'be',
'been',
'being',
'have',
'has',
'had',
'having',
'do',
'does',
'did',
'doing',
'a',
'an',
'the',
'and',
'but',
'if',
'or',
'because',
'as',
'until',
'while',
'of',
'at',
'by',
'for',
'with',
'about',
'against',
'between',
'into',
'through',
'during',
'before',
'after',
'above',
'below',
'to',
'from',
'up',
'down',
'in',
'out',
'on',
'off',
'over',
'under',
'again',
'further',
'then',
'once',
'here',
'there',
'when',
'where',
'why',
'how',
'all',
'any',
'both',
'each',
'few',
'more',
'most',
'other',
'some',
'such',
'no',
'nor',
'not',
'only',
'own',
'same',
'so',
'than',
'too',
'very',
's',
't',
'can',
'will',
'just',
'don',
'should',
'now','-','--','\n'];


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
class textClass:
    def __init__(self,className, listText):
        self.className = className
        self.listText = listText
    def createBagofWords(self):
        self.vectorizer = CountVectorizer(self.listText)
        self.bagMatriz= self.vectorizer.fit_transform( self.listText).todense() 


class distancesClass:
    def __init__(self,distance, elements,className):
        self.distance = distance
        self.elements = elements
        self.className = className
    
     


# In[5]:




from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
def removeStopWords(word):
    word=word.lower()
    word=re.sub('[^A-Za-z0-9- ]+', ' ',word)
    stop_words = stopWords
    
    
    
    wordTokens = word_tokenize(word)
    wordFiltered = [w for w in wordTokens if not w in stop_words]
    return  ' '.join(wordFiltered) 


# In[6]:


import os
import re

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
newsgroups_train = fetch_20newsgroups(subset='train')
categoriesList=newsgroups_train.target_names


print(categoriesList)
print('alt.atheism')

classList = []
#print(categories)
for currentCategory in categoriesList:
        #print(currentCategory)
        currentData = fetch_20newsgroups(subset='train',categories=[currentCategory])
        className=currentCategory
        textList = []
        iii=0
        for currentNew in currentData.data:
            currentNew=removeStopWords(currentNew)  
            textList.append(currentNew)
        objectText= textClass(className, textList)
        objectText.createBagofWords()
        classList.append(objectText)



       


# In[7]:


import math
import numpy as np

distancesList=[]
def computedDistance(bagMatriz,newDataVector,className):
    bagMatrizArray=np.array(bagMatriz)
    lenRowMatrix = len(bagMatrizArray)
    lenColumnMatrix = len(bagMatrizArray[0])   
    for x in range(lenRowMatrix):
            rest= math.sqrt(sum((bagMatriz[x,k] - newDataVector[0,k])**2 for k in range(lenColumnMatrix)))
            objectDistance= distancesClass(rest, bagMatriz[x],className)
            distancesList.append(objectDistance)
            


# In[8]:


wordaa =input("Ingrese el texto a clasificar")


# In[ ]:


worrr=removeStopWords(wordaa)
for i in range(len(classList)):
    newDataVector = classList[i].vectorizer.transform([worrr]).todense()
    computedDistance(classList[i].bagMatriz,newDataVector,classList[i].className)

     
    
sorted_cards = sorted(distancesList, key=lambda x: x.distance)
for i in range(len(sorted_cards)):
    print(sorted_cards[i].distance)
    print(sorted_cards[i].className)

    


# In[ ]:





# In[ ]:




