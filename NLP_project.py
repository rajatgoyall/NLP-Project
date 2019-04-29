import os
import zipfile
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import  RandomForestClassifier
nltk.download('stopwords')



train_data=pd.read_csv('labeledTrainData.tsv',delimiter='\t',quoting=3)
test_data=pd.read_csv('testData.tsv',delimiter='\t',quoting=3)




def clean_text(data):
  cleaned_text=[]
  all_stop=stopwords.words('english')   #list of stopwords
  ss=SnowballStemmer("english")         #list of stemmer
  for i in range(data.shape[0]):
    soup=BeautifulSoup(data[i])            
    html_text=soup.get_text().lower()   #for removing html tags 
    slash=re.sub('[\\\\]',"",html_text) # for removing backward slash
    quotes_text=re.sub('[\"¨]'," <QUOT> ",slash) # for removing double quotes
    com=re.sub('[,]'," <COM> ",quotes_text) # for removing comma
    apos=re.sub("[']"," <APOS> ",com) # for removing apostrophe
    dott=re.sub('\.+'," <DOT> ",apos)  # for removing full stop
    exc=re.sub('!+'," <EXC> ",dott)  # for removing exclamation
    num=re.sub("\d+"," <NUM> ",exc)  # for removing number
    brac=re.sub('[()]+'," <BRAC> ",num)  # for removing brackets
    ques=re.sub("\?+"," <QUES> ",brac)  # for removing question mark
    split_text=ques.split()  # for splitting the text
    stop_text=[i for i in split_text if i not in all_stop]  # for removing stopwords
    stem_text=[ss.stem(i) for i in stop_text]  # for stemming each word
    final_text=[i for i in stem_text if len(i)>=2]  # for removing words with length 1
    cleaned_text.append(" ".join(final_text)) # converting the list of cleaned words into a single sentence
  return np.array(cleaned_text)




cleaned_train_reviews=clean_text(train_data.review)
cleaned_test_reviews=clean_text(test_data.review)



print(cleaned_train_reviews.shape,cleaned_test_reviews.shape)


tf=TfidfVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,ngram_range=(1,3),max_features =10000)
train=tf.fit_transform(cleaned_train_reviews)
test=tf.transform(cleaned_test_reviews)



rc=RandomForestClassifier(150)
rc.fit(train,train_data.sentiment)

test_prediction=rc.predict(test)

output = pd.DataFrame( data={"id":test_data["id"], "sentiment":test_prediction} )

output.shape

output.head()


