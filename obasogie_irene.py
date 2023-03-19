#!/usr/bin/env python
# coding: utf-8

# In[85]:


from bs4 import BeautifulSoup #the only library capable of reading SGML files 
import csv #for the csv exportation
import re #for regular expressions 
import pandas as pd #to transform the data into tables that are easier to work with for both descriptive and classification tasks
import numpy as np #or the vectorialization step and to deal with everything as it is a matrix


# In[8]:


balises = ["topics", 
           "title"
          ,"body","target"] #the columns of our table 
interesting_topics = ["earn","ship","interest","acq","money-fx"] #the subjects that we are interseted in


# After reading the document semantic exercise, it seems I need to focus on only 5 topics, here called interesting_topics, and from the data we have, only title body and topics seemed relevent for the task (it contains the article)

# In[9]:


#creation of an empty dataset. in that dataset the columns will be the "balises"
dataset = pd.DataFrame(columns = balises)


# This is just to creat an empty dataset of the right size (thats why columns = balises we have only 4 variables)

# In[10]:


#this function will search for the tags mentionned
def format_data(balises,#"topics","title","body","target"
                topic,#YES, NO, BYPASS
                lewissplit):#train, test, not-used
    data = pd.DataFrame(columns = balises)#the columns are the balises
    #full_patent corresponds to all the documents 
    #soup.find_all = returns an object of ResultSet which offers index based access to the result of found occurrences and can be printed using a for loop
    full_patent= soup.find_all("reuters", topics=topic, lewissplit=lewissplit)
    i = 0 #range starts from 0
    for flp in full_patent: # flp loads all the documents that are in the file and flp is one of them while full_patent is all of them 
        l = [] #creation of an empty list
        #flp : everything that it’s included in the tags < REUTERS > </REUTERS>.
        if flp.find("topics").string in interesting_topics: #to find the topics within the interesting_topics
            for name in balises: #if the balises are empty we create an empty cell by adding an none value
                if flp.find(name) is None: 
                    #print(flp.find(name))
                    l.append(None) # we add the non value
                else: #otherwise 
                    l.append(flp.find(name).string) 
            data.loc[i] = l # we specify rows and columns by their integer position values
            i=i+1 #we reassign the variable 
            data["target"]=topic # TOPICS="YES"
    return data # we send the function's results


# Find all the balises that start with reuters, in those select ones that have option topics = "YES" and select lewissplit ="Train" for the train data and lewissplit ="test" for the test data

# In[11]:


def get_data_train(): #we define a new function 
    for topic in ["YES"]:#possible “answers” : YES, NO and BYPASS  #only for the YES targets
        # concatenate/merge two or multiple pandas DataFrames across rows or columns
        #we either add rows or columns. So here by using axis = 0 I say that I already have all the columns and I need new rows
        dataset_train = pd.concat([dataset, format_data(balises,topic,"TRAIN")], axis = 0) # 
    return dataset_train # we send the results


# In[12]:


#same for the function get_data_notused
def get_data_notused():
    for topic in ["YES"]:
        dataset_notused = pd.concat([dataset, format_data(balises,topic,"NOT-USED")], axis = 0)
    return dataset_notused


# In[13]:


#same for the function get_data_test
def get_data_test():
    for topic in ["YES"]:
        dataset_test = pd.concat([dataset, format_data(balises,topic,"TEST")], axis = 0)
    return dataset_test


# In[14]:


#BeautifulSoup to read the SGML files
#encoding transformation with bash for the file 17
common_name_file = "reut2-0" #name of file 
for i in range(22): #we have 22 docs 
    print(i) # the output is 21 bc we start from 0
    if i<10: # conditon. i=22 
        name_file = common_name_file +str(0)+str(i)+".sgm" #bc its 000 - 009
    else:
        name_file = common_name_file +str(i)+".sgm" #010-021
    with open(name_file) as fp: #we open the files
        soup = BeautifulSoup(fp, 'html.parser')# for the tags in html/xml
    if i == 0: #we include all three LEWISSPLIT
        data_train = get_data_train()
        data_test  = get_data_test()
        data_notused = get_data_notused()
    else: #we concatenate and add rows 
        data_train = pd.concat([data_train, get_data_train()], axis = 0)
        data_test = pd.concat([data_test, get_data_test()], axis = 0)
        data_notused = pd.concat([data_notused, get_data_notused()], axis = 0)


# Concat the dataset from the 22 documents according to the selection "train" "test" or "not-used", so instead of having 22 data source mixed between test and train and not used, we will have 3 containing all the train all the test all the not used

# # Creating the tables
# - Since the data is contained in different documents, I used a loop over them and kept only what seems relevent
# - ## file 17 was corrupted I had to manually fix it 

# In[15]:


data_train['train'] = 1
data_test['train'] = 0
data_train = pd.concat([data_train,data_test])


# In[16]:


data_train.to_csv("data_train.csv", index = False)
data_test.to_csv("data_test.csv",index = False)
#two csv files one for each : test and train 
#ndex = false, means that I already have indexes so it won’t add more


# In[17]:


#import CountVectorizer and I remove the punctuation to clean the data
from sklearn.feature_extraction.text import CountVectorizer
import string

def remove_punctuation(text):
    
    if text is not None:
        no_punct=[words for words in text if words not in string.punctuation]
        return ''.join(no_punct) #we replace the ponctuation with an empty string
    else : # join takes all items in an iterable and joins them into one string
        return None


# In[18]:


#apply lambda to lance the expression 

data_train['body_wo_punct'] = data_train['body'].apply(lambda x: remove_punctuation(x))
data_train['title_wo_punct'] = data_train['title'].apply(lambda x: remove_punctuation(x))


# In[19]:


def tokenize(text): # tokenization 
    if text is not None :
        split=re.split("\W+",text.lower())#regex #split only in caracter chains and lowrcase in the same time 
        return split
    else :
        return None
data_train['title_wo_punct_split']=data_train['title_wo_punct'].apply(lambda x: tokenize(x))
data_train['body_wo_punct_split']=data_train['body_wo_punct'].apply(lambda x: tokenize(x))


# In[21]:


import nltk #removal of empty words like articles, adverbs etc
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')


# In[22]:


def remove_stopwords(text): #we create the function for the removal of the stopwords
    if text is not None :
        text=[word for word in text if word not in stopword]
        return text
    else : 
        return None

data_train['title_wo_punct_split_wo_stopwords'] = data_train['title_wo_punct_split'].apply(lambda x: remove_stopwords(x))
data_train['body_wo_punct_split_wo_stopwords']=data_train['body_wo_punct_split'].apply(lambda x: remove_stopwords(x))


# In[23]:


data_train = data_train[data_train["title_wo_punct_split_wo_stopwords"]!= None]
data_train = data_train[data_train["title"]!= None]


# In[24]:


data_train


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


count_vectorize = CountVectorizer()  #instance of a class, vectorization to convert text to numerical data


# # Cleaning the text
# - Classical step in NLP, we get ride of punctuation and non content words (stopwords)
# - In order to use CountVectorizer I had to format the text into long chain of words instead of lists containing words

# In[27]:


#I had lists for the table but the algorithm doesn't accept them, so I need to pass from a list into a sentence
def merge_list(l):
    if l is not None:
        sentence = ""
        for words in l:
            if len(sentence)==0:
                sentence = sentence+words #no space in the first word
            else :
                sentence = sentence +" "+words #space in the second one 
        return sentence
    else :
        return None 


# In[28]:


data_train["title_clean"] = data_train['title_wo_punct_split_wo_stopwords'].apply(lambda x: merge_list(x))
data_train["body_clean"] = data_train['body_wo_punct_split_wo_stopwords'].apply(lambda x: merge_list(x))


# In[29]:


data_train = data_train[data_train["title_clean"]!= None]
data_train = data_train[data_train["body_clean"]!= None]


# In[30]:


data_train.to_csv("data_train.csv", index = False)
data_test.to_csv("data_test.csv",index = False)


# In[31]:


data_train = pd.read_csv("data_train.csv",dtype=str)


# In[32]:


max_features = 1000 #limite


# In[33]:


#take the words and transform them into this matrix and have as output the two merged columns in cleaned version 

def analyze_freq(data, topic, body_or_title):
    data = data[data["topics"] == topic] 
    count_vectorize = CountVectorizer(lowercase = False, max_features = max_features)
    if body_or_title == "body" :
        vectorized_body = count_vectorize.fit_transform(data['body_clean'].apply(lambda x : str(x)))
    else:
        vectorized_body = count_vectorize.fit_transform(data['body_clean'].apply(lambda x : str(x)))   
    X = vectorized_body.toarray()
    return np.sum(X, axis = 0)


# In[34]:


count_vectorize = CountVectorizer(lowercase = False, max_features = max_features)
vectorized_body = count_vectorize.fit_transform(data_train['body_clean'].apply(lambda x : str(x)))
vectorized_title = count_vectorize.fit_transform(data_train['title_clean'].apply(lambda x : str(x)))


# In[35]:


#is creating the matrix
X = vectorized_body.toarray()


# In[36]:


#to verify the size of the matrix
#contains 6923 articles and 1000 words.
np.shape(X)


# # Vect
# The goal of this section is to transform words into numbers, the idea is quite simple, fix a number of words that I'm going to look at (here is 1000), create a vector with 1000 positions, and each time a word is assiocated with a positionI add 1.
# For instance if the sentence was "good morning friend cold this morining" => [1,2,1,1,1] and i do this for all the paragraphs

# In[37]:


#1) descriptive analysis 
#I managed to overcome the error that I had 
#permits to visualize the words with the biggest frequency, the frequency of every word itself
#for body
threhsold_frequence = 20
for topic in interesting_topics:
    output = analyze_freq(data_train, topic, "body")
    most_freq_indices = sorted(range(len(output)), key=lambda i: output[i])[-threhsold_frequence:]
    most_freq = [0]*max_features
    for i in most_freq_indices:
        most_freq[i] = 1
    print(topic)
    print(sorted(output)[-threhsold_frequence:])
    x=np.array(most_freq)
    x=x.reshape(1,len(most_freq))
    print(count_vectorize.inverse_transform(x))


# In[38]:


#for title
threhsold_frequence = 20
for topic in interesting_topics:
    output = analyze_freq(data_train, topic, "title")
    most_freq_indices = sorted(range(len(output)), key=lambda i: output[i])[-threhsold_frequence:]
    most_freq = [0]*max_features
    for i in most_freq_indices:
        most_freq[i] = 1
    print(topic)
    print(sorted(output)[-threhsold_frequence:])
    x=np.array(most_freq)
    x=x.reshape(1,len(most_freq))
    print(count_vectorize.inverse_transform(x))


# In[39]:


#2) predictive analysis
# Encode target labels with value between 0 and n_classes-1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data_train["topics"])


# In[75]:


# size of my data set
X_train = X[:4940]
y_train = y[:4940]
X_test = X[4940:]
y_test = y[4940:]


# In[76]:


#searches which word is associated with each frequency
#1st algo
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# In[77]:


#With this algorithm I measured 5 times (cv = 5). 
#It is expected to have different outputs, but I just needed to have the idea of the performance and keep the best value
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)


# In[78]:


scores


# In[79]:


#1st algo for train set
#the best option is 0.25
#This number is associated with the strongest accuracy
#train set
param_grid = [0.25,0.5,1,1.5]
for alpha in param_grid:
    clf = MultinomialNB(alpha = alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(alpha)
    print(np.mean(scores))


# In[80]:


#used RandomForestClassifier that aggregates a lot of decision trees into one and I tested it again 5 times to have a good view of the accuracy.
#train set 

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth= 15, random_state=1)
scores = cross_val_score(clf, X_train, y_train, cv=5)


# In[81]:


scores


# In[82]:


#for the test set

scores = cross_val_score(clf, X_test, y_test, cv=5)


# In[83]:


#results
np.mean(scores)

