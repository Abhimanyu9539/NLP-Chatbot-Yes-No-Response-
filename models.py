## Import required libraries
import nltk
import os
import csv
from nltk.stem.snowball import SnowballStemmer
import random
from nltk.classify import SklearnClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
from preprocessing import preprocess, extract_tagged, extract_feature_from_doc,extract_feature,word_feats

## Get content from the text file which has responses and categories present
def get_content(filename):
    doc = os.path.join(filename)
    with open(doc, 'r') as content_file:
        lines = csv.reader(content_file,delimiter='|')
        data = [x for x in lines if len(x) == 3]
        return data

## Relative path of the file needs to be mentioned    
filename = 'C:/Users/iabhi/Desktop/DS & ML Projects/NLP - Chatbot/NLTK - Chatbot - YesNo Response/responses.txt'
data = get_content(filename)
features_data, corpus, answers = extract_feature_from_doc(data)

## split data into train and test sets
split_ratio = 0.8

## Split the data
def split_dataset(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    train_split = int(data_length * split_ratio)
    return (data[:train_split]), (data[train_split:])
training_data, test_data = split_dataset(features_data, split_ratio)

# save the data
np.save('training_data', training_data)
np.save('test_data', test_data)

## Load the data
training_data = np.load('training_data.npy',allow_pickle=True)
test_data = np.load('test_data.npy',allow_pickle=True)


## Model built using Decision Tree
def train_using_decision_tree(training_data, test_data):
    classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    #print('training set accuracy: ', training_set_accuracy)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    #print('test set accuracy: ', test_set_accuracy)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy
dtclassifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_decision_tree(training_data, test_data)



## Model built using Naive Bayes
def train_using_naive_bayes(training_data, test_data):
    classifier = nltk.NaiveBayesClassifier.train(training_data)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy

nbclassifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_naive_bayes(training_data, test_data)
