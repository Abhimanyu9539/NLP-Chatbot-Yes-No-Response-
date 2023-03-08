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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

## Preprocess the sentence fed to the bot
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return filtered_words

## Extract the tagged words from sentence
def extract_tagged(sentences):
    features = []
    for tagged_word in sentences:
        word, tag = tagged_word
        if tag=='NN' or tag == 'VBN' or tag == 'NNS' or tag == 'VBP' or tag == 'RB' or tag == 'VBZ' or tag == 'VBG' or tag =='PRP' or tag == 'JJ':
            features.append(word)
    return features

## Using stemmer and lemmatizer
def extract_feature(text):
    words = preprocess(text)
    tags = nltk.pos_tag(words)
    extracted_features = extract_tagged(tags)
    stemmed_words = [SnowballStemmer("english").stem(x) for x in extracted_features]
    result = [WordNetLemmatizer().lemmatize(x) for x in stemmed_words]
   
    return result


## Bag of words
def word_feats(words):
    return dict([(word, True) for word in words])

## Parsing the whole document or sentences
def extract_feature_from_doc(data):
    result = []
    corpus = []
    # The responses of the chat bot
    answers = {}
    for (text,category,answer) in data:

        features = extract_feature(text)

        corpus.append(features)
        result.append((word_feats(features), category))
        answers[category] = answer

    return (result, sum(corpus,[]), answers)