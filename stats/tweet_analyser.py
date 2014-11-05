# -*- coding: utf-8 -*-
"""
Created on Fri Oct  17 18:03:50 2014

@author: Kaushalya
"""

import pandas as pd
import random
import numpy as np
import nltk

from tweet_tokenizer import tokenize_tweet
import tweet_tokenizer as twk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from Feature import FeatureStacker
from Feature import BadWordCounter

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics

from time import time


def get_tweets(folder):
    tweets = pd.DataFrame()
    for tweet_file in folder:
        tweets.append(pd.read_csv(tweet_file))
        
    return tweets

def print_performance(real, predicted):
    print 'precision: '+ str(metrics.precision_score(real, predicted))
    print 'recall: ' + str(metrics.recall_score(real, predicted))
    print 'accuracy: ' + str(metrics.accuracy_score(real, predicted))
    
def build_base_model():
    select = SelectPercentile(score_func=chi2, percentile=16)
    countvect_char = TfidfVectorizer(ngram_range=(1, 5), analyzer="char", binary=True)
    countvect_word =  TfidfVectorizer(tokenizer=tokenize_tweet, analyzer="word", 
                                      binary=True, ngram_range=(1,2), 
                                      stop_words='english')
    #countvect_postag = TfidfVectorizer(tokenizer=twk.get_pos_tags, binary=False, 
    #                                   ngram_range=(1,3), analyzer="word") 
    bad_words = BadWordCounter()
                                   
    clf = LogisticRegression(tol=1e-6, C=6)
    ft = FeatureStacker([('words', countvect_word), 
                        ('bad_words', bad_words)])
    pipeline = Pipeline([('vect', ft), ('select', select), ('logr', clf)])
    return pipeline
    
def run_gridsearch(train_data, test_data):
    clf = build_base_model()
    param_grid = dict(logr__C=np.arange(1, 20, 5))
    cv = ShuffleSplit(len(train_data), n_iterations=10, test_size=0.2) 
    grid_search = GridSearchCV(clf, cv=cv, param_grid=param_grid, n_jobs=6)
    grid_search.fit(train_data['tweet'], train_data['formal']) 
    print(grid_search.best_score_)
    print(grid_search.best_params_)                         
                                      
def run_tests(train_data, test_data):
    #TODO add POS tags as a feature  
    Y_train = np.array(train_data['formal'])
    Y_test = np.array(test_data['formal'])
    
    t0 = time()
#    clf = LogisticRegression()
#    clf.fit(X_train, Y_train)
    base_model = build_base_model()
    base_model.fit(train_data['tweet'], Y_train)
    train_time = time() - t0
    print "train time: %.3fs" % train_time
    
    predicted = base_model.predict(test_data['tweet'])
    print "*** LogReg base model"
    print_performance(Y_test, predicted)
    return predicted
    

def get_datasets(gold_tweets):
    gold_n = len(gold_tweets)
    train_n = int(0.8*gold_n)
    random.seed(123)
    train_ids = random.sample(xrange(gold_n-1), train_n)
    test_ids = list(set(xrange(gold_n)) - set(train_ids))
    
    train_tweets = gold_tweets.iloc[train_ids]
    test_tweets = gold_tweets.iloc[test_ids]
    return [train_tweets, test_tweets]


if __name__ == '__main__':
    tweets_class1 = pd.read_csv('../../data/Ebola2/training/verified_tweets_v2.csv')
    tweets_class2 = pd.read_csv('../../data/Ebola2/training/emoji_tweets_v2.csv')
    tweets_class1['formal'] = 1 #Verified tweets
    tweets_class2['formal'] = 0 #tweets with emoticons
    class_size = 2000
    gold_tweets = pd.concat([tweets_class1[:class_size], tweets_class2[:class_size]])
    data = get_datasets(gold_tweets)
    predicted = run_tests(data[0], data[1])
    #run_gridsearch(data[0], data[1])