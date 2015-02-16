# -*- coding: utf-8 -*-
"""
Created on Fri Oct  17 18:03:50 2014

@author: Kaushalya
"""

import pandas as pd
import random
import numpy as np

from tweet_tokenizer import get_stemmed_tokens
from tweet_tokenizer import get_lemas
import tweet_tokenizer as twt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from Feature import FeatureStacker
from Feature import BadWordCounter

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics

from time import time
import cPickle


def get_tweets(folder):
    tweets = pd.DataFrame()
    for tweet_file in folder:
        tweets = tweets.append(pd.read_csv(tweet_file))
        
    return tweets

def load_model(model_f):
    try:
        with open(model_f, 'rb') as fid:
            model = cPickle.load(fid)
        print 'model loaded successfully'
        return model
        #break
    
    except IOError:
        print 'Error loading model file'        
           
    return None

def save_model(model, path):
    with open(path, 'wb') as fid:
        cPickle.dump(model, fid)        


def print_performance(real, predicted, model=None):
    print 'precision: '+ str(metrics.precision_score(real, predicted))
    print 'recall: ' + str(metrics.recall_score(real, predicted))
    print 'accuracy: ' + str(metrics.accuracy_score(real, predicted))
    print 'F1: '+str(metrics.f1_score(real, predicted))
    
#    if(model is not None):
#        model_params = model.get_params()
#        clf_coef = model_params['clf'].raw_coef_
#        important = np.argsort(np.abs(clf_coef.ravel()))[::-1][:100]
#        ft_params = model_params['vect'].get_params()['words']
#        feature_names = np.array(ft_params.get_feature_names())
#        f_imp = feature_names[important]
#        print f_imp[:20]
    
def build_base_model():
    select = SelectPercentile(score_func=chi2, percentile=16)
    #countvect_char = TfidfVectorizer(ngram_range=(1, 5), analyzer="char", binary=True)
    countvect_word =  TfidfVectorizer(tokenizer=get_stemmed_tokens, analyzer="word",
                                      binary=True, ngram_range=(1,2), use_idf=True, smooth_idf=True, 
                                      stop_words='english')
    countvect_postag = TfidfVectorizer(tokenizer=twt.get_pos_tags, binary=False, 
                                       ngram_range=(3,3), analyzer="word") 
    bad_words = BadWordCounter()
                                   
    #clf = LogisticRegression(tol=1e-6, C=19)
    #clf = MultinomialNB()
    clf = LinearSVC(class_weight='auto', C=1)                                                            
                       
                   
    ft = FeatureStacker([('words', countvect_word),
                         ('bad_words', bad_words),
                         ('pos_tags', countvect_postag)])
                         
    pipeline = Pipeline([('vect', ft), ('select', select), ('clf', clf)])
    return pipeline
    
def run_gridsearch(train_data, test_data):
    clf = build_base_model()
    param_grid = dict(clf__C=np.arange(1, 25, 2))
    cv = ShuffleSplit(len(train_data), n_iter=10, test_size=0.2) 
    grid_search = GridSearchCV(clf, cv=cv, param_grid=param_grid, n_jobs=6)
    grid_search.fit(train_data['tweet'], train_data['formal']) 
    print(grid_search.best_score_)
    print(grid_search.best_params_)                         
                                      
def run_tests(train_data, test_data, model_path=None):  
    Y_train = np.array(train_data['formal'])
    Y_test = np.array(test_data['formal'])
    
    t0 = time()
    print "training started"
    base_model = build_base_model()
    base_model.fit(train_data['tweet'], Y_train)
    train_time = time() - t0
    print "train time: %.3fs" % train_time
    
    predicted = base_model.predict(test_data['tweet'])
    print "*** LogReg base model"
    print_performance(Y_test, predicted, base_model)
    
    if model_path!=None:
        save_model(base_model, model_path)
        
    return predicted

def eval_model(model, X, real):
    predicted = model.predict(X)
    print_performance(real, predicted, model)
    return predicted
    

def _get_sample(data, n_sample, rand_selection=True):
    
    if not rand_selection:
        return data.iloc[:n_sample]
    
    ids = random.sample(xrange(data.shape[0]-1), n_sample)
    return data.iloc[ids]

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
    tweets_class1 = pd.read_csv('../../data/training/SLvNZ/training_verified.csv').dropna()
    tweets_class2 = pd.read_csv('../../data/training/SLvNZ/training_emoji.csv').dropna()
    tweets_class1['formal'] = 1 #Verified tweets
    tweets_class2['formal'] = 0 #tweets with emoticons
    class_size = min([ 3001,len(tweets_class1), len(tweets_class2)])-1
    print "%s instances from each class"%class_size    
    gold_tweets = pd.concat([_get_sample(tweets_class1, class_size, rand_selection=True), 
                             _get_sample(tweets_class2.dropna(),class_size, rand_selection=True)])
    data = get_datasets(gold_tweets)
    #gold_tweets = pd.concat([tweets_class1, tweets_class2])
    
    #gold_tweets = pd.read_csv('../../data/gold standard/ebola_gold.csv')
    #model = load_model('/home/kaushalya/Code/MCS Project/tweet_summarizer/models/Ebola/linsvc_model_stems_senna_pos.pkl')
    #predicted = eval_model(model, gold_tweets['tweet'], gold_tweets['formal'])
    #gold_tweets['predicted'] = predicted
    #gold_tweets.to_csv('../../data/output/Ebola/gold_output.csv', index=False)
    predicted = run_tests(data[0], data[1], model_path="../models/cricket/linsvc_model_stems_pos_senna.pkl")
    #run_gridsearch(data[0], data[1])