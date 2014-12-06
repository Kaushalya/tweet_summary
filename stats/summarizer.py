# -*- coding: utf-8 -*-
'''
Created on Sun Nov 09 22:52

'''
import sklearn
import cPickle
import glob
import pandas as pd
import numpy as np
from text_similarity import get_jaccard_sim
from text_similarity import ner_jaccard
from text_similarity import get_ner_tags

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN

from Feature import BadWordCounter
from tweet_analyser import load_model

#from sklearn.metrics.pairwise.distance_metrics import 

from time import time

data_path = '/home/kaushalya/Code/MCS Project/data/Ebola3/*.csv'

def _is_novel(tweet, selected_tweets):
    for t1 in selected_tweets:
        if(get_jaccard_sim(tweet, t1)>0.75):
            return False
    
    #print tweet
    return True

def _dummy_preprocess(text):
    return text

def remove_duplicates(tweets, is_objective, n):
    selected = np.empty([n], dtype='bool')
    selected.fill(False)   
    
    for i in range(n):
        if(is_objective[i]==True):
            selected[i] = _is_novel(tweets['tweet'].iloc[i], tweets['tweet'][selected==True])
    
    return selected
    
def cluster_tweets(tweets):
    #TODO get TFIDF vector
    #do clustering
    vectorizer = TfidfVectorizer(preprocessor=_dummy_preprocess, tokenizer=get_ner_tags,
                                 binary=True,
                                 min_df=0, use_idf=True, smooth_idf=True)
    tfidf = vectorizer.fit_transform(tweets['tweet']) 
    
    #ner_tags = [get_ner_tags(tweet) for tweet in tweets['tweet']]
    print "clustering started"
    t0 = time()
    #cluster = AgglomerativeClustering(n_clusters=3, affinity="cosine" )
    #cluster = MiniBatchKMeans(n_clusters=10, max_iter=100, batch_size=100)    
    cluster = DBSCAN(min_samples=2, eps=0.3)    
        
    clustered = cluster.fit(tfidf.todense())
       
    #clustered = cluster.fit(ner_tags)
    labels = clustered.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "clustering finished in %.3f seconds"%(time()-t0)   
    print "%d clusters detected"%n_clusters_
    
    return clustered

#TODO select the informative tweets first
def select_tweets(model, tweets):
    #for tw_file in glob.glob(data_path):     
    bdw = BadWordCounter()
    num_tweets = tweets.shape[0]
    print 'read %d tweets'%num_tweets
        
    good_tweets = np.array([(not bdw.has_bad_word(tweet) and not bdw.has_slang(tweet)) for tweet in tweets['tweet']])        
    predicted = model.predict(tweets[good_tweets==True]['tweet'])     
        
    print "num of good tweets: %d"%count_nonzero(good_tweets)        
        
    bad_tweets = np.empty([num_tweets], dtype='bool')
    bad_tweets.fill(False)
        
    bad_tweets[good_tweets==True] = predicted        
    print "predicted good = %d"%count_nonzero(predicted)        
        
    print 'duplication detection running'
    t0 = time()
    selected = remove_duplicates(tweets, bad_tweets, num_tweets)
    t1 = time()-t0
    #print 'duplication detection completed in %.3f'%t1
    #print str(len(selected))
    important_tweets = tweets[selected==True]   
    tweets['predicted'] = selected
    print 'here'
    return important_tweets

if __name__ == '__main__':
    model = load_model('../models/Ebola/linsvc_model_stems_pos.pkl')
    tweets = pd.DataFrame()
    ind = 0
    
    for tw_file in glob.glob('/home/kaushalya/Code/MCS Project/data/Ebola4/tweets_20141111*.csv'):
        print '%d : %s '%(ind,tw_file)
        file_data = pd.read_csv(tw_file, header=0, index_col='status_id')
        tweets = tweets.append(file_data.dropna())
        ind += 1
        
    #tw_file = '/home/kaushalya/Code/MCS Project/data/Ebola4/tweets_20141111-093430.csv'
    
    important = select_tweets(model, tweets)
    print 'Number of objective tweets: '+str(important.shape[0])
    cluster_tweets = cluster_tweets(important)
    #tweets['cluster'] = cluster_tweets.labels_
    important['cluster'] = cluster_tweets.labels_
    important.to_csv('/home/kaushalya/Code/MCS Project/data/output/Ebola/dbs_pos_gate_entity_senna_20141111.csv')
    #tweets.to_csv('/home/kaushalya/Code/MCS Project/data/output/dbscan.csv')