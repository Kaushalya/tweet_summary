# -*- coding: utf-8 -*-
'''
Created on Sun Nov 09 22:52
Labels tweets as ibjective/subjective and clusters them using DBScan algorithm
on NER tags of a tweet.
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
    ner_tags = [get_ner_tags(tweet).tolist() for tweet in tweets['tweet']]
    vectorizer = TfidfVectorizer(preprocessor=_dummy_preprocess, tokenizer=lambda x:x,
                                 binary=True,
                                 min_df=0, use_idf=True, smooth_idf=True)
    tfidf = vectorizer.fit_transform(ner_tags) 
    
    #ner_tags = [get_ner_tags(tweet) for tweet in tweets['tweet']]
    print "clustering started"
    t0 = time()
    #cluster = AgglomerativeClustering(n_clusters=3, affinity="cosine" )
    #cluster = MiniBatchKMeans(n_clusters=10, max_iter=100, batch_size=100) 
    #metric=sklearn.metrics.pairwise.cosine_distances
    cluster = DBSCAN(min_samples=2, eps=0.5)    
        
    clustered = cluster.fit(tfidf.todense())
       
    #clustered = cluster.fit(ner_tags)
    labels = clustered.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "clustering finished in %.3f seconds"%(time()-t0)   
    print "%d clusters detected"%n_clusters_
    
    tweets['cluster'] = labels
    tweets['ner'] = ner_tags
    return tweets

#select the informative tweets first
def select_tweets(model, tweets):
    #for tw_file in glob.glob(data_path):     
    bdw = BadWordCounter()
    num_tweets = tweets.shape[0]
    print 'read %d tweets'%num_tweets
        
    good_tweets = np.array([(not bdw.has_bad_word(tweet) and not bdw.has_slang(tweet)) for tweet in tweets['tweet']])        
    predicted = model.predict(tweets[good_tweets==True]['tweet'])     
        
    bad_tweets = np.empty([num_tweets], dtype='bool')
    bad_tweets.fill(False)
        
    bad_tweets[good_tweets==True] = predicted        
    print "predicted good = %d"%np.count_nonzero(predicted)        
        
    print 'duplication detection running'
    t0 = time()
    selected = remove_duplicates(tweets, bad_tweets, num_tweets)
    t1 = time()-t0
    print 'duplication detection completed in %.3f'%t1
    
    important_tweets = tweets[selected==True] 
    
    return important_tweets

def summarize_text(model, inputfile, outputfile):
    if model==None:
        return "Model loading failed"
          
    tweets = pd.DataFrame()
    ind = 0
    
    for tw_file in glob.glob(inputfile):
        print '%d : %s '%(ind,tw_file)
        file_data = pd.read_csv(tw_file, header=0, index_col='status_id')
        tweets = tweets.append(file_data.dropna())
        ind += 1
        
    important = select_tweets(model, tweets)
    print 'Number of objective tweets: '+str(important.shape[0])
    important_clustered = cluster_tweets(important)

    #important['cluster'] = cluster_tweets.labels_
    important_clustered.to_csv(outputfile)
    #tweets.to_csv('/home/kaushalya/Code/MCS Project/data/output/dbscan.csv')

if __name__ == '__main__':
    model = load_model('/home/kaushalya/Desktop/Defense/Models/Cricket.pkl')
    tweets = pd.DataFrame()
    ind = 0
    
    for tw_file in glob.glob('/home/kaushalya/Code/MCS Project/data/Ebola4/tweets_20141111-093430.csv'):
        print '%d : %s '%(ind,tw_file)
        file_data = pd.read_csv(tw_file, header=0, index_col='status_id')
        tweets = tweets.append(file_data.dropna())
        ind += 1
        
    #tw_file = '/home/kaushalya/Code/MCS Project/data/Ebola4/tweets_20141111-093430.csv'
    
    important = select_tweets(model, tweets)
    print 'Number of objective tweets: '+str(important.shape[0])
    important_clustered = cluster_tweets(important)

    #important['cluster'] = cluster_tweets.labels_
    #important_clustered.to_csv('/home/kaushalya/Code/MCS Project/data/output/SLvENG/SLvENG_summary3.csv')
    #tweets.to_csv('/home/kaushalya/Code/MCS Project/data/output/dbscan.csv')