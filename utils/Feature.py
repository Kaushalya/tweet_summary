# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
import numpy as np
from scipy import sparse
from tweet_tokenizer import tokenize_tweet
import tweet_tokenizer as twt

import enchant

'''
This class is based on amueller's kaggle_insluts project.
https://github.com/amueller/kaggle_insults
'''
class BadWordCounter(BaseEstimator):
    '''
    Counts thr number of words not in a dictionary and returns the
    ratio of out-of-words per sentence
    '''
    
    def __init__(self):
        with open('../../data/my_badlist.txt', 'rb') as bad_file:
            bad_words = [line.strip().lower() for line in bad_file.readlines()]
            
        with open('../../data/slang.txt', 'rb') as slang_file:
            slang_words = [line.strip() for line in slang_file.readlines()] 
            
        self.bad_words = bad_words
        self.slang_words = slang_words
        
        self.dic = enchant.Dict("en_US")
    
    def get_feature_names(self):
        return np.array(['n_words', 'has_bad', 'has_slang', 'mean_word_len', 'allcaps_ratio', 
        'bad_ratio', 'mention_ratio', 'hashtag_ratio'])

    def fit(self, documents, y=None):
        return self
    
    def has_bad_word(self, tweet):
        for bad in self.bad_words:
            if tweet.lower().split().count(bad)>0:
                return True
        return False
    
    def has_slang(self, tweet):
        for slang in self.slang_words:
            if tweet.split().count(slang)>0:
                return True
        return False    
    
    def transform(self, documents):
        tokens = [tokenize_tweet(c) for c in documents]
        words = [tweet.split() for tweet in documents]
        
        n_words = np.array([len(sent) for sent in words], dtype=np.float)
        mean_word_len = [np.mean([len(word) for word in sent])
                    for sent in words]
        allcaps = [np.sum([word.isupper() for word in sent])
                    for sent in words]
        n_mistakes = [np.sum([not self.dic.check(word.lower()) for word in sent])
                    for sent in tokens]
        n_mentions = [np.sum([word=='MENTION' for word in sent])
                    for sent in tokens] 
        n_hashtags = [len(twt.re_hashtag.findall(sent)) for sent in documents]
#        n_bad = [np.sum([sent.lower().count(bad) for bad in self.bad_words])
#                    for sent in documents] 
        has_bad = [self.has_bad_word(sent) for sent in documents]
        has_slang = [self.has_slang(sent) for sent in documents] 
                    
        allcaps_ratio = np.array(allcaps) / n_words
        mistake_ratio = np.array(n_mistakes) / n_words
        mention_ratio = np.array(n_mentions) / n_words
        hashtag_ratio = np.array(n_hashtags) / n_words
        return np.array([n_words, has_bad, has_slang, mean_word_len, 
                         allcaps_ratio, mistake_ratio, mention_ratio, hashtag_ratio], dtype='float').T                
                        

class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        features = []
        for name, trans in self.transformer_list:
            features.append(trans.transform(X))
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out
            
if __name__ == "__main__":
    documents = ["HELLOOO darling", "Can you hear me"]
    tweet = "#Americanpublic #Ebola #PresidentObama #WestAfricannations Who will pay 2 stop Ebola? http://t.co/3fhP5c530u"
    bdw = BadWordCounter()    
    print bdw.has_slang(tweet)
    
