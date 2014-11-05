# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
import numpy as np
from scipy import sparse
from tweet_tokenizer import tokenize_tweet

import enchant

'''
This class is taken from amueller's kaggle_insluts project.
https://github.com/amueller/kaggle_insults
'''
class BadWordCounter(BaseEstimator):
    '''
    Counts thr number of words not in a dictionary and returns the
    ratio of out-of-words per sentence
    '''
    
    def __init__(self):
        self.dic = enchant.Dict("en_US")
    
    def get_feature_names(self):
        return np.array(['n_words', 'mean_word_len', 'allcaps_ratio', 'bad_ratio'])

    def fit(self, documents, y=None):
        return self
    
    def transform(self, documents):
        words = [tokenize_tweet(c) for c in documents]
        n_words = [len(sent) for sent in words]
        mean_word_len = [np.mean([len(word) for word in sent])
                    for sent in words]
        allcaps = [np.sum([word.isupper() for word in sent])
                    for sent in words]
        n_bad = [np.sum([not self.dic.check(word.lower()) for word in sent])
                    for sent in words]
        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)
        return np.array([n_words, mean_word_len, allcaps_ratio, bad_ratio]).T                
                        

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
    
