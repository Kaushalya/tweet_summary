# -*- coding: utf-8 -*-
'''
This class contains text similarity measures that are used to
cluster similar tweets together.
'''
import tweet_tokenizer as twk
from tweet_tokenizer import clean_tweet
from tweet_tokenizer import get_clean_tokens
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics.distance import jaccard_distance
import nltk

import numpy as np
from nltk.tree import Tree

def _remove_tags(tweet):
    tweet = tweet.replace('MENTION','')
    tweet = tweet.replace('HASH_TAG', '')
    tweet = tweet.replace('RT', '')
    tweet = tweet.replace('URL', '')
    #tweet = tweet.replace('NUMERIC', '')
    return tweet

def extract_entities(text):
     for sent in nltk.sent_tokenize(text):
         for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
             if hasattr(chunk, 'node'):
                 print chunk.node, ' '.join(c[0] for c in chunk.leaves())

def get_ner_tags(text):
    tokens = get_clean_tokens(text)
    tokens2 = get_clean_tokens(text.lower())
    pos_tags = nltk.pos_tag(tokens2)
   
    pos2 = twk.ptagger.tag(tokens)
    #pos2 = [(tokens[i], pos_tags[i][1]) for i in range(len(tokens))]    
    #print pos2
    ne_tags = nltk.ne_chunk(pos2)
    #print pos_tags
    #print pos2
    
    entities = []    
    
    for entity in ne_tags:
        if isinstance(entity, Tree):
            #e_str = entity.label()+ '_' + '_'.join(c[0] for c in entity.leaves())
            #print e_str
            entities.extend(c[0] for c in entity.leaves())         
            #entities.append(e_str)
            #print entity.label(), ' '.join(c[0] for c in entity.leaves())
      
#    if len(entities)==0:
#        return entities.add('nil')   
    return np.unique(entities)

def get_entity_similarity(text1, text2):
    ne1 = get_ner_tags(text1)
    ne2 = get_ner_tags(text2)
    
    return ner_jaccard(ne1, ne2)

def get_jaccard_sim(text1, text2):
#    countvect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), binary=True)
#    countvect.fit([text1, text2])
    t1 = _remove_tags(clean_tweet(text1))
    t2 = _remove_tags(clean_tweet(text2))
    tokens1 = t1.split()
    tokens2 = t2.split()
    
    return 1-jaccard_distance(set(tokens1), set(tokens2))

def ner_jaccard(ne1, ne2):
    if(len(ne1)==0 or len(ne2)==0):
        return 1
        
    return 1-jaccard_distance(set(ne1), set(ne2))
#Removes user names, hashtags and urls before calculating the
#edit distance
def get_edit_distance(text1, text2):
    t1 = _remove_tags(clean_tweet(text1))
    t2 = _remove_tags(clean_tweet(text2))
        
    return nltk.metrics.distance.edit_distance(t1.strip(), t2.strip())

if __name__ == '__main__':
    t1 = "Last US Ebola Patient Is Cured: Dr. Craig Spencer To Be Released… http://t.co/92JfMm2LaN | http://t.co/NoFij4iACl #news"
    t2 = '#Ebola Ebola Outbreak: US Free of Virus After New York Doctor Craig Spencer Cleared - International Business Times UK' 
    
    print nltk.metrics.distance.edit_distance(t1, t2)
    print get_edit_distance(t1, t2)
    print get_ner_tags(t1)
    print get_ner_tags(t2)
    print 'jaccard sim= %f'%get_jaccard_sim(t1, t2)
    print 'entity similarity= %.2f'%get_entity_similarity(t1,t2)