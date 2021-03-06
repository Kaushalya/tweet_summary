# -*- coding: utf-8 -*-
'''
created on Fri 17 Oct 2014
@author kaushlaya

tokenizes a set of tweets.

'''
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.corpus.reader.wordnet as wordnet
from nltk.tag.stanford import POSTagger
from nltk.tag.stanford import NERTagger
import os
import re
import twokenize as twk
import nltk.tag.senna as senna

java_path = "/usr/local/jdk1.6.0_45/bin/java"
os.environ['JAVAHOME'] = java_path

stn_tagger = POSTagger('/home/kaushalya/Code/MCS Project/stanford_postagger/models/gate-EN-twitter.model',
                   '/home/kaushalya/Code/MCS Project/stanford_postagger/stanford-postagger.jar',
                   encoding='utf-8')
ptagger = senna.POSTagger('/home/kaushalya/Code/MCS Project/tools/senna')

emoji_ptn_str = u'[\U0001F601-\U0001F64F]'
emoticon_ptn_str = ':-*[\\)\\(|/@)pPD]'
#Removes emoticons and emoji from the training dataset
re_emotion = re.compile( '|'.join([emoji_ptn_str, emoticon_ptn_str]) ) 
re_url = re.compile('http://[a-zA-Z0-9_#/.]*')
re_mention = re.compile('@[a-zA-Z0-9:_]+')
re_hashtag = re.compile('#[a-zA-Z0-9:_]+')
re_numeric = re.compile('\$?[0-9]+')
re_punct = re.compile('[#\(\)\[\]\{\}\.!\?:\-<>]')

def clean_tweet(tweet):
    tweet = unicode(tweet)
    tweet = tweet.replace("\n", "")
    tweet = re_url.sub('URL', tweet)   
    tweet = re_mention.sub('MENTION', tweet)
    #tweet = re_hashtag.sub('HASH_TAG', tweet)
    tweet = re_numeric.sub('NUMERIC', tweet)
    tweet = re_emotion.sub('', tweet)
    tweet = re_punct.sub('', tweet)
    return tweet

def _remove_tags(tweet):
    tweet = tweet.replace('MENTION','')
    #tweet = tweet.replace('HASH_TAG', '')
    tweet = tweet.replace('RT', '')
    tweet = tweet.replace('MT', '')
    tweet = tweet.replace('URL', '')
    tweet = tweet.replace('NUMERIC', '')
    return tweet

def tokenize_tweet(tweet):
    tokens = []  
    ctweet = clean_tweet(tweet)
    #Replace this with TweetNLP tokenizer
    #tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(tweet.lower());
    tokens = twk.tokenizeRawTweetText(ctweet)
    return tokens
  
def get_clean_tokens(tweet):
    tokens = []  
    ctweet = _remove_tags(clean_tweet(tweet))
    #Replace this with TweetNLP tokenizer
    #tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(tweet.lower());
    tokens = twk.tokenizeRawTweetText(ctweet)
    return tokens
  
def get_stemmed_tokens(tweet):
    tokens = get_clean_tokens(tweet)
    stemmer = PorterStemmer()
    #stemmer = WordNetLemmatizer()
    stems = [stemmer.stem(word) for word in tokens]
    return stems

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN       
        
def get_lemas(tweet):
    tokens = get_clean_tokens(tweet)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemas = [lemmatizer.lemmatize(tag[0], pos=get_wordnet_pos(tag[1])) for tag in pos_tags]
    return lemas
    
    
def get_pos_tags(tweet):
    tokens = get_clean_tokens(tweet)             
    pos_tags = ptagger.tag(tokens)
    tags = [tag[1] for tag in pos_tags]
    return tags
    
def get_nltk_pos_tags(tweet):
    tokens = get_clean_tokens(tweet)             
    pos_tags = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in pos_tags]
    return tags

if __name__=='__main__':
    tweet = u"RT @HooperHardin: Cuffin season 12 cancelled due to ebola. Pay $12 😂 http://t.co/ggtth45"
    #tweet = "Click http://t.co/23ffr to win"
    tokens = tokenize_tweet(unicode(tweet))
    pos_tags = tagger.tag(tokens)
    stems = get_stemmed_tokens(tweet) 
    lemas = get_lemas(tweet)
    print clean_tweet(tweet)
    print pos_tags
    print stems
    print lemas   