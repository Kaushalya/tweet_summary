# -*- coding: utf-8 -*-
'''
created on Fri 17 Oct 2014
@author kaushlaya

tokenizes a set of tweets.

'''
import nltk
import re
import twokenize as twk

def tokenize_tweet(tweet):
    tokens = []
    emoji_ptn_str = u'[\U0001F601-\U0001F64F]'
    emoticon_ptn_str = ':-*[\\)\\(|/@)]'
    #Removes emoticons and emoji from the training dataset
    re_emotion = re.compile( '|'.join([emoji_ptn_str, emoticon_ptn_str]) ) 
    re_url = re.compile('http://.*')
    re_mention = re.compile('@[a-zA-Z0-9:_]+')
    
    tweet = re_emotion.sub('', unicode(tweet))
    tweet = re_url.sub('URL', tweet)
    tweet = re_mention.sub('MENTION', tweet)
   
    #Replace this with TweetNLP tokenizer
    #tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(tweet.lower());
    tokens = twk.tokenizeRawTweetText(tweet)
    return tokens
   
def tag_tweet(tweet):
    RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar /home/kaushalya/Code/MCS\ Project/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar"
    

if __name__=='__main__':
    tweet = "RT @HooperHardin: Cuffin season cancelled due to ebola 😂😴"
    print tokenize_tweet(tweet)    