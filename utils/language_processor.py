
# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import brown
import pandas as pd
import json

'''
Creates a json file to submit to www.sentiment140.com
'''

def main():
    #tweet = "DeKalb County officials urge Ebola education http://t.co/Rg9bZCvbXh"
    tweet = "Hello I love you :-)"

    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(tweet)
    tags = nltk.pos_tag(tokens)
    print tags

#brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
#tag_fd = nltk.FreqDist(word for (word, tag) in brown_news_tagged if tag=='VBP')
#tweet_sentiments = pd.read_json('/home/kaushalya/Code/MCS Project/data/json_requests/sample_tweets.json')
sentiment_file = open('/home/kaushalya/Code/MCS Project/data/json_requests/sentiments_sample.json', 'r')
json_str = sentiment_file.read()
json_str = json_str.replace("\n","")
sentiment_file.close()

json_str = json_str[8:-1]

#Convert json string to unicode, otherwise this raises an encoding error
json_str = unicode(json_str, 'latin-1')
#tweet_sentiments = json.loads(json_str)
tweet_sentiments = pd.read_json(json_str)
tweet_sentiments.to_csv('/home/kaushalya/Code/MCS Project/data/tweet_sentiments.csv',
                        index=False, quote=False)

if __name__ == '__main__':
    main()
