# -*- coding: utf-8 -*-
##Reads tweet files and does exploratory analysis

import csv
import pandas as pd
import numpy as np
import glob
import re

data_path = '/home/kaushalya/Code/MCS Project/data/SLvNZ/*.csv'
verified_tweets = pd.DataFrame()
emoji_tweets = pd.DataFrame()
tweet_count = 0

def has_emoji(tweets):
    emoji_ptn_str = u'[\U0001F601-\U0001F64F]'
    emoticon_ptn_str = ':-*[\\)\\(|/@)] '
    subjective_ptn = re.compile( '|'.join([emoji_ptn_str, emoticon_ptn_str]) )
    
    emojis  = [subjective_ptn.search(unicode(t)) for t in tweets]
    return emojis

ind = 0

for tw_file in glob.glob(data_path):
    print tw_file
    tweets = pd.read_csv(tw_file, header=0, index_col='status_id')
    tweet_count += len(tweets)
    verified_tweets = verified_tweets.append(tweets[tweets['verified']==True])
    emoji_tweet_indices = [i for i,t in enumerate(has_emoji(tweets['tweet'])) if t!=None]
    e_t = [tweets.iloc[i] for i in emoji_tweet_indices]    
    emoji_tweets = emoji_tweets.append(tweets.iloc[emoji_tweet_indices])
   
print "Read %d total tweets" % tweet_count   
print "Found %d tweets with emoji" %len(emoji_tweets)
print "Found %d tweets from verified accounts" %len(verified_tweets)
verified_tweets.to_csv('/home/kaushalya/Code/MCS Project/data/SLvNZ/training/training_verified.csv')
emoji_tweets.to_csv('/home/kaushalya/Code/MCS Project/data/SLvNZ/training/training_emoji.csv')
#verified_tweets[['screen_name', 'tweet']].to_csv('/home/kaushalya/Code/MCS Project/data/output/verified_sample.csv')

#with open('../data/verified_sample.csv', 'w', newline='') as output:
#    csv_writer
#    
#if __name__ == '__main__':
#    main()