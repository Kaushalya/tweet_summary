# -*- coding: utf-8 -*-
import re
#import nltk.corpus.reader.wordnet
from nltk.corpus import wordnet
from nltk.wsd import lesk
import nltk
import nlpnet
from tweet_tokenizer import clean_tweet
from tweet_tokenizer import get_clean_tokens

from nltk.tree import Tree

#text1 = 'They are going to rob the bank'
#text2 = 'I paid using my credit card'
#
#tweet = "Ebola Screenings Begin at New York's JFK Airport - Wall Street Journal http://t.co/UTLCy2UjLU"
#tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(tweet.lower());
#pos_tags = nltk.pos_tag(tokens)
#ner = nltk.ne_chunk(pos_tags)
#
#word = 'Screenings'
#syns = wordnet.synsets(word)
#s1 = syns[0]
#
tweet = 'RT @NickKristof: Congrats to Dr. Craig Spencer, now cured of Ebola and ready to leave hospital in NY. Of 9 Ebola patients in US hospitals, …'
tweet2 = 'RT @MicahGrimes: NEW: Craig Spencer, doctor who has been treated for Ebola at a NYC hospital, has been declared free of the virus. http://t…'
tweet3 = "CBS' Lara Logan voluntarily quarantined after filing Ebola report from Liberia  (from @AP) http://t.co/dwiqmHz3mj"
#nlpnet.set_data_dir('/home/kaushalya/Code/MCS Project/nlpnet_models/srl')
#tagger = nlpnet.SRLTagger()
#tags = tagger.tag(unicode(tweet))

tokens = get_clean_tokens(tweet3)
pos_tags = nltk.pos_tag(tokens)
ner = nltk.ne_chunk(pos_tags)

print ner
entities = []
for entity in ner:
    if isinstance(entity,Tree):
        print "entity"+str(entity)
        entities.append(str(entity))
#print(ner)