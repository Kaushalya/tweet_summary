# -*- coding: utf-8 -*-
import re

text = ["I don't have Ebola fyi lol 😂", "hello world"]

#1F601 - 1F64F
pattern = re.compile(u'[\U0001F601-\U0001F64F]')

emojis  = [pattern.search(unicode(t)) for t in text]
emoji_indices = [unicode(text[i]) for i,t in enumerate(emojis) if t!=None]

#emoji_texts = [unicode(text[i]) for i in emoji_indices ] 