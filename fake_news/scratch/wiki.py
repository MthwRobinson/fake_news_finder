#https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
from wikipedia import page
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re

#Get data from wiki : https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
title = "Word2vec"
wikipage = page(title)
raw_content = wikipage.content
alphanumeric_content = re.sub('[^0-9a-zA-Z ]+', ' ', raw_content)

text_file = open("Output.txt", "w")
text_file.write(alphanumeric_content)
text_file.close()

sentences = LineSentence("Output.txt", max_sentence_length=10)
print(sentences)
#exit()

min_count = 2
size = 50
window = 4
model = Word2Vec(sentences, min_count=min_count, size=size, window=window)
#print(model.wv.vocab)
for i in model.wv.vocab:
	print(i)
	print(model[i])
  
#print(model[page_list[0]])
#print(model.batch_words)