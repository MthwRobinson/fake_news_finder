#https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
from wikipedia import page
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import math
import re

def vectorize(raw_contents_list, vectorization_function):
	vectorization = []
	for raw_content in raw_contents_list:
		vectorization.append(vectorization_function(raw_content))
	return vectorization


def get_doc2vec_avg(raw_content):
	words_factors_list_of_lists = get_doc2vec(raw_content)
	#axis = 0 means apply operations across the rows. If arr is n rows x m cols, at end we get 1 row and m columns.
	doc_vec = np.apply_along_axis(get_mean, axis=0, arr=words_factors_list_of_lists)
	return doc_vec


def get_doc2vec_maxmin(raw_content):
	words_factors_list_of_lists = get_doc2vec(raw_content)
	magnitudes = np.apply_along_axis(get_magnitude, axis=1, arr=words_factors_list_of_lists)
	index_of_max = np.argmax(magnitudes)
	index_of_min = np.argmin(magnitudes)
	doc_vec = np.append(words_factors_list_of_lists[index_of_max], words_factors_list_of_lists[index_of_min])
	return doc_vec

def get_doc2vec(raw_content):
	alphanumeric_content = re.sub('[^0-9a-zA-Z ]+', ' ', raw_content)

	#maybe add here some stop-word removal if needed ?

	text_file = open("Output.txt", "w")
	text_file.write(alphanumeric_content)
	text_file.close()

	sentences = LineSentence("Output.txt", max_sentence_length=10)
	#print(sentences)

	#Ref: https://radimrehurek.com/gensim/models/word2vec.html 
	min_count = 2
	size = 50
	window = 4
	model = Word2Vec(sentences, min_count=min_count, size=size, window=window)
	#print(model.wv.vocab)
	size_vocab = 0

	words_factors_list_of_lists = []
	for i in model.wv.vocab:
		words_factors_list_of_lists.append(model[i])
	return words_factors_list_of_lists
	

def get_mean(x):
	return np.mean(x)

def get_magnitude(x):
	return math.sqrt(np.inner(x,x))


#Get data from wiki : https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
titles = ["Word2vec", "Machine learning", "Giant panda"]
raw_contents_list = []
for title in titles:
	wikipage = page(title)
	raw_content = wikipage.content	
	raw_contents_list.append(raw_content)

#Average:
vectorization = vectorize(raw_contents_list, get_doc2vec_avg)
print(vectorization)

#MaxMin:
vectorization = vectorize(raw_contents_list, get_doc2vec_maxmin)
print(vectorization)

