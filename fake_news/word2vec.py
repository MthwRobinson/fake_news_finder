from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import math
import re

class Word2VecSimple:


  def __init__(self, min_count = 1, size = 50, window = 4, vectorization_function = "maxmin"):
    self.min_count = min_count
    self.size = size
    self.window = window
    self.vectorization_function = vectorization_function

  def fit(self, corpus):
    print("Fit method is empty for word2vec vectorization because each document is modelled independently. In later tests, we can try to model all documents together.")


  def transform(self, corpus):
    vectorization = []
    for raw_content in corpus:
      if(self.vectorization_function == 'maxmin'):
        vectorization_item = self.get_doc2vec_maxmin(raw_content)
        if vectorization_item is not None:
          vectorization.append(vectorization_item)
      else:
        vectorization_item = self.get_doc2vec_avg(raw_content)
        if vectorization_item is not None:
          vectorization.append(vectorization_item)

    w = csr_matrix(np.asarray(vectorization))
    return normalize(w, norm='l1', axis=0)

  def get_doc2vec_avg(self, raw_content):
    words_factors_list_of_lists = get_doc2vec(raw_content)
    if words_factors_list_of_lists is None:
      return None
    #axis = 0 means apply operations across the rows. If arr is n rows x m cols, at end we get 1 row and m columns.
    doc_vec = np.apply_along_axis(get_mean, axis=0, arr=words_factors_list_of_lists)
    return doc_vec


  def get_doc2vec_maxmin(self, raw_content):
    words_factors_list_of_lists = self.get_doc2vec( raw_content)
    if words_factors_list_of_lists is None:
      return None
    magnitudes = np.apply_along_axis(get_magnitude, axis=1, arr=words_factors_list_of_lists)
    index_of_max = np.argmax(magnitudes)
    index_of_min = np.argmin(magnitudes)
    doc_vec = np.append(words_factors_list_of_lists[index_of_max], words_factors_list_of_lists[index_of_min])
    return doc_vec

  def get_doc2vec(self, raw_content):
    alphanumeric_content = re.sub('[^0-9a-zA-Z ]+', ' ', raw_content)

    #maybe add here some stop-word removal if needed ?
    text_file = open("Output.txt", "w")
    text_file.write(alphanumeric_content)
    text_file.close()

    sentences = LineSentence("Output.txt", max_sentence_length=10)
    #print(sentences)

    #Ref: https://radimrehurek.com/gensim/models/word2vec.html 
    #min_count = 2
    #size = 50
    #window = 4
    #print(self.min_count)
    #print(self.size)
    #print(self.window)
    try:
      model = Word2Vec(sentences, min_count=self.min_count, size=self.size, window=self.window)
      size_vocab = 0

      words_factors_list_of_lists = []
      for i in model.wv.vocab:
        words_factors_list_of_lists.append(model[i])
      return words_factors_list_of_lists
    except Exception as exception:
      print(exception.args)
      print(alphanumeric_content)
      print(raw_content)
      return None
    #print(model.wv.vocab)
    
    

def get_mean(x):
  return np.mean(x)

def get_magnitude(x):
  return math.sqrt(np.inner(x,x))
