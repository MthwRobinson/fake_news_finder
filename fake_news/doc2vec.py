from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import math
import re

class Doc2VecSimple:


  def __init__(self, min_count = 1, size = 500, window = 4,  alpha=0.5, dm=1):
    self.min_count = min_count
    self.size = size
    self.window = window
    self.normalize = True
    self.alpha = alpha
    self.dm = dm


  def fit(self, corpus):
    taggedDocs = []
    self.docDictionary = dict()
    tag = 0
    for raw_document in corpus:
      doc_id = raw_document[:100]
      self.docDictionary[doc_id] = tag
      alphanumeric_content = re.sub('[^0-9a-zA-Z ]+', ' ', raw_document)
      words = ' '.join(alphanumeric_content.split()).split()
      tags = []
      tags.append(tag)
      taggedDoc = TaggedDocument(words, tags)
      taggedDocs.append(taggedDoc)
      tag = tag + 1

    self.model = Doc2Vec(taggedDocs, min_count=100, size=self.size, window=self.window, alpha=self.alpha, dm=2, iter = 20)

    self.normalized_model = []
    for doc in self.model.docvecs:
      self.normalized_model.append(doc)

    # Scaling:
    self.normalized_model = np.array(self.normalized_model)
    self.normalized_model = np.apply_along_axis(scaling, axis=0, arr=self.normalized_model)
    self.normalized_model = self.normalized_model.tolist()
    # Normalizing:
    #   w = csr_matrix(np.asarray(self.normalized_model))
    #   self.normalized_model = normalize(w, norm='l1', axis=0)
    #   self.normalized_model = self.normalized_model.todense().tolist()
    #   print(self.normalized_model[:,26:])
    print("Finish model")


  def transform(self, corpus):
    if self.model is None:
      print("model was not created. Please call fit method before transform. Exiting!")
      exit()

    vectorization = []
    for raw_document in corpus:
      doc_id = raw_document[:100]
      tag = self.docDictionary[doc_id]
      #Process the document
      vectorization_item = self.normalized_model[tag]
      if vectorization_item is not None:
        vectorization.append(vectorization_item)

    w = csr_matrix(np.asarray(vectorization))
    return w

 

def scaling(x):
  maxim = max(x)
  minim = min(x)

  if maxim == minim:
    size = len(x)
    return np.ones(size) * 0.5
  return (x - minim) / (maxim - minim)


