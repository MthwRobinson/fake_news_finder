from fake_news.vectorizer import Vectorizer
from fake_news.classifier import NewsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


#vectorizer = Vectorizer('word2vec', sample = True, size = 500, window = 4, min_count= 10)
#vectorizer.build_model()
#vectorizer.save_model()
#print('Finish vectorizer = Vectorizer(word2vec, sample = True, size = 500, window = 4, min_count= 10) ')

vectorizer = Vectorizer('word2vec', sample = False, size = 500, window = 4, min_count= 10)
vectorizer.build_model()
vectorizer.save_model()
print("Finish vectorizer = Vectorizer('word2vec', sample = False, size = 500, window = 4, min_count= 10)")

#vectorizer = Vectorizer('doc2vec', sample = True, size = 3000, window = 10, min_count= 50)
#vectorizer.build_model()
#vectorizer.save_model()
#print("Finish vectorizer = Vectorizer('doc2vec', sample = True, size = 3000, window = 10, min_count= 50)")

vectorizer = Vectorizer('doc2vec', sample = False, size = 3000, window = 10, min_count= 50)
vectorizer.build_model()
vectorizer.save_model()
print("Finish vectorizer = Vectorizer('doc2vec', sample = False, size = 3000, window = 10, min_count= 50)")