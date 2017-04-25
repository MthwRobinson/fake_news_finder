from fake_news.vectorizer import Vectorizer
from fake_news.classifier import NewsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


vectorizers = ['word2vec']
#vectorizers = ['word2vec', 'tfidf']
#Sample = False
#for vectorizer_name in vectorizers:
#	vectorizer = Vectorizer(vectorizer_name, sample = Sample)
#	vectorizer.build_model()
#	vectorizer.save_model()
#
#exit()

#print("After building model")


vectorizers = ['tfidf']
#classifiers = ['random forests']
classifiers = ['bayes','svm']
metrics = ['fscore']
iterations = 1
for vectorizer in vectorizers:
	
	for classifier in classifiers:

		for metric in metrics:

			if classifier == 'random forests':
				print("creating random forest classifier")
				clf = RandomForestClassifier(n_estimators=70, class_weight = "balanced")
			elif classifier == 'svm':
				print("creating SVM classifier")
				#clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
				clf = svm.SVC(kernel='linear')
			elif classifier == 'bayes':
				print("creating Bayes classifier")
				clf = MultinomialNB(alpha=0.5, fit_prior=True, class_prior=None)
			elif classifier == 'MLP':
				clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
			nc = NewsClassifier(clf, vectorizer)
			results = nc.bootstrap_evaluate(iters = iterations, metric = metric, pct = .8)
			#results = nc.evaluate_model()
			print("[" + classifier + "] Printing results [" + metric + "]: " + vectorizer)
			print(results)
	

