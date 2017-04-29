from fake_news.vectorizer import Vectorizer
from fake_news.classifier import NewsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

windows = [1,2,3, 4, 6, 10, 20, 50, 100]
sizes = [10, 25, 50, 100, 300, 500, 750, 1000, 2000]
#windows = [4]
#min_counts = [1, 10, 50, 100, 300, 500, 900, 1500]
#sizes = [50, 500, 750, 1000, 2000, 3000, 500]
#windows = [1]
#sizes = [1]

min_coun = 1
size_results = []
for siz in sizes:
	windows_results = []
	for windo in windows:
		vectorizers = ['doc2vec']
		#vectorizers = ['word2vec', 'tfidf']
		Sample = True
		create_model = True
		if create_model:
			for vectorizer_name in vectorizers:
				vectorizer = Vectorizer(vectorizer_name, sample = Sample, size = siz, window = windo, min_count= min_coun)
				vectorizer.build_model()
				vectorizer.save_model()

		#exit()

		#print("After building model")


		classifiers = ['random forests']
		#classifiers = ['random forests','bayes','svm']
		metrics = ['fscore']
		iterations = 5
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
					
					nc = NewsClassifier(clf=clf, vec=vectorizer)
					results = nc.bootstrap_evaluate(iters = iterations, metric = metric, pct = .8)
					#results = nc.evaluate_model()
					print("[MinCount:"+str(min_coun)+"][Size:"+str(siz)+"][Window:"+str(windo)+"][" + classifier + "] Printing results [" + metric + "]: " + vectorizer)
					print(results)
					#if results[0] is not None and max_score < results[0]:
					#	max_score = results[0]
					windows_results.append(results)
	size_results.append(windows_results)
print(size_results)



