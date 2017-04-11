from fake_news.vectorizer import Vectorizer
from fake_news.classifier import NewsClassifier
from sklearn.ensemble import RandomForestClassifier

vectorizers = ['word2vec']
Sample = False 
for vectorizer_name in vectorizers:
	vectorizer = Vectorizer(vectorizer_name, sample = Sample)
	vectorizer.build_model()
	vectorizer.save_model()


#print("After building model")


vectorizers = ['word2vec']
for vectorizer in vectorizers:
	clf = RandomForestClassifier(n_estimators=70)
	nc = NewsClassifier(clf, vectorizer)
	results = nc.bootstrap_evaluate(iters = 1, metric = 'fscore', pct = .8)
	#results = nc.evaluate_model()
	print("Printing results: " + vectorizer)
	print(results)

