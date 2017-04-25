from __future__ import division
import sys, os
from sklearn.ensemble import RandomForestClassifier
import conf
if sys.version_info >= (3,0):
    import pickle
else:
    import cPickle as pickle
import random
import numpy as np
from scipy.stats import ttest_ind


class NewsClassifier():
    """
    Trains a classifier for fake vs. real news
    clf is a sklearn type classifier
    vec is the name of the pickle file with the vectorized
        news article examples
    """
    def __init__(self, clf = None, vec = None):
        """
        Instantiates the class
        clf is an sklearn-type classfier
        vec is a string that corresponds to a pickle file
        """
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.vec = None
        if vec:
            self.set_vectorizer(vec)
        if clf:
            self.set_classifier(clf)

    def set_classifier(self, clf):
        """
        Sets the classifier
        """
        self.clf = clf

    def set_vectorizer(self, vec):
        """
        Sets the vectorizer for the feature space
        """
        # Only import the model if the vectorizer has changed
        if self.vec != vec:
            vec_file = self.dir + '/models/vecs/' + vec + '.pickle'
            with open(vec_file, 'rb') as f:
                self.df = pickle.load(f)
        self.vec = vec

    def train_test_sets(self, pct, random_seed = 21189):
        """
        Splits the data into training and test sets
        """
        # Set the random seed for reproducibility
        if random_seed:
            random.seed(random_seed)

        # Choose which indices will belong to each set
        num_articles = self.df.shape[0]
        idx = range(num_articles)
        self.df.index = idx
        num_train = int(num_articles*pct)
        num_test = num_articles - num_train
        train_idx = random.sample(idx, num_train)
        test_idx = [x for x in idx if x not in train_idx]

        # Constructing the training and test sets
        self.train_df = self.df.iloc[train_idx]
        self.test_df = self.df.iloc[test_idx]

    def train_model(self):
        """ 
        Trains the classifier model
        """
        # Turn the training data frame into an appropriate
        # matrix / vector
        y = self.train_df['y']
        X = self.train_df.copy()
        del X['y']

        # Train the model
        self.clf.fit(X,y)

    def predict_labels(self):
        """
        Predicts the labels of the test set
        """
        # Turn the test data frame into an appropriate
        # matrix / vector
        X = self.test_df.copy()
        del X['y']

        # Predict the label of the test examples
        self.ypred = self.clf.predict(X)

    def evaluate_model(self):
        """
        Evaluates the effectiveness of the models according
            to a number of metrics for classifier performance
        """
        # Check the predictions against observed values
        y = list(self.test_df['y'])
        ypred = list(self.ypred)
        tp = 0 # True positives
        tn = 0 # True negatives
        fp = 0 # False positives
        fn = 0 # False negatives
        for i in range(len(y)):
            if y[i] == 1 and ypred[i] == 1:
                tp += 1
            if y[i] == 0 and ypred[i] == 0:
                tn += 1
            if y[i] == 1 and ypred[i] == 0:
                fp += 1
            if y[i] == 0 and ypred[i] == 1:
                fn += 1

        # Compute summary statistics
        total = tp+tn+fp+fn
        accuracy = (tp+tn) / total
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        fscore = 2*((precision*recall)/(precision+recall))
        stats = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'fscore' : fscore
        }
        return stats

    def bootstrap_evaluate(self, iters = 30, metric = 'fscore', 
            pct = .8, random_seed = None):
        """
        Evaluates the effectiveness of the model on
            various random partitions of the data
        Saves a vector of performance statistics that
            can be used in a two sample t-test
        This procedure may be expensive for large sample sizes
        """
        # Set the random seed
        if random_seed:
            random.seed(random_seed)

        # Evaluate the model on b different random partitions
        results = []
        for i in range(iters):
            self.train_test_sets(pct = pct)
            self.train_model()
            self.predict_labels()
            stats = self.evaluate_model()
            results.append(stats[metric])
        return results

    def compare_models(self, model_list, iters = 30, metric = 'fscore',
            pct = .8, random_seed = None):
        """
        Compares the performance of different models and 
            determines if there is a statistically
            significant difference in their performance
        Models are specified as follows, and entered in
            as list
            model = {
                'name' : 'RandomForest',
                'clf' : RandomForestClassifier(n_estimators=70)
                'vec' : 'tfidf'
            }
        """
        # Check to make sure all of the models have the
        #   proper keys
        for model in model_list:
            bad = False
            if 'name' not in model:
                bad = True
            if 'clf' not in model:
                bad = True
            if 'vec' not in model:
                bad = True
            if bad:
                err = 'Each model must have name, clf and vec keys'
                raise TypeError(err)

        # Compute the results for each model
        # The same random seed gets used for each model, to ensure a
        #   that they are evaluated on the same set of data
        for model in model_list:
            print(('Now testing %s')%(model['name']))
            self.set_classifier(model['clf'])
            self.set_vectorizer(model['vec'])
            results = self.bootstrap_evaluate(iters = iters, metric = metric,
                    pct = pct, random_seed = random_seed)
            model['results'] = results

        # Compare the results and see if there is a statistically
        #   significant difference in model performance
        perf = {}
        for model in model_list:
            model_perf = {}
            model_perf['results'] = model['results']
            model_perf['mean'] = np.mean(model['results'])
            model_perf['median'] = np.mean(model['results'])
            model_perf['max'] = np.max(model['results'])
            model_perf['min'] = np.min(model['results'])
            model_perf['ttest'] = {}
            # Perform a two sample t-test with all of the other
            #   models to see if there is a statistically
            #   significant difference in performance
            for other_model in model_list:
                if model['name'] != other_model['name']:
                    ttest = ttest_ind(model['results'], other_model['results'])
                    pval = ttest.pvalue
                    model_perf['ttest'][other_model['name']] = pval
            perf[model['name']] = model_perf
        return perf




