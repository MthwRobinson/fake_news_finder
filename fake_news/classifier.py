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


class NewsClassifier():
    """
    Trains a classifier for fake vs. real news
    clf is a sklearn type classifier
    vec is the name of the pickle file with the vectorized
        news article examples
    """
    def __init__(self, clf, vec):
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.clf = clf
        vec_file = self.dir + '/models/vecs/' + vec + '.pickle'
        with open(vec_file, 'rb') as f:
            self.df = pickle.load(f)

    def train_test_sets(self, pct, random_seed = 21189):
        """
        Splits the data into training and test sets
        """
        # Set the random seed for reproducibility
        if random_seed:
            random.seed(random_seed)

        # Choose which indices will belong to each st
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
        # Turn the training data frame into an appropriate
        # matrix / vector
        y = self.train_df['y']
        X = self.train_df.copy()
        del X['y']

        # Train the model
        self.clf.fit(X,y)

    def predict_labels(self):
        # Turn the test data frame into an appropriate
        # matrix / vector
        X = self.test_df.copy()
        del X['y']

        # Predict the label of the test examples
        self.ypred = self.clf.predict(X)

    def evaluate_model(self):
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






