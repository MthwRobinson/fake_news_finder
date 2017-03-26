from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import os
import conf
from unidecode import unidecode
import random

class Vectorizer():
    """
    The vectorizer chosen must be an sklearn-type vectorizer
    with a fit_transform method that converts the test to 
    a matrix
    """
    def __init__(self, name = None, sample = True):
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.fake_news_file = self.dir + conf.fake_news_file
        self.real_news_file = self.dir + conf.real_news_file

        self.fake_news = conf.fake_news
        self.real_news = conf.real_news

        self.load_examples()


        tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=.5)
        self.vectorizers = {
            'tfidf' : tfidf 
        }

        self.vectorizer = self.vectorizers[name]
        self.name = name
        self.sample = sample

    def load_examples(self):
        try:
            with open(self.fake_news_file,'rb') as f:
                self.fake_news_examples = pickle.load(f)
        except:
            self.fake_news_examples = []
        
        try:
            with open(self.real_news_file,'rb') as f:
                self.real_news_examples = pickle.load(f)
        except:
            self.real_news_examples = []

    def build_model(self, random_seed = 8675309):
        # Build corpus. Sample from the real news because it's
        # much larger
        print('Building corpus ...')
        fake_corpus = [x['content'] for x in self.fake_news_examples]
        real_corpus = [x['content'] for x in self.real_news_examples]
        if self.sample:
            if random_seed:
                random.seed(random_seed)
            num_fake = len(fake_corpus)
            num_real = len(real_corpus)
            idx = random.sample(range(1,num_real), num_fake)
            real_sample = []
            for i in idx:
                real_sample.append(real_corpus[i])
            real_corpus = [x for x in real_sample]

        # Build model
        print('Building model ...')
        self.vectorizer.fit(real_corpus + fake_corpus)
        fake_model = self.vectorizer.transform(fake_corpus)
        real_model = self.vectorizer.transform(real_corpus)

        # Build matrix for training/testing
        print('Building matrix ...')
        fake_df = pd.DataFrame(fake_model.toarray())
        fake_df['y'] = 1
        real_df = pd.DataFrame(real_model.toarray())
        real_df['y'] = 0
        full_df = pd.concat([fake_df, real_df])
        self.model_df = full_df      

    def save_model(self):
        # Either use the specified model name or pick a
        # numbered model name that has not been used yet
        model_dir = self.dir+'/models/vecs'
        print('Saving model ...')
        if self.name:
            filename = model_dir + '/' + self.name + '.pickle'
        else:
            files = os.listdir(model_dir)
            already_used = True
            i = 0
            while already_used:
                filename = 'model_'+ str(i) + '.pickle'
                if filename in files:
                    i += 1
                else:
                    already_used = False
            filename = model_dir + '/' + filename
        with open(filename, 'wb') as f:
            pickle.dump(self.model_df, f)

