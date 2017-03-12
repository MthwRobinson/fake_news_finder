from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import os
import conf

VECTORIZERS = [
    TfidfVectorizer(stop_words='english', min_df=5, max_df=.5)
]

class Vectorizer():
    """
    The vectorizer chosen must be an sklearn-type vectorizer
    with a fit_transform method that converts the test to 
    a matrix
    """
    def __init__(self, vectorizer, name = None, test = False):
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.fake_news_file = self.dir + conf.fake_news_file
        self.real_news_file = self.dir + conf.real_news_file

        self.fake_news = conf.fake_news
        self.real_news = conf.real_news

        self.load_examples()

        self.vectorizer = vectorizer
        self.name = name
        self.test = test

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

    def build_model(self):
        # Build corpus
        print('Building corpus ...')
        fake_corpus = [x['content'] for x in self.fake_news_examples]
        real_corpus = [x['content'] for x in self.real_news_examples]
        if self.test:
            fake_corpus = fake_corpus[:500]
            real_corpus = real_corpus[:500]


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
        real_df['y'] = 1
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

