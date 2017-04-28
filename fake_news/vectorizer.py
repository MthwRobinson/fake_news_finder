from sklearn.feature_extraction.text import TfidfVectorizer
from fake_news.word2vec import Word2VecSimple
from fake_news.doc2vec import Doc2VecSimple
import _pickle as pickle
import pandas as pd
import os
from fake_news import conf
from unidecode import unidecode
import random
import sys
import math

class Vectorizer():
    """
    The vectorizer chosen must be an sklearn-type vectorizer
    with a fit_transform method that converts the test to 
    a matrix
    """
    def __init__(self, name = None, sample = True, size = 50, window = 4, min_count=1):
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.fake_news_file = self.dir + conf.fake_news_file
        self.real_news_file = self.dir + conf.real_news_file

        self.fake_news = conf.fake_news
        self.real_news = conf.real_news

        self.load_examples()


        tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=.5)
        word2vec = Word2VecSimple(min_count = 0, size = size, window = window, vectorization_function = "max") #vectorization_function maxmin, max, avg
        doc2vec = Doc2VecSimple(min_count = min_count, size = size, window = window, alpha=0.5, dm=2) 
        self.vectorizers = {
            'tfidf' : tfidf,
            'word2vec' : word2vec,
            'doc2vec' : doc2vec
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

        object_to_be_saved = self.model_df
        rows_count = len(self.model_df.index)
        #due to bug we need to split in sections lower than 2 GB:
        #http://deo.im/2016/09/20/Pickle-can-t-dump-2GB-file/
        if sys.getsizeof(self.model_df) >= 2147483648:
             rows_count = len(self.model_df.index)
             size_byte = sys.getsizeof(self.model_df) 
             size_gbyte = size_byte / (1024 * 1024 * 1024)
             sections_row_count = math.floor( (rows_count * 1.9) / (size_gbyte) ) 
             sections_count = math.ceil(rows_count/sections_row_count)
             next_start = 0
             next_end = sections_row_count
             for j in range(0,sections_count):
                section = self.model_df.iloc[next_start:next_end]
                next_start = next_end
                next_end = next_end + sections_row_count
                if self.name:
                    filename = model_dir + '/' + self.name + '.' + str(j) + '.pickle'
                else:
                    files = os.listdir(model_dir)
                    already_used = True
                    i = 0
                    while already_used:
                        filename = 'model_'+ str(i) + '.' + str(j) + '.pickle'
                        if filename in files:
                            i += 1
                        else:
                            already_used = False
                    filename = model_dir + '/' + filename
                with open(filename, 'wb') as f:
                    pickle.dump(section, f, protocol=4)

        else:
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
                pickle.dump(self.model_df, f, protocol=4)

