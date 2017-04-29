from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

fake_news_file = '/data/fake_news.pickle'
real_news_file = '/data/real_news.pickle'

matrix_folder = '/modesl/vecs/'

fake_news = [
    'http://abcnews.com.co',
    'http://drudgereport.com.co',
    'http://usatoday.com.co',
    'http://infowars.com',
    'http://prntly.com',
    'http://naturalnews.com',
    'http://nationalreport.net'
]

real_news = [
    'http://washingtonpost.com',
    'http://nbcnews.com',
    'http://cnn.com',
    'http://npr.org',
    'http://foxnews.com',
    'http://cnbc.com',
    'http://msnbc.com',
    'http://reuters.com',
    'http://bloomberg.com',
    'http://cbs.com',
    'http://nytimes.com',
    'http://wsj.com',
    'http://bbc.com'
]

model_list = [
    {
        'name' : 'Naive Bayes: Doc2Vec',
        'clf' : MultinomialNB(),
        'vec' : 'doc2vec'

    },
    {
        'name' : 'Naive Bayes: Word2Vec',
        'clf' : MultinomialNB(),
        'vec' : 'word2vec'

    },
    {
        'name' : 'Random Forest: Doc2Vec',
        'clf' : RandomForestClassifier(n_estimators=70),
        'vec' : 'doc2vec'
    },
    {
        'name' : 'Random Forest: Word2Vec',
        'clf' : RandomForestClassifier(n_estimators=70),
        'vec' : 'word2vec'
    }#,
    #{
    #    'name' : 'Support Vector Machine: Doc2Vec',
    #    'clf' : SVC(kernel='linear'),
    #    'vec' : 'tfidf'
    #},
    #{
    #    'name' : 'Support Vector Machine: Word2Vec',
    #    'clf' : SVC(kernel='linear'),
    #    'vec' : 'word2vec'
    #},
    #{
    #    'name' : 'Logistic Regression: TFIDF',
    #    'clf' : LogisticRegression(),
    #    'vec' : 'tfidf'
    #},
    #{
    #    'name' : 'Logistic Regression: Word2Vec',
    #    'clf' : LogisticRegression(),
    #    'vec' : 'word2vec'
    #},
]
