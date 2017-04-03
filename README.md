# Fake News Finder
A machine learning model that takes in news stories as input, and classifies them as either real or fake.

By Team Panda

:panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face::panda\_face:

## Installation
```
git clone git@github.com:MthwRobinson/fake_news_finder.git
cd fake_news_finder
pip install -e .
```

## Scrape News Examples
From a python interactive session, use:
```python
from fake_news.scraper import Scraper
scraper = Scraper()
scraper.scrape_examples()
```
Or you can use the CLI
```bash
fake_news scrape --real --fake
```

## Build matrices for training / testing the classifier
The vectorizer class takes as input an sklearn type vectorizer with
a fit and transform method and then produces and saves a matrix that can be used to train a binary classifer. An example with tfidf is shown below. The model can be accessed as a dataframe through the model\_df attribute.
```python
from fake_news.vectorizer import Vectorizer
vectorizer = Vectorizer('tfidf', sample = True)
vectorizer.build_model()
vectorizer.save_model()
```

Or you can use the CLI
```bash
fake_news vectorize --name tfidf --sample
```

## Test a model
The news classifier class contains methods for training and evaluating
a model. To instantiate the news classifier class, used the following
code
```python
from fake_news.classifier import NewsClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=70)
nc = NewsClassifier(clf, 'tfidf')
```

The bootstrap evaluate method evaulates the performace of the model
by testing it on n random partitions of the data set. The number
of random splits is set by the iters parameter. It returns a vector
of results that can be used in two sample t tests. The model can
be evaluated using accuracy, precision, recall or fscore.
```python
results = nc.bootstrap_evaluate(iters = 30, metric = 'fscore', pct = =.8)
```

We can test a bunch of models against one another at the same time
using the compare models method. The models are entered as a list, and
each model is a dicitonary with the following form:
```python
model = {
  'name' : 'Random Forest',
  'clf' : RandomForestClassifier(n_estimators=70),
  'vec' : 'tfidf'
}
```

You can test the models inn python using the following code:
```python
perf = nc.compare_models(
  model_list = model_list,
  iters = iters,
  metric = metric,
  pct = pct
  random_seed = random_seed
)
```

Or from the CLI with the following command
```bash
fake_news test_models --iters 30 --metric fscore --pct .8 --random_seed 20
```

The list of models is stored in the `conf.py` file.
