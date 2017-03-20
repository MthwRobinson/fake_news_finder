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
