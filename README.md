# Fake News Finder
A machine learning model that takes in news stories as input, and classifies them as either real or fake

## Installation
```
git clone git@github.com:MthwRobinson/fake_news_finder.git
cd fake_news_finder
pip install -e .
```

## Scrape News Examples
```python
from fake_news.scraper import Scraper
scraper = Scraper()
scraper.scrape_examples()
```
## Build matrices for training / testing the classifier
The vectorizer class takes as input an sklearn type vectorizer with
a fit and transform method and then produces and saves a matrix that can be used to train a binary classifer. An example with tfidf is shown below. The model can be accessed as a dataframe through the model\_df attribute.
```python
from fake_news.vectorizer import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=.5)
vectorizer = Vectorizer(tfidf)
vectorizer.build_model()
vectorizer.save_model()
```
