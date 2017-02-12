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
