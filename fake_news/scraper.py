import conf
import pytz
import datetime
import os, sys
from copy import deepcopy
from unidecode import unidecode
#if sys.version_info >= (3,0):
#    import pickle
#    import newspaper3k as newspaper
#    from newspaper3k import Source, Article
#else:
import pickle
#import cPickle as pickle
from newspaper import Source, Article
import newspaper

class Scraper():
    def __init__(self):
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.fake_news_file = self.dir + conf.fake_news_file
        self.real_news_file = self.dir + conf.real_news_file

        self.fake_news = conf.fake_news
        self.real_news = conf.real_news

        self.load_examples()

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

    def save_examples(self):
        with open(self.fake_news_file,'wb') as f:
            pickle.dump(self.fake_news_examples, f)
        with open(self.real_news_file,'wb') as f:
            pickle.dump(self.real_news_examples, f)

    def scrape_examples(self):
        print('Loading examples ..')
        self.load_examples()
        print('Scraping fake news ...')
        self.build_fake_news()
        print('Scraping real news ...')
        self.build_real_news()
        print('Saving examples')
        self.save_examples()

    def build_fake_news(self):
        for url in self.fake_news:
            fake_docs = self.build_articles(url)
            for doc in fake_docs:
                self.fake_news_examples.append(doc)
        print('Cleaning fake news')
        cleaned = self.clean_examples(self.fake_news_examples)
        self.fake_news_examples = cleaned

    def build_real_news(self):
        for url in self.real_news:
            real_docs = self.build_articles(url)
            for doc in real_docs:
                self.real_news_examples.append(doc)
        cleaned = self.clean_examples(self.fake_news_examples)
        self.fake_news_examples = cleaned

    def build_articles(self, url):
        """
        Takes in a url and returns a list of dictionaries
        with data about a news story
        """
        paper = newspaper.build(url)
        articles = []
        for url in paper.article_urls():
            article = Article(url, keep_article_html = True, request_timeout = 5)
            try:
                article.download()
                article.parse()
                print("Building %s :  %s"%(paper.brand,article.title))
            except:
                print("Couldn't download article")

            doc = {}
            doc['title'] = article.title
            doc['date'] = datetime.datetime.now(pytz.utc).isoformat()
            doc['url'] = article.url
            doc['content'] = article.text 
            doc['source'] = paper.brand
            doc['html'] = article.article_html
            articles.append(doc)
        return articles

    def clean_articles(self, articles):
        good_articles = []
        titles = []
        for i, article_ in enumerate(articles):
            if i%1000 == 0:
                print(('%s articles completed')%(i))
            article = deepcopy(article_) 
            # Get rid of common tag lines
            blurbs = [
                "There's a war on for your mind!",
                "Alex Jones' Infowars:",
                "NaturalNews.com",
                "2013 NaturalNews.com"
            ]
            for blurb in blurbs:
                article['content'] = article['content'].replace(blurb,'')
                article['title'] = article['title'].replace(blurb,'')
            article['content'] = str(unidecode(article['content']))
            article['title'] = str(unidecode(article['title']))

            # Skip bad articles
            skip = False
            if len(article['content']) == 0:
                skip = True
            if 'page not found' in article['content'].lower():
                skip = True
            if 'page not found' in article['title'].lower():
                skip = True
            if '404' in article['content']:
                skip = True
            if 'reuters' in article['source'].lower():
                skip = True
            if article['title'] in titles:
                skip = True
            
            # Keep the good articles! :)
            if not skip:
                good_articles.append(article)
                titles.append(article['title'])

        return good_articles


                


if __name__ == '__main__':
    scraper = Scraper()
    scraper.scrape_examples()
