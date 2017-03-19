import conf
import pytz
import datetime
import os
import sys
if sys.version_info >= (3,0):
    import newspaper3k as newspaper
    from newspaper3k import Source, Article
    import pickle
else:
    import newspaper
    import cPickle as pickle
    from newspaper import Source, Article

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

    def build_real_news(self):
        for url in self.real_news:
            real_docs = self.build_articles(url)
            for doc in real_docs:
                self.real_news_examples.append(doc)

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

if __name__ == '__main__':
    scraper = Scraper()
    scraper.scrape_examples()
