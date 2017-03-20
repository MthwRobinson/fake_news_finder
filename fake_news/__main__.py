import click
from fake_news.scraper import Scraper
from datetime import datetime

@click.group()
def main():
    """
    Welcome to the Fake News Finder CLI!

    To learn more about a command, type the command and the --help flag

    $ fake_news scrape --help
    """
    pass

@click.command('scrape', help='Scrapes news articles')
@click.option('--fake', is_flag=True, help='Scrapes fake news if true')
@click.option('--real', is_flag=True, help='Scrapes real news if true')
def scrape(fake, real):
    start = datetime.utcnow()
    scraper = Scraper()
    print('Loading examples ...')
    scraper.load_examples()
    if fake:
        print('Scraping fake news ...')
        scraper.build_fake_news()
    if real:
        print('Scraping real news ...')
        scraper.build_real_news()
    print('Saving examples ...')
    scraper.save_examples()
    end = datetime.utcnow()
    diff = end - start
    seconds = diff.seconds
    minutes = seconds/60
    print(('Done! Scraping took %s minutes')%(minutes))

main.add_command(scrape)

if __name__ == '__main__':
    main()
