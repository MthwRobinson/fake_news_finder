import click
from fake_news.scraper import Scraper
from fake_news.vectorizer import Vectorizer
from fake_news.classifier import NewsClassifier
import fake_news.conf as conf
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

@click.command('vectorize', help='Converts articles into vectorized dataframe')
@click.option('--name', type=click.STRING, help='Choose the type of vectorizer')
@click.option('--sample', is_flag=True, help='Samples from real news if true')
def vectorize(name, sample):
    start = datetime.utcnow()
    print('Loading vectorizer ...')
    vectorizer = Vectorizer(name = name, sample = sample)
    vectorizer.build_model()
    vectorizer.save_model()
    end = datetime.utcnow()
    diff = end - start
    seconds = diff.seconds
    minutes = seconds/60
    print(('Done! Vector representation built in %s minutes')%(minutes))
main.add_command(vectorize)

@click.command('test_models', help='Compares the models in the conf file')
@click.option('--iters', default=10, help='Number of iterations')
@click.option('--metric', default='fscore', help='Sets the comparison metric')
@click.option('--pct', default=.8, help='Percentage of articles in trng set')
@click.option('--random_seed', default=8675309, help='Sets the random seed')
def test_models(iters, metric, pct, random_seed):
    start = datetime.utcnow()
    print('Loading news classifier ...')
    # Instantiate the classifier
    model_list = conf.model_list
    nc = NewsClassifier()

    # Compare the models
    perf = nc.compare_models(
        model_list = model_list,
        iters = iters,
        metric = metric,
        pct = pct,
        random_seed = random_seed
    )
    models = perf.keys()
    for model in models:
        print('')
        print('--------------------------------------')
        print(model)
        print(('Mean: %s')%(perf[model]['mean']))
        print(('Median: %s')%(perf[model]['median']))
        print(('Max: %s')%(perf[model]['max']))
        print(('Min: %s')%(perf[model]['min']))
        print('T-tests:')
        for other_model in perf[model]['ttest']:
            pval = perf[model]['ttest'][other_model]
            print(('    P-val (%s): %s')%(other_model, pval))
        print('--------------------------------------')

    # See how long it took!
    end = datetime.utcnow()
    diff = end - start
    seconds = diff.seconds
    minutes = seconds/60
    print(('Done! Models evaluated in %s minutes')%(minutes))
main.add_command(test_models)




if __name__ == '__main__':
    main()
