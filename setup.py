# To upload to PyPi run python setup.py sdist upload -r pypi
from distutils.core import setup
from pip.req import parse_requirements

# install_reqs = parse_requirements('./requirements.txt')
# reqs = [str(x.req) for x in install_reqs]

setup(
    name='fake_news',
    version='0.0.1',
    author='Matthew Robinson',
    author_email='mthw.wm.robinson@gmail.com',
    packages=['fake_news','fake_news'],
    url='https://github.com/MthwRobinson/fake_news_finder',
    description='machine learning classifier for fake news',
    # install_requires=reqs
)
