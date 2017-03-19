# To upload to PyPi run python setup.py sdist upload -r pypi
from distutils.core import setup

setup(
    name='fake_news',
    version='0.0.1',
    author='Matthew Robinson',
    author_email='mthw.wm.robinson@gmail.com',
    packages=['fake_news','fake_news'],
    url='https://github.com/MthwRobinson/fake_news_finder',
    description='machine learning classifier for fake news',
    entry_points = {
        'console_scripts' : ['fake_news=fake_news.__main__:main']
    }
)
