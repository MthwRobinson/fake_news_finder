#https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
from wikipedia import page
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re

titles = ["Word2vec", "Giant panda", "Artificial intelligence"]
taggedDocs = []
tag = 0
for title in titles:

	wikipage = page(title)
	raw_content = wikipage.content
	alphanumeric_content = re.sub('[^0-9a-zA-Z ]+', ' ', raw_content)
	words = ' '.join(alphanumeric_content.split()).split()
	tags = []
	tags.append(tag)
	taggedDoc = TaggedDocument(words, tags)
	taggedDocs.append(taggedDoc)
	tag = tag + 1

min_count = 2
size = 50
window = 4
alpha = 0.5
dm = 1
model = Doc2Vec(taggedDocs, min_count=min_count, size=size, window=window, alpha=alpha, dm=dm)
#model.save("savedModel")

for docvec in model.docvecs:
	print(docvec)
