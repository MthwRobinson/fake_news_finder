import pickle

with open('/Users/sergiosainz/Projects/vtech/DAII/project/data/real_news.pickle', 'rb') as pickle_file:
	content = pickle.load(pickle_file)

print(len(content[0]))
print(content[0]['content'])