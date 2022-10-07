from unicodedata import category
from nltk.corpus import brown
import nltk
#nltk.download()
corpus = brown.tagged_words(categories = 'news', tagset = 'universal')
print(corpus)