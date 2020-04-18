import nltk, random
from nltk.corpus import movie_reviews

nltk.download('names')
nltk.download('movie_reviews')

TRAIN_RATIO = 0.9

class MRClass(object):
  def __init__(self):
    self.doc = [(list(movie_reviews.words(id)), cat) for cat in movie_reviews.categories() for id in movie_reviews.fileids(cat)]
    random.shuffle(self.doc)
    self.word_corpa = nltk.FreqDist([ w.lower() for w in movie_reviews.words()])
    self.wordlist = list(set(self.word_corpa.keys()))
    self.features = [(MRClass.get_features(self.wordlist, doc), cat) for doc , cat in self.doc]
    train_size = int(len(self.features)*TRAIN_RATIO)
    self.train_set = self.features[:train_size]
    self.test_set = self.features[train_size:]
    self.clf = nltk.NaiveBayesClassifier.train(self.train_set)

  @staticmethod
  def get_features(wordlist,movie_review):
    features = dict()
    for word in wordlist:
      features['contains' + word] = word in movie_review
    return features


  #def classify(self, name): 
  #  print('Name: ', name, '\n' , 'Gender: ', self.clf.classify(MRClass.get_features(self.wordlist, name)))

  def accuracy(self):
    print('Training Accuracy: ', nltk.classify.accuracy(self.clf, self.train_set))
    print('Test Accuracy: ', nltk.classify.accuracy(self.clf, self.test_set))

  def informative_features(self, n = 10):
    print('Most Informative features: ', self.clf.show_most_informative_features(n))


if __name__ == '__main__':
  app = MRClass()
  app.accuracy()
  app.informative_features
  #app.classify(nltk.word_tokenize('Mohit'))
