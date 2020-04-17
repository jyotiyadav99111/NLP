import nltk, random
nltk.download('names')

TRAIN_RATIO = 0.9

class GenderApp(object):
  def __init__(self):
    self.sample = [(name, 'Male') for name in nltk.corpus.names.words('male.txt')] + [(name, 'Female') for name in nltk.corpus.names.words('female.txt')]
    random.shuffle(self.sample)
    self.features = [(GenderApp.get_features(name), gender) for name, gender in self.sample]
    train_size = int(len(self.features)*TRAIN_RATIO)
    self.train_set = self.features[:train_size]
    self.test_set = self.features[train_size:]
    self.clf = nltk.NaiveBayesClassifier.train(self.train_set)

  @staticmethod
  def get_features(name):
    word = str(name).lower()
    features = dict()
    features['Firstname'] = name[0]
    features['Lastname'] = name[-1]
    for char in 'abcdefghijklmnopqrstuvwxyz':
      features['count' + char] = word.count(char)
      features['has' + char] = char in word
    return features

  def classify(self, name): 
    print('Name: ', name, '\n' , 'Gender: ', self.clf.classify(GenderApp.get_features(name)))

  def accuracy(self):
    print('Training Accuracy: ', nltk.classify.accuracy(self.clf, self.train_set))
    print('Test Accuracy: ', nltk.classify.accuracy(self.clf, self.test_set))

  def informative_features(self, n = 10):
    print('Most Informative features: ', self.clf.show_most_informative_features(n))


if __name__ == '__main__':
  app = GenderApp()
  app.accuracy()
  app.informative_features
  app.classify('Mohit')
