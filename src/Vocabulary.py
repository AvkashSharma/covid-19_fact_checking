class Vocabulary:

  def __init__(self, word_list, classifier):
    self.word_list = word_list
    self.classifier = classifier
    self.get_tweet()


  def get_tweet(self):
    return list(zip(self.word_list,self.classifier))


lst = Vocabulary({("i",1),("w",4)},"yes")
print(lst.get_tweet())
