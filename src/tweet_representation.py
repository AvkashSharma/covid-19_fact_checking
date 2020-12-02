class Representation:

  def __init__(self, sentence, classifier):
    self.sentence = sentence
    self.classifier = classifier
    self.tweet_repr()


  def word_to_freq(self,word_list):
    word_freq = [word_list.count(p) for p in word_list]
    return dict(list(zip(word_list,word_freq)))


  # Task is to return tweet in the form <(actual word, frequency), classifier=yes/no>
  def tweet_repr(self):
    word_list = self.sentence.lower().split()
    dict_tweet = self.word_to_freq(word_list)
    print(dict_tweet)
    
      


Representation("I am the king","yes")