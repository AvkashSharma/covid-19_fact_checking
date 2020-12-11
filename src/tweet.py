class Tweet:
  represent = []
  def __init__(self, tid, tweet, classifier):
    self.tid = tid
    self.tweet = tweet
    self.classifier = classifier
    self.representation()
    # print(self.represent)


  def word_to_freq(self, word_list):
    """
    generate dictionary with frequency and their words
    """
    word_freq = [word_list.count(p) for p in word_list]
    return dict(list(zip(word_list,word_freq)))


  # Task is to return tweet in the form <(actual word, frequency), classifier=yes/no>
  def representation(self):
    word_list = self.tweet.lower().split()
    self.represent = self.word_to_freq(word_list)
    



# t = Tweet(1234, "this is fucking bullshit this is","yes")
# print(t.represent)