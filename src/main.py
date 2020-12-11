
# Multinomial Naive Bayes Classifier will have three parameters
# Vocabulary,Smoothing, Log
from helper import parse_all_csv

# parse data, get the tweets and all the vocab
tweets, OV = parse_all_csv('./data/sample.tsv')

print(OV)

