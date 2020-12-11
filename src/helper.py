import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tweet import Tweet


def parseOV(inputFile):
    v = ""

def parseFV(inputFile):
    v = ""

# parse tsv file into a list of vector array
def parse_csv(input):
    dataset = pd.read_csv(input, sep='\t')
    text_list = dataset.iloc[:, 1]

    # use CountVectorizer library to format text and convert into vector
    vectorizer = CountVectorizer()
    textVector = vectorizer.fit_transform(text_list);


    data = pd.DataFrame(textVector.toarray(), columns=vectorizer.get_feature_names())
    data["tid"] = text_list = dataset.iloc[:, 0]
    data["q1"] = text_list = dataset.iloc[:, 2]

    print(text_list)
    print(vectorizer.get_feature_names())
    print(textVector.toarray())
    print(data)

    # last 2 columns contains the vector info
    return data



# use manually to format text and convert into vector
# we dont use it
def old_method_parsing_data():
    dataset = pd.read_csv(input, sep='\t')
    dataset_tweets = dataset.iloc[:, :3]

    # Task 1 - First fold the training set in lower case
    dataset_tweets["text"] = dataset_tweets["text"].str.lower()
    print(dataset_tweets)

    tweets_list = []
    vocabulary_list =[]

    # Task 2 - Build a list of all words appearing in the training set. This list is vocabulary
    for item in dataset_tweets.values:
        tweets_list.append(Tweet(item[0], item[1], item[2]))
        # contains duplicate, might have to remove duplicate
        vocabulary_list.extend(item[1].split())
    return [tweets_list, vocabulary_list]


def outputFile():
    a =""
# return list of vocabulary and list of tweets
# word count
# parse_csv('./data/covid_training.tsv')
# print(parse_all_csv('./data/sample.tsv'))

parse_csv('./data/sample.tsv')
