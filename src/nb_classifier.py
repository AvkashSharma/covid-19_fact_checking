from pandas.core.frame import DataFrame
from helper import parseOV
from math import log10
import pandas as pd


class nb_classifier:
    def __init__(self, vocabulary, smoothing, log):
        self.vocabulatry = vocabulary
        # use 0.01
        self.smoothing = smoothing
        # use log10
        self.log = log

    def train(self, data_X: DataFrame, data_Y: DataFrame, class_name: str):
        """
        Train the model

        """
        # TODO make sure data_X.count == data_Y.count()

        # All classes i.e: yes/no
        self.classes = data_Y.iloc[:, -1].drop_duplicates().values.tolist()
        # list of vocabulary according to data_X
        self.vocab = data_X.iloc[:-1, :-1].columns.values
        # list of all the document
        self.documents = data_X.set_index('tid').join(data_Y.set_index('tid'))

        # store the conditional probabilities
        self.cond = pd.DataFrame(index=self.classes, columns=self.vocab)
        # store all the (prior probability, total words, totals docs ) by class
        self.prior = pd.DataFrame(index=self.classes, columns=[
                                  "prior", "total_words", "total_doc"])

        # total document in whole training set
        total_doc = len(self.documents.index)

        # conditional probabilities (likely hood)
        # probability of each word given a class
        # count frequency of a word within a class/total number of word in a class

        # itereate over all the classes
        for clas in self.classes:
            # total documents for class
            self.prior.loc[clas, 'total_doc'] = data_Y[data_Y == clas].count()[
                class_name]
            # total words for class
            self.prior.loc[clas, 'total_words'] = (
                self.documents[self.documents[class_name] == clas].iloc[:, :-1].sum()).sum()
            # prior probabilty for class
            self.prior.loc[clas, 'prior'] = self.prior.loc[clas,
                                                           'total_doc']/total_doc

            # iterate over all word in vocabulary
            for v in self.vocab:
                # p(word|class) = frequency of word in class/ total number of words
                self.cond.loc[clas, v] = self.documents[self.documents['q1'] == clas][v].sum(
                )/self.prior.loc[clas, 'total_words']


        print(self.prior)
        print(self.cond)
      

    def predict(self, data):

        # compute score for yes
        # compute score for no

        a = ""

    def score(self):
        a = ""

    def prob():
        prob = ""


def main():
    train_data = parseOV('./data/sample1.tsv')
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -2:]
    nb_class = nb_classifier(train_data, 0.01, 'log')
    nb_class.train(train_data_X, train_data_Y, "q1")


main()
