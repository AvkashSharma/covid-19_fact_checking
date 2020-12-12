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

    def train(self, data_X: DataFrame, data_Y: DataFrame):

        # TODO make sure data_X.count == data_Y.count()

        self.classes = data_Y.iloc[:, -1].drop_duplicates().values.tolist()
        self.vocab = data_X.iloc[:-1, :-1].columns.values
        self.documents = data_X.set_index('tid').join(data_Y.set_index('tid'))

                # store the conditional probabilities
        self.cond  = pd.DataFrame(index =self.classes, columns=self.vocab)
        # store all the (prior probability, total words, totals docs ) by class
        self.prior = pd.DataFrame(index=self.classes, columns=["prior", "total_words", "total_doc"])
        print(self.prior)
        
        total_yes_doc = data_Y[data_Y == "yes"].count()[1]
        total_no_doc = data_Y[data_Y == "no"].count()[1]
        total_doc = len(self.documents.index)

        # prior probabilities
        # propbabilities of each class
        # TODO can later change to detect the classes and use a vector
        prob_yes = total_yes_doc/total_doc
        prob_no = total_no_doc/total_doc

        print("total doc: "+str(total_doc))
        print("total YES doc: "+str(total_yes_doc) + "\tprob yes: "+str(prob_yes))
        print("total NO doc: "+str(total_no_doc)+"\tprob no: "+str(prob_no))
        print("total Vocab: "+ str(len(self.vocab)))
        
        # conditional probabilities (likely hood)
        # probability of each word given a class
        # count frequency of a word within a class/total number of word in a class



        # itereate over all the classes
        for clas in self.classes:
            # total words for class
            total_class_word =  (self.documents[self.documents['q1']==clas].iloc[:,:-1].sum()).sum()
            print('Total words '+clas+": "+str(total_class_word))

            # iterate over all word in vocabulary
            for v in self.vocab:
                # p(word|class) = frequency of word in class/ total number of words
                self.cond.loc[clas,v] = self.documents[self.documents['q1']==clas][v].sum()/total_class_word

        print(self.cond)

    # for all classes ci
        #   compute p(ci) = count()

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
    nb_class.train(train_data_X, train_data_Y)


main()
