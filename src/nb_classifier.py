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

    def train(self, data_X:DataFrame, data_Y:DataFrame):

        print(data_X)
        print(data_Y)
        
        self.classes = data_Y.drop_duplicates()
        self.words = data_X

        print(self.classes)

        # for all classes ci
        #   for all words wj in vocabulary
        #       compute p(wj|ci) = count(wj,ci)/(total words in ci)
        total_yes = data_Y[data_Y =="yes"].count()
        total_no =  data_Y[data_Y =="no"].count()

        for c in self.classes:
            
            print(c)
            # word_prob = 

        # for all classes ci
        #   compute p(ci) = count()

    def predict(self, data):
        a=""

    def score(self):
        a=""
    
    def prob():
        prob =""


def main():
    train_data = parseOV('./data/covid_training.tsv')
    train_data_X = train_data.iloc[:,:-2]
    train_data_Y = train_data.iloc[:,-2]

    nb_class = nb_classifier(train_data, 0.01, 'log')
    nb_class.train(train_data_X, train_data_Y)

main()

