import numpy as np
from pandas.core.frame import DataFrame
from helper import getOriginalVocabulary
from math import log10
from tqdm import tqdm
import pandas as pd


class nb_classifier:
    def __init__(self, smoothing=0.01, log="log"):
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
        self.train_set = data_X.set_index('tid').join(data_Y.set_index('tid'))

        # store the conditional probabilities
        self.cond = pd.DataFrame(index=self.classes, columns=self.vocab)
        # store the conditional probabilities with smoothing
        self.cond_smooth = pd.DataFrame(index=self.classes, columns=self.vocab)
        # store all the (prior probability, total words, totals docs ) by class
        self.prior = pd.DataFrame(index=self.classes, columns=[
                                  "prior", "total_words", "total_doc"])

        # total document in whole training set
        total_doc = len(self.train_set.index)
        print('Vocabulary size: '+ str(len(self.vocab)))
        print('Total documents: '+str(total_doc))

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
                self.train_set[self.train_set[class_name] == clas].iloc[:, :-1].sum()).sum()
            # prior probabilty for class
            self.prior.loc[clas, 'prior'] = self.prior.loc[clas,
                                                           'total_doc']/total_doc

            vocab_size = len(self.vocab)
            # iterate over all word in vocabulary
            # tqdm used for progress bar
            print(clas+" in progress")
            for v in tqdm(self.vocab.tolist()):
                # p(word|class) = frequency of word in class/ total number of words
                # with smoothing = frequency of word in class + smoothing/ total number of words + smoothing(vocab.size)
                freq_word = self.train_set[self.train_set['q1'] == clas][v].sum()
                self.cond_smooth.loc[clas, v] = (freq_word + self.smoothing)/(self.prior.loc[clas, 'total_words'] + vocab_size* self.smoothing)
                self.cond.loc[clas, v] = freq_word/self.prior.loc[clas, 'total_words']
                
                
        print("prior probability")
        print(self.prior)
        print("conditional probability with no smoothing")
        print(self.cond)
        print("conditional probability with smoothing")
        print(self.cond_smooth)
      

    def predict(self, data_X: DataFrame, data_Y: DataFrame, class_name: str):

        to_format = data_X

        # remove test words that are not in vocab
        # format data
        common_cols = self.train_set.columns.intersection(to_format.columns)
        cols_to_add = self.train_set.iloc[:, :-
                                          1].columns.difference(to_format.columns)
        cols_to_remove = (
            to_format.iloc[:, :-1].columns.difference(self.train_set.columns)).tolist()

        predict_data = to_format.drop(columns=cols_to_remove)
        # predict_data[cols_to_add.tolist()] = 0
        predict_data = predict_data.reindex(
            sorted(predict_data.columns), axis=1).set_index('tid')
        print(predict_data)

        output = pd.DataFrame(index=predict_data.index, columns=[
                              "tid","prediction", "score"])

        print("Predicting")
        tweets = predict_data.index

        progressCounter = 0
        # iterate over all the tweets to predict
        for tweet in tweets:
            final_score = -1000000
            prediction = ''
            # print(tweet)
            for clas in self.classes:
                # print('\t'+clas)
                score = 0
                score = log10(self.prior.loc[clas, 'prior'])
                # iterate over vocabulary
                for word in predict_data.columns.tolist():
                    # check if word frequency > 0 per tweet
                    if predict_data.loc[tweet, word] > 0:
                        # compute conditional probability with frequenct
                        # score = freq of word * log(conditional probability)
                        score = score + \
                            (predict_data.loc[tweet, word] *
                             log10(self.cond_smooth.loc[clas, word]))
                        # print(score)
                # max score, and final prediction
                # print(score)
                if score > final_score:
                    final_score = score
                    prediction = clas
            # print('\t\t'+str(final_score))
            output.loc[tweet, 'score'] = "{:e}".format(final_score)
            output.loc[tweet, 'prediction'] = prediction


        # add correct answer to output
        output = output.join(data_Y.set_index('tid'))
        # add correctness
        output.loc[output['prediction'] == output[class_name], 'correctness'] = 'correct'
        output.loc[output['prediction'] != output[class_name], 'correctness'] = 'wrong'
        output['tid'] = output.index
        print(output)
        self.output = output
        getAccuracy(output)
        getEvaluationMetrics(output)
        return output

    def score(self):
        a = ""

def getAccuracy(output):
    sumOfCorrect = 0
    sumOfWrong = 0
    for i in output.correctness:
        if(i == 'correct'):
            sumOfCorrect = sumOfCorrect + 1
        else:
            sumOfWrong = sumOfWrong + 1

    accuracy = sumOfCorrect / len(output.correctness)
    #print("Accuracy: " + str(accuracy))
    return accuracy

def getEvaluationMetrics(output):
    matrixTable = [[0, 0], [0, 0]]
    
    for i, j in zip(output.prediction, output.q1):
        if(i == 'yes' and j == 'yes'):
            matrixTable[0][0] = matrixTable[0][0] + 1
        elif(i == 'yes' and j == 'no'):
            matrixTable[0][1] = matrixTable[0][1] + 1
        elif(i == 'no' and j == 'yes'):
            matrixTable[1][0] = matrixTable[1][0] + 1
        elif(i == 'no' and j == 'no'):
            matrixTable[1][1] = matrixTable[1][1] + 1

    TPOfYes = matrixTable[0][0]
    FPOfYes = matrixTable[0][1]
    FNOfYes = matrixTable[1][0]

    TPOfNo = matrixTable[1][1]
    FPOfNo = matrixTable[1][0]
    FNOfNo = matrixTable[0][1]

    #print(matrixTable)

    PrecisionOfYes = TPOfYes / (TPOfYes + FPOfYes)
    RecallOfYes = TPOfYes / (TPOfYes + FNOfYes)
    F1OfYes = 2 * PrecisionOfYes * RecallOfYes / (PrecisionOfYes + RecallOfYes)

    PrecisionOfNo = TPOfNo / (TPOfNo + FPOfNo)
    RecallOfNo = TPOfNo / (TPOfNo + FNOfNo)
    F1OfNo = PrecisionOfNo * RecallOfNo / (PrecisionOfNo + RecallOfNo)

    filename = "output/eval_NB-BOW-OV.txt"
    f = open(filename, "w")

    f.write(str(round(getAccuracy(output), 4)) + '\n')
    f.write(str(round(PrecisionOfYes, 4)) + " " + str(round(PrecisionOfNo, 4)) + '\n')
    f.write(str(round(RecallOfYes, 4)) + " " + str(round(RecallOfNo, 4)) + '\n')
    f.write(str(round(F1OfYes, 4)) + " " + str(round(F1OfNo, 4)) + '\n')
    f.close()

def main():
    train_data = getOriginalVocabulary('./data/sample1.tsv')
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -2:]

    test_data = getOriginalVocabulary('./data/covid_test_public.tsv')
    test_data_X = test_data.iloc[:, :-1]
    test_data_Y = test_data.iloc[:, -2:]

    nb_class = nb_classifier(0.01, 'log')
    nb_class.train(train_data_X, train_data_Y, "q1")

    ov_output = nb_class.predict(test_data_X, test_data_Y, "q1")

    filename = './output/trace_NB-BOW-OV.txt'
    # with open(filename,'w') as outfile:
    #     ov_output.to_string(outfile, index=False, header=False)

    # output to file
    np.savetxt(filename, ov_output.values, fmt='%s  %s  %s  %s  %s')


main()
