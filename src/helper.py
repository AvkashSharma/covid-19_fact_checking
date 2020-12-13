import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def removeUnnecessaryChars(listToRemoveCharsFrom):
    for i in range(len(listToRemoveCharsFrom)):
        listToRemoveCharsFrom[i] = listToRemoveCharsFrom[i].replace(",","")
        listToRemoveCharsFrom[i] = listToRemoveCharsFrom[i].replace(".","")
        listToRemoveCharsFrom[i] = listToRemoveCharsFrom[i].replace("?","")
        listToRemoveCharsFrom[i] = listToRemoveCharsFrom[i].replace("!","")

    return listToRemoveCharsFrom


def getOriginalVocabulary(input):
    dataset = pd.read_csv(input, sep='\t')
    listOfRowsOfTweets = dataset.iloc[:, 1].str.lower()
    #print(listOfRowsOfTweets)

    listOfWords = []
    
    for i in listOfRowsOfTweets:
        lists = i.split()
        lists = removeUnnecessaryChars(lists)
        for j in lists:
            if(wordExistsInList(listOfWords, j) < 0):
                listOfWords.append(j)
        
    #print(listOfWords)

    listOfRowsOfValues = []

    for i in listOfRowsOfTweets:
        lists = i.split()
        lists = removeUnnecessaryChars(lists)
        listOfValues = [0] * len(listOfWords)
        for j in lists:
            if(wordExistsInList(listOfWords, j) >= 0):
                index = wordExistsInList(listOfWords, j)
                listOfValues[index] = listOfValues[index] + 1
        listOfRowsOfValues.append(listOfValues)

    #print(listOfRowsOfValues)

    listOfVocabulary = []
    listOfVocabulary.append(listOfWords)
    listOfVocabulary.append(listOfRowsOfValues)

    data = pd.DataFrame(listOfRowsOfValues,
                        columns=listOfWords)
    data["tid"] = dataset.iloc[:, 0]
    data["q1"] = dataset.iloc[:, 2]

    print(data)

    return data

def parseKarthiAndSharmaWay(input):
    dataset = pd.read_csv(input, sep='\t')
    text_list = dataset.iloc[:, 1]

    vectorizer = CountVectorizer()
    textVector = vectorizer.fit_transform(text_list)

    data = pd.DataFrame(textVector.toarray(),
                        columns=vectorizer.get_feature_names())
    data["tid"] = text_list = dataset.iloc[:, 0]
    data["q1"] = text_list = dataset.iloc[:, 2]

    print(data)

    return data

def wordExistsInList(listOfWordsToCheck, word):
    for i in range(len(listOfWordsToCheck)):
        if(listOfWordsToCheck[i] == word):
            return i
    return -1


def main():
    parseKarthiAndSharmaWay('./data/sample.tsv')
    dataList = getOriginalVocabulary('./data/sample.tsv')

main()
