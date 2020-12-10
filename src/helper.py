import pandas as pd
import numpy as np



dataset = pd.read_csv('../data/covid_training.tsv',sep = '\t')
dataset_tweets  = dataset.iloc[:,1]
#print(type(dataset_tweets)) 

data_lowercase = []


# Task 1 - First fold the training set in lower case
for item in dataset_tweets:
  data_lowercase.append(item.lower())

#print(len(data_lowercase))

#Task 2 - Build a list of all words appearing in the training set. This list is vocabulary 
vocabulary_list = []

for item in data_lowercase:
  vocabulary_list.append(item.split())


# for item in vocabulary_list:
#   print(item)

# word count



