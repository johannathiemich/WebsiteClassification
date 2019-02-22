#this file will contain training the model and testing its results
from pandas import read_pickle
import fasttext

csv_path_train = 'files/data_cleaned_train.txt'
csv_path_test = 'files/data_cleaned_test.txt'

model = fasttext.skipgram(csv_path_train, 'model')

classifier = fasttext.supervised(csv_path_train, 'model2')
