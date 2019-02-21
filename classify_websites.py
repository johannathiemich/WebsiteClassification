#this file will contain training the model and testing its results
from pandas import read_pickle
import fasttext

df = read_pickle('files/data_cleaned_train.pkl')

model = fasttext.skipgram('test.txt', 'model')


df[['Category', 'Text']].to_csv('data.txt', sep='\t', index=False, header=True)

model = fasttext.skipgram('data.txt', 'model')


classifier = fasttext.supervised('data.txt', 'model2')


df[['Category', 'Text']].to_csv('data2.txt', sep='\t', index=False, header=True)



