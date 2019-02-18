#this file will contain training the model and testing its results
from pandas import read_pickle
import fasttext
import pickle

df = read_pickle('data.pkl')

model = fasttext.skipgram('test.txt', 'model')