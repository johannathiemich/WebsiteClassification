from pandas import read_csv
from preprocess import *
from numpy import nan
import csv
import fasttext

df = read_csv('data2.txt', sep='\t')

clean_text = [None] * len(df)
for i, text in enumerate(df.Text):
    try:
        clean_text[i] = normalize(text)
    except:
        print(i)
        clean_text[i] = None

clean_text_string = [' '.join(text) if text else nan for text in clean_text]


df['clean_text'] = clean_text_string

df2 = df.dropna()

df2.Category = df2.Category.str.replace(" ","")

df2['Category2'] = '__label__' + df2['Category']

df2.Category2 = df2.Category2.str.strip()

df2[['Category2', 'clean_text']].to_csv('cleandata.txt', sep=' ')