#this script only needs to be run once in the beginning to create the .pkl files

from pandas import DataFrame
from json import load, loads
from pandas import read_pickle
import numpy as np
import nltk
#only un-comment these for first use (need to download these!):
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data = []
json_path = 'scrapedsites.json'
pkl_path = 'data.pkl'

#AJ:
with open(json_path) as f:
    for line in f:
        d = loads(line)
        data.append(d)
    dataFrame = DataFrame(data)
    dataFrame.to_pickle(pkl_path)
    np.savetxt(r'C:\Users\johan\OneDrive\Dokumente\Auslandssemester USA\Kurse\Machine Learning\project\WebsiteClassification\np.txt', dataFrame.values, fmt='%d')

df = read_pickle(pkl_path)
print("printing head:")
print(df.head)
#JT:
#removing rows with empty text
df = df[df['Text'] != ""]
df = df[:5]
print(df)
#removing stopwords
stop_words = set(stopwords.words('english'))
for index, row in df.iterrows():
    word_tokens = word_tokenize(row['Text'])
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    #for testing only
    #print(word_tokens)
    #print(filtered_sentence)

print(df)
df.to_pickle("data_cleaned.pkl")
#df.groupby('Category').count()

print(len(df[df['Text'] == ""]))
