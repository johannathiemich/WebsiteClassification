#this script only needs to be run once in the beginning to create the .pkl files

from pandas import DataFrame
from json import load, loads
from pandas import read_pickle
import nltk
#only un-comment these for first use (need to download these!):
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data = []
path = 'scrapedsites.json'

#AJ:
with open(path) as f:
    for line in f:
        d = loads(line)
        data.append(d)
    dataFrame = DataFrame(data)

df = read_pickle('data.pkl')

#JT:
#removing rows with empty text
df = df[df['Text'] != ""]

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

df.to_pickle("./data_cleaned.pkl")
df.groupby('Category').count()

print(len(df[df['Text'] == ""]))
