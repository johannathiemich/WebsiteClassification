#this script only needs to be run once in the beginning to create the .pkl files

from pandas import DataFrame
from json import load, loads
from pandas import read_pickle
#from textblob import TextBlob
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect
import numpy as np
import nltk
from sklearn.model_selection import train_test_split

#only un-comment these for first use (need to download these!):
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data = []
json_path = 'files/scrapedsites.json'
pkl_path = 'files/data.pkl'
pkl_path_cleaned = 'files/data_cleaned.pkl'
pkl_path_train = 'files/data_cleaned_train.pkl'
pkl_path_test = 'files/data_cleaned_test.pkl'

#AJ:
with open(json_path) as f:
    for line in f:
        d = loads(line)
        data.append(d)
    dataFrame = DataFrame(data)
    dataFrame.to_pickle(pkl_path)
    #not working:
    #np.savetxt(r'C:\Users\johan\OneDrive\Dokumente\Auslandssemester USA\Kurse\Machine Learning\project\WebsiteClassification\np.txt', dataFrame.values, fmt='%d')

df = read_pickle(pkl_path)
print("printing head:")
print(df.head)
#JT:
#removing rows with empty text
df = df[df['Text'] != ""]
size_before = df.count

#df = df[:5]
#removing stopwords
stop_words = set(stopwords.words('english'))
for index, row in df.iterrows():
    #wanted to use TextBlob but it won't work (HTTP request error)
    #b = TextBlob(row['Text'])
    try:
        lang = detect(row['Text'])
    except:
        #No language could be detected --> delete row
        lang = ''
        df.drop(df.index[index], inplace = True)
        index = index - 1
    #making sure to only take the english websites
    if lang == 'en':
        word_tokens = word_tokenize(row['Text'])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        row['Text'] = filtered_sentence
        filtered_sentence = []

#    for w in word_tokens:
#        if w not in stop_words:
#            filtered_sentence.append(w)

size_after = df.count
print("size before: ", size_before)
print("size after: ", size_after)
#print(df["Text"])
df.to_pickle(pkl_path_cleaned)
train, test = train_test_split(df, test_size=0.3)
train.to_pickle(pkl_path_train)
test.to_pickle(pkl_path_test)
