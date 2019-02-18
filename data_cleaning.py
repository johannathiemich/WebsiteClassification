#this script only needs to be run once in the beginning to create the .pkl files

from pandas import DataFrame
from json import load, loads
from pandas import read_pickle
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
#df = df[:5]
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


#still need to correct this: save the filtered sentence, not the original df
df.to_pickle(pkl_path_cleaned)
train, test = train_test_split(df, test_size=0.3)
train.to_pickle(pkl_path_train)
test.to_pickle(pkl_path_test)
