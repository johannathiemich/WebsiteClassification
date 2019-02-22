#this script only needs to be run once in the beginning to create the .pkl files

from pandas import DataFrame, read_csv
from json import load, loads
#from textblob import TextBlob
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect
import nltk
from sklearn.model_selection import train_test_split
import preprocess

#only un-comment these for first use (need to download these!):
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data = []
json_path = 'files/scrapedsites.json'
csv_path = 'files/data.txt'
csv_path_cleaned = 'files/data_cleaned.txt'
csv_path_train = 'files/data_cleaned_train.txt'
csv_path_test = 'files/data_cleaned_test.txt'

#AJ:
with open(json_path) as f:
    for line in f:
        d = loads(line)
        data.append(d)

    dataFrame = DataFrame(data)
    for index, row in dataFrame.iterrows():
        dataFrame['Category'] = '__label__' + dataFrame['Category']
        print("modifying label")
    dataFrame[['Category', 'Text']].to_csv(csv_path, sep=' ', index=False, header=False)

df = read_csv(csv_path, header = None, names = ['Category', 'Text'], sep =' ')

#JT:
#removing rows with empty text
df = df[df['Text'] != ""]
size_before = df.count()[0]

#df = df[:10]
#removing stopwords
stop_words = set(stopwords.words('english'))
delete_indices = []

for index, row in df.iterrows():
    #wanted to use TextBlob but it won't work (HTTP request error)
    #b = TextBlob(row['Text'])
    try:
        lang = detect(row['Text'])
    except:
        #No language could be detected --> delete row
        lang = ''
        delete_indices.append(index)
    #making sure to only take the english websites
    if lang == 'en':
        try:
            row['Text'] = preprocess.normalize_opt(row['Text'])
        except:
            delete_indices.append(index)

        #word_tokens = word_tokenize(row['Text'])
        #filtered_sentence = [w for w in word_tokens if not w in stop_words]
        #row['Text'] = filtered_sentence
        #filtered_sentence = []

if len(delete_indices) > 0:
    delete_indices_upd = [i for i in delete_indices if i < df.count()[0]]
    df.drop(df.index[delete_indices_upd], inplace = True)

size_after = df.count()[0]
print("size before: ", size_before)
print("size after: ", size_after)

df.to_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
train, test = train_test_split(df, test_size=0.3)
train.to_csv(csv_path_train, header = None, names = ['Category', 'Text'], sep =' ')
test.to_csv(csv_path_test, header = None, names = ['Category', 'Text'], sep =' ')
