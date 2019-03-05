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
from numpy import nan

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
    #dataFrame = dataFrame[:10]
    #for index, row in dataFrame.iterrows():
    #    copydf = dataFrame['Category']
    #    dataFrame['Category'] = '__label__' + copydf
    dataFrame[['Category', 'Text']].to_csv(csv_path, sep=' ', index=False, header=False)

df = read_csv(csv_path, header = None, names = ['Category', 'Text'], sep =' ')

#JT:
#removing rows with empty text
df = df[df['Text'] != ""]
size_before = df.count()[0]

#removing stopwords
stop_words = set(stopwords.words('english'))
delete_indices = []
clean_text = [None] * len(df)

for i, text in enumerate(df.Text):
#for index, row in df.iterrows():
    #wanted to use TextBlob but it won't work (HTTP request error)
    #b = TextBlob(row['Text'])
    try:
        lang = detect(text)
    except:
        #No language could be detected --> delete row
        lang = ''
        delete_indices.append(i)
    #making sure to only take the english websites
    if lang == 'en':
        try:
            #row['Text'] = preprocess.normalize_opt(row['Text'])
            clean_text[i] = preprocess.normalize_opt(text)
            print("row: ", i)
        except:
            delete_indices.append(i)

        #word_tokens = word_tokenize(row['Text'])
        #filtered_sentence = [w for w in word_tokens if not w in stop_words]
        #row['Text'] = filtered_sentence
        #filtered_sentence = []

clean_text_string = [' '.join(text) if text else nan for text in clean_text]
df['clean_text'] = clean_text_string
df = df.dropna()
df.Category = df.Category.str.replace(" ","")
df['Category'] = '__label__' + df['Category']
df.Category = df.Category.str.strip()

if len(delete_indices) > 0:
    delete_indices_upd = [i for i in delete_indices if i < df.count()[0]]
    df.drop(df.index[delete_indices_upd], inplace = True)

size_after = df.count()[0]
print("size before: ", size_before)
print("size after: ", size_after)

#df.to_csv(csv_path_cleaned, header = None, sep =' ')
df[['Category', 'clean_text']].to_csv(csv_path_cleaned, sep=' ', index=False, header=False)

train, test = train_test_split(df, test_size=0.3)
train[['Category', 'clean_text']].to_csv(csv_path_train, sep=' ', index=False, header=False)
test[['Category', 'clean_text']].to_csv(csv_path_test, sep=' ', index=False, header=False)
