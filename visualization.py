from sklearn.cluster import KMeans
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import spacy
import matplotlib.pylab as pl

csv_path_cleaned = 'files/data_cleaned.txt'
df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
X = df['Text']
y = df['Category']

# path_sent_embeddings = "spacy_sent_embeddings_all_data.csv"
# sent_embeddings = []
# with open(path_sent_embeddings) as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=' ')
#    for row in readCSV:
#
#        sent_embeddings.append([row])

# sentence_embeddings = []
# for row in X:
#     sentence_embeddings.append(nlp(row).vector)

# According to their documentations, this will work faster for large size corpus

def get_embeddings(path='doc_embeddings.npy'):
    return np.load(path)

def generate_embeddings():
    nlp = spacy.load('en_core_web_lg')
    docs = list(nlp.pipe(X))
    return [doc.vector for doc in docs]


sentence_embeddings = get_embeddings()
sentence_embeddings = sentence_embeddings[:10]



#print("X shape: ", X.shape[0])
pca = PCA(n_components=10, svd_solver='auto')
X = np.array(sentence_embeddings)
#dimensionality reduction
pca.fit_transform(X)

#clustering
#kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(X)
#print(kmeans)

X = X[:300]
for i in range(X.shape[0]):
    if y[i] == '__label__sports':
        c1 = pl.scatter(X[i,0], X[i,1], c='r', marker='+')
    elif y[i] == '__label__food&drink':
        c2 = pl.scatter(X[i, 0], X[i, 1], c='yellow', marker='*')
    elif y[i] == '__label__travel':
        c3 = pl.scatter(X[i, 0], X[i, 1], c='b', marker='>')
    elif y[i] == '__label__games&toys':
        c4 = pl.scatter(X[i,0], X[i, 1], c='green', marker='o')

pl.legend([c1, c2, c3, c4], ['Sports', 'Food&Drink',
    'Travel', 'Games&Toys'])
pl.title('Website classification embeddings visualization')
pl.show()
#visualisation



