import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import spacy
import matplotlib.pylab as pl
from sklearn.manifold import TSNE
from ggplot import *


def get_embeddings(path='doc_embeddings.npy'):
    return np.load(path)

def generate_embeddings():
    nlp = spacy.load('en_core_web_lg')
    docs = list(nlp.pipe(X))
    return [doc.vector for doc in docs]

def plot_matplot(X, y, df):
    for i in range(X.shape[0]):
        if y[i] == '__label__sports':
            c1 = pl.scatter(df['x-tsne-pca'][i], df['y-tsne-pca'][i], c='r', marker='+')
        elif y[i] == '__label__food&drink':
            c2 = pl.scatter(df['x-tsne-pca'][i], df['y-tsne-pca'][i], c='yellow', marker='*')
        elif y[i] == '__label__travel':
            c3 = pl.scatter(df['x-tsne-pca'][i], df['y-tsne-pca'][i], c='b', marker='>')
        elif y[i] == '__label__games&toys':
            c4 = pl.scatter(df['x-tsne-pca'][i], df['y-tsne-pca'][i], c='green', marker='o')

    pl.legend([c1, c2, c3, c4], ['Sports', 'Food&Drink',
        'Travel', 'Games&Toys'])
    pl.title('Website classification embeddings visualization')
    pl.show()


if __name__ == '__main__':

    csv_path_cleaned = 'files/data_cleaned.txt'
    df = pd.read_csv(csv_path_cleaned, header=None, names=['Category', 'Text'], sep=' ')
    df['label'] = df.Category.str.replace('__label__', '')
    y = df['Category']
    X = get_embeddings()

    pca = PCA(n_components=50, svd_solver='auto')
    pca_result = pca.fit_transform(X)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)


    df['x-tsne-pca'] = tsne_results[:,0]
    df['y-tsne-pca'] = tsne_results[:,1]

    chart = ggplot(df, aes(x='x-tsne-pca', y='y-tsne-pca', color='label')) \
            + geom_point(size=75, alpha=0.8) \
            + ggtitle("Website Classification Embeddings Visualization")

    ggplot.save(chart, 'visual.png')