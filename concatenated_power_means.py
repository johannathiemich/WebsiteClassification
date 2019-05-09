#An approach that we did not end up using. Just leaving it in here for future work
import tensorflow_hub as hub
import pandas as pd

csv_path_cleaned = 'files/data_cleaned.txt'
df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
X = df['Text']
y = df['Category']

url_monolingual = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/monolingual/1'
embed = hub.Module(url_monolingual)
print("here")
representations = embed(["A_en long_en sentence_en ._en", "another_en sentence_en"])
representations = embed(X[0])
print(representations)
print("finished")
