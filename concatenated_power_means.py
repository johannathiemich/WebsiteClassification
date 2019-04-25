import tensorflow as tf
import tensorflow_hub as hub

csv_path_train = 'files/data_cleaned_train.txt'
csv_path_test = 'files/data_cleaned_test.txt'
url_de = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/en-de/1'
url_monolingual = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/monolingual/1'
embed = hub.Module(url_monolingual)
print("here")
representations = embed(["A_en long_en sentence_en ._en", "another_en sentence_en"])
print("finished")
