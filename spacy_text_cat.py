#By Alejandro Robles
#Training an CNN for text classification

import pandas as pd
import spacy
from spacy.util import minibatch, compounding
import random


csv_path_cleaned = 'files/data_cleaned.txt'
df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
X = df['Text']
y = df['Category']
nlp = spacy.load('en_core_web_md')

# X = X[:10]
# docs = list(nlp.pipe(X))
# sentence_embeddings = [doc.vector for doc in docs]



# add the text classifier to the pipeline if it doesn't exist
# nlp.create_pipe works for built-ins that are registered with spaCy
if "textcat" not in nlp.pipe_names:
    textcat = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "simple_cnn",
        }
    )
    nlp.add_pipe(textcat, last=True)
# otherwise, get it, so we can add labels to it
else:
    textcat = nlp.get_pipe("textcat")

# add label to text classifier
textcat.add_label("__label__sports")
textcat.add_label("__label__travel")
textcat.add_label("__label__games&toys")
textcat.add_label("__label__food&drink")



df2 = pd.get_dummies(df, columns=['Category'], prefix='', prefix_sep='', dtype='bool')
cats = df2.loc[:, df2.columns != 'Text']
cats = cats.to_dict('records')

n_iter = 10
X = df2.loc[:, 'Text']

train_data = list(zip(X, [{"cats": cats} for cats in cats]))


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
    batch_sizes = compounding(4.0, 32.0, 1.001)
    for i in range(n_iter):
        print(i)
        losses = {}
        # batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)


    test_text = X[1]
    doc = nlp(test_text)
    print(test_text, doc.cats)


def evaluate(tokenizer, textcat, texts, cats):
    import operator
    docs = (tokenizer(text) for text in texts)
    acc = 0
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        prediction = max(doc.cats.items(), key=operator.itemgetter(1))[0]
        if gold == prediction:
            acc += 1
    return acc / len(texts)

scores = evaluate(nlp.tokenizer, textcat, X, y)


