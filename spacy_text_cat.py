#By Alejandro Robles
#Training spacy's CNN model from scratch for text classification


import pandas as pd
import spacy
from spacy.util import minibatch, compounding
import random
from numpy import random


def get_text_cats(df):
    df_one_hot = pd.get_dummies(df, columns=['Category'], prefix='', prefix_sep='', dtype='bool')
    cats = df_one_hot.loc[:, df_one_hot.columns != 'Text']
    cats = cats.to_dict('records')
    X = df_one_hot.loc[:, 'Text']
    y = cats
    return X, y

def get_text_label(df):
    return df['Text'], df['Category']

def save_model(nlp, optimizer, output_dir = 'savedmodel'):
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to ", output_dir)

def load_model(output_dir):
    print("Loading from ", output_dir)
    nlp = spacy.load(output_dir)
    return nlp


def evaluate(tokenizer, textcat, texts, cats):
    import operator
    docs = (tokenizer(text) for text in texts)
    acc = 0
    good_indices = [False] * len(texts)
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        prediction = max(doc.cats.items(), key=operator.itemgetter(1))[0]
        if gold == prediction:
            acc += 1
            good_indices[i] = True
    return acc / len(texts), good_indices

def split(df, percentage= 0.8):
    msk = random.rand(len(df)) < percentage
    traindf = df[msk]
    testdf = df[~msk]
    return traindf, testdf

def set_pipe(starting_model='en_core_web_md'):
    nlp = spacy.load(starting_model)
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
    return nlp, textcat

def train_model(nlp, n_iter=10, save=False, output_dir=None):
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

    if save:
        save_model(nlp, optimizer, output_dir)
    return nlp


if __name__ == '__main__':

    # Load in data
    csv_path_cleaned = 'files/data_cleaned.txt'
    df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
    train_df, test_df = split(df)
    X_train, y_train = get_text_cats(train_df)
    X_test, y_test = get_text_label(test_df)
    train_data = list(zip(X_train, [{"cats": cat} for cat in y_train]))

    # Set up pipelines
    nlp, textcat = set_pipe()
    n_iter = 10

    # Train Model
    save_my_model = False
    output_dir = 'savedmodel_directory'
    nlp = train_model(nlp, n_iter, save=save_my_model, output_dir=output_dir)

    # Test on one example
    test_doc = X_test.values[0]
    doc = nlp(test_doc)
    print(test_doc, doc.cats)

    # Get Accuracy
    textcat = nlp.get_pipe('textcat')
    accuracy = evaluate(nlp.tokenizer, textcat, X_test, y_test)
    print('Accuracy: {accuracy}'.format(accuracy=accuracy))

    # Load Model
    load_my_model = True
    if load_my_model:
        nlp2 = load_model(output_dir)

