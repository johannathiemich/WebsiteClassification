#By Alejandro Robles
#Playing with spacy modules

from spacy_text_cat import *
from numpy import logical_not

# Load in data
csv_path_cleaned = 'files/data_cleaned.txt'
df = pd.read_csv(csv_path_cleaned, header=None, names=['Category', 'Text'], sep=' ')
train_df, test_df = split(df)
X_train, y_train = get_text_cats(train_df)
X_test, y_test = get_text_label(test_df)
train_data = list(zip(X_train, [{"cats": cat} for cat in y_train]))

# Load Model
load_my_model = True
output_dir='savedmodel'
if load_my_model:
    nlp = load_model(output_dir)

# Get Accuracy
accuracy = evaluate(nlp.tokenizer, nlp.get_pipe('textcat'), X_test.values, y_test.values)
print('Accuracy: {accuracy}'.format(accuracy=accuracy[0]))


y_test = y_test.str.replace("__label__", "")

good_ones = y_test[accuracy[1]]
bad_ones = y_test[logical_not(accuracy[1])]
metric = good_ones.value_counts() / y_test.value_counts()