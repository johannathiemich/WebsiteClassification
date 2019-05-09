#By Johanna Thiemich
#Training an SVM on our data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

csv_path_cleaned = 'files/data_cleaned.txt'

df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
X = df['Text']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#linear classifier (SVM)

#this is the pipeline transforming the input data
#CountVectorizer and TfidfTransformer are used for feature extraction
#The SGDClassifier class creates the SVM model
sgd_linear = Pipeline((('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf',
                        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=100, tol=1e-3
                        ))))

#train the model
sgd_linear.fit(X_train, y_train)

y_pred_linear = sgd_linear.predict(X_test)

print('accuracy linear model %s' % accuracy_score(y_pred_linear, y_test))

print("report linear model", classification_report(y_test, y_pred_linear,target_names=sgd_linear.classes_))
