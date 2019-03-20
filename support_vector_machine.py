import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm

csv_path_cleaned = 'files/data_cleaned.txt'

df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
X = df['Text']
y = df['Category']

grouped_df = df.groupby('Category').count()
print(grouped_df)
'''count_food = grouped_df.
count_games = grouped_df['__label__games&toys']
count_sports = grouped_df['__label__sports']
count_travel = grouped_df['__label__travel']

count_games = count_travel = 1
count_sports = count_sports / count_games
count_food = count_food / count_games'''

count_games = count_travel = 1
count_sports = 2.14
count_food = 0.63

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#fine tune hyperparameters using cross validation
#linear classifier without consideration of unbalanced data
sgd_linear = Pipeline((('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf',
                        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=20, tol=1e-3,
                        ))))

sgd_linear.fit(X_train, y_train)


#linear classifier with considering unbalanced data
sgd_linear_weights = Pipeline((('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf',
                        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=20, tol=1e-3,
                                        class_weight = {'__label__food&drink': count_food,
                                                        '__label__games&toys': count_games,
                                                        '__label__sports': count_sports,
                                                        '__label__travel': count_travel }
                        ))))

sgd_linear_weights.fit(X_train, y_train)

sgd_nonlinear = Pipeline((('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', svm.SVC(C=1.0,
                                       kernel='rbf',
                                       degree=3,
                                       gamma='auto_deprecated',
                                       coef0=0.0,
                                       shrinking=True,
                                       probability=False,
                                       tol=0.001,
                                       cache_size=200,
                                       class_weight = {'__label__food&drink': count_food,
                                                       '__label__games&toys': count_games,
                                                       '__label__sports': count_sports,
                                                       '__label__travel': count_travel },
                                       verbose=False,
                                       max_iter=-1,
                                       decision_function_shape='ovr',
                                       random_state=None))))

'''sgd_nonlinear = svm.SVC(kernel='linear', class_weight = {'__label__food&drink': count_food,
                                                         '__label__games&toys': count_games,
                                                         '__label__sports': count_sports,
                                                         '__label__travel': count_travel
                                                        })'''
sgd_nonlinear.fit(X_train, y_train)

sgd3 = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=20, tol=1e-3)),
               ])
sgd3.fit(X_train, y_train)


y_pred_linear = sgd_linear.predict(X_test)
y_pred_linear_weights = sgd_linear_weights.predict(X_test)
y_pred_nonlinear = sgd_nonlinear.predict(X_test)

print('accuracy linear no weights %s' % accuracy_score(y_pred_linear, y_test))
print('accuracy linear weights %s' % accuracy_score(y_pred_linear_weights, y_test))
print('accuracy nonlinear weights %s' % accuracy_score(y_pred_nonlinear, y_test))

target_names = ['__label__food&drink', '__label__games&toys', '__label__sports', '__label__travel']

print("report linear no weights", classification_report(y_test, y_pred_linear,target_names=target_names))
print("report linear weights", classification_report(y_test, y_pred_linear_weights,target_names=target_names))
print("report nonlinear weights", classification_report(y_test, y_pred_nonlinear,target_names=target_names))