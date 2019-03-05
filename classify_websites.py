#this file will contain training the model and testing its results
from pandas import read_pickle
import fasttext

csv_path_train = 'files/data_cleaned_train.txt'
csv_path_test = 'files/data_cleaned_test.txt'
model_path = 'files/model'
model_clf_path = 'files/model_clf'

#model = fasttext.skipgram(csv_path_train, model_path)

#classifier = fasttext.supervised(csv_path_train, model_clf_path)

classifier = fasttext.load_model('files/model_clf.bin')
model = fasttext.load_model('files/model.bin')
print(model.words) # list of words in dictionary
print(model['cooking'])

texts = ['recipes cooking best recipes web skip content toggle mobile menu recipes cooking search breakfast desserts sweets drinks dinner salads soups stews recipes cooking best recipes web food recipes amazing avocado recipes need try kaya february desserts sweets brownie recipes satisfy sweet tooth kaya february february food recipes keto recipes breakfast lunch dinner kaya february february dinner quick easy keto dinner recipes make minutes kaya january desserts sweets best valentine day dessert recipes kaya january dinner quick easy minute dinner recipes kaya january january salads best healthy easy salad recipes kaya january january food recipes easy fun christmas food ideas kaya october october food recipes best halloween party food ideas kaya september september appetizers creative halloween party appetizers kaya september september desserts sweets best halloween cake recipes kaya september september food recipes amazing avocado recipes need try kaya february avocados guacamole healthy fruit used variety ways salads side dishes continue reading desserts sweets brownie recipes satisfy sweet tooth kaya february february food recipes keto recipes breakfast lunch dinner kaya february february dinner quick easy keto dinner recipes make minutes kaya january desserts sweets best valentine day dessert recipes kaya january dinner quick easy minute dinner recipes kaya january january salads best healthy easy salad recipes kaya january january food recipes easy fun christmas food ideas kaya october october food recipes best halloween party food ideas kaya september september appetizers creative halloween party appetizers kaya september september desserts sweets best halloween cake recipes kaya september september posts navigation next recipes cooking recent posts amazing avocado recipes need try brownie recipes satisfy sweet tooth keto recipes breakfast lunch dinner quick easy keto dinner recipes make minutes best valentine day dessert recipes popular easy party food ideas kids throwing party child soon festive th july dessert recipes planning host independence day different ways make pizza people ask favorite mouthwatering fruit leather recipes love fruit leathers nutritious high energy snacks best chicken recipes need try chicken one popular foods love contact privacy policy powered wordpress theme cali athemes']
labels = classifier.predict(texts)
labels = classifier.predict_proba(texts)
print(labels)
result = classifier.test(csv_path_test)
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)