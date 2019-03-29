#By Johanna Thiemich
#Script for examining the data (number of rows, examining count in each class, etc.)

from pandas import DataFrame, read_csv
import  matplotlib.pyplot as plt
import numpy as np

data = []
json_path = 'files/scrapedsites.json'
csv_path = 'files/data.txt'
csv_path_cleaned = 'files/data_cleaned.txt'

df_after = read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
df_before = read_csv(csv_path, header = None, names = ['Category', 'Text'], sep =' ')
size_before = df_before.count()[0]
size_after = df_after.count()[0]

print("sife before: ", size_before)
print("sife after: ", size_after)

grouped_df_before = df_before.groupby('Category').count()
print("before: ", grouped_df_before)

grouped_df_after = df_after.groupby('Category').count()
print("after: ", grouped_df_after)

categories = ('Sports', 'Games & Toys', 'Travel', 'Food & Drink')
y_pos = np.arange(len(categories))
performance = [4766, 2225, 2225, 1410]

max_words = 0
min_words = 2000
word_count = 0
for index, row in df_after.iterrows() :
    mytext = df_after.iloc[index]['Text']
    mytext = mytext.split()
    if len(mytext) > max_words:
        max_words = len(mytext)
    if len(mytext) < min_words:
        min_words = len(mytext)
    word_count = word_count + len(mytext)

average_words = word_count / df_after.count()
print("Max words: ", max_words)
print("Min words: ", min_words)
print("Average words: ", average_words)
print("Sum words: ", word_count)

#bar diagram showing distribution of instances across classes in cleaned data
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, categories)
plt.ylabel('Count')
plt.title('Distribution of instances across classes')
plt.savefig("classes.png")
plt.show()
