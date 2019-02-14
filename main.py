from pandas import DataFrame
from json import load, loads
from pandas import read_pickle

data = []
path = 'scrapedsites.json'


with open(path) as f:
    for line in f:
        d = loads(line)
        data.append(d)
    dataFrame = DataFrame(data)


df = read_pickle('data.pkl')

df.groupby('Category').count()

print(len(df[df['Text'] == ""]))
