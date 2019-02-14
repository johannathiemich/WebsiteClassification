from pandas import DataFrame
from json import load, loads


data = []
path = 'scrapedsites.json'

with open(path) as f:
    for line in f:
        d = loads(line)
        data.append(d)
    dataFrame = DataFrame(data)

from pandas import read_pickle
df = read_pickle('data.pkl')
df.groupby('Category').count()

len(df[df['Text'] == ""])