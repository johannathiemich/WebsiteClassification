from pandas import DataFrame
from json import load, loads


data = []
path = 'scrapedsites.json'

with open(path) as f:
    for line in f:
        d = loads(line)
        data.append(d)
    dataFrame = DataFrame(data)
