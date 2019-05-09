import pandas as pd

csv_path_cleaned = 'files/data_cleaned.txt'
df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
df.to_pickle(path="data_cleaned.pkl")




