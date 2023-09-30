import pandas as pd

df = pd.read_csv('train.csv')
df = df.dropna(axis=1)
