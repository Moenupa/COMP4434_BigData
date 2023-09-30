import pandas as pd
import numpy as np

y_pred = np.array([1, 2, 3, 4, 5])
y_test = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})

for i, row in df.iterrows():
    print(row)