import xgboost as xgb
import pandas as pd
import numpy as np

COLS = [
        'Customer ID', 'Name', 'Age', 'Profession', 'Location', 'Expense Type 1', 'Expense Type 2',
        'No. of Defaults', 'Property ID', 'Property Type', 'Co-Applicant', 'Property Price'
    ]
COLS_WITH_NA = [
    'Gender', 'Income (USD)', 'Income Stability', 'Type of Employment', 'Current Loan Expenses (USD)',
    'Dependents', 'Credit Score', 'Has Active Credit Card', 'Property Age', 'Property Location', 'Loan Amount'
]

COLS_ID = ['Customer ID', 'Property ID']
COLS_INFO = ['Name', 'Gender']
COLS_IGNORE = ['Property Age', 'Property Location', 'Property Type', ] # + ['Income (USD)', 'Dependents']
COLS_DM = [
    'Location', 'Expense Type 1', 'Expense Type 2', 'Has Active Credit Card', 'Income Stability', 'Profession', 'Type of Employment', 'Dependents'
]

train = pd.read_csv("train.csv")
train = train[~train['Loan Amount'].isna()]
test = pd.read_csv("test.csv")
test_ids = np.array(test['Customer ID']).flatten()

def preprocessing(dataframe, given_cols=None):
    df = dataframe.copy()
    df = df.drop(columns = COLS_INFO + COLS_IGNORE)
    df = pd.get_dummies(df, columns=COLS_DM)
    median_cols = ['Income (USD)', 'Credit Score', 'Current Loan Expenses (USD)']
    if not given_cols:
        for i in range(len(median_cols)):
            name = median_cols[i]
            col = df[name]
            median = col.median()
            print(f"{name} fillna with {median} {col.isna().sum()}")
            col.fillna(median, inplace=True)
    else:
        for col, val in zip(median_cols, given_cols):
            df[col].fillna(val, inplace=True)
        
        
    df = df.loc[:, ~df.columns.isin(COLS_ID)]
    return df

dataset = preprocessing(train)
X_test = preprocessing(test, [2219.34, 740.1, 375.385])



X_train = dataset.copy()
y_train = X_train.pop('Loan Amount')
for col in X_train.columns.to_list():
    if col not in X_test.columns.to_list():
        print(f"{col} not in test")
        X_test[col] = np.nan
X_test.fillna(0, inplace=True)

model = xgb.XGBRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(test_ids[:5])
print(y_pred[:5])
pd.DataFrame({'Customer ID': test_ids, 'Loan Amount': y_pred}).to_csv("submission.csv", index=False)