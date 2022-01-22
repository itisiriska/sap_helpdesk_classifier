import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

XTRAIN_PATH = "data/train.csv"
XTEST_PATH = "data/test.csv"
DATASET_PATH = "data/dataset.xlsx"
RANDOM_STATE = 42

dataset = pd.read_excel(DATASET_PATH, index=False)
dataset.head()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = [], [], [], []
for train_index, test_index in split.split(dataset['text'], dataset['class']):
    X_train, X_test = dataset['text'][train_index], dataset['text'][test_index]
    y_train, y_test = dataset['class'][train_index], dataset['class'][test_index]

X = X_train.to_frame()
X['class'] = y_train

T = X_test.to_frame()
T['class'] = y_test
T.groupby('class').count().plot(kind='bar', figsize=(15, 6))

X.to_csv(XTRAIN_PATH, index=False)
T.to_csv(XTEST_PATH, index=False)
