import pandas as pd 
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

print(wine)
print(wine.info())
print(wine.describe())

data = wine[['alcohol', 'sugar', 'pH']]

target = wine['class']

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

print(tr_input.shape, te_input.shape)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(tr_input)
tr_scaled = ss.transform(tr_input)
te_scaled = ss.transform(te_input)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(tr_scaled, tr_target)
print(lr.score(tr_scaled, tr_target))
print(lr.score(te_scaled, te_target))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(tr_scaled, tr_target)

print(dt.score(tr_scaled, tr_target))
print(dt.score(te_scaled, te_target))

import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))

plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])

plt.show()