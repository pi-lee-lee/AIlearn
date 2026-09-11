import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']]
fish_target = fish['Species']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(fish_input, fish_target,random_state=42)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(tr_input)
tr_scaled = ss.transform(tr_input)
te_scaled = ss.transform(te_input)

from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(tr_scaled, tr_target)
print(sc.score(tr_scaled, tr_target))
print(sc.score(te_scaled, te_target))

sc.partial_fit(tr_scaled,tr_target)
print(sc.score(tr_scaled, tr_target))
print(sc.score(te_scaled, te_target))

