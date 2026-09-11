import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


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

sc = SGDClassifier(loss='log_loss',max_iter=100, tol=None, random_state=42)
tr_score = []
te_score = []

classes1 = np.unique(tr_target)

sc.fit(tr_scaled, tr_target)


# for _ in range(0, 300):
#     sc.partial_fit(tr_scaled, tr_target, classes=classes1)
#     tr_score.append(sc.score(tr_scaled, tr_target))
#     te_score.append(sc.score(te_scaled, te_target))

print(sc.predict(te_scaled[:5]))
print(te_target[:5])
print(sc.decision_function(te_scaled[:5]).round(3))

print(sc.score(tr_scaled, tr_target))
print(sc.score(te_scaled, te_target))

# plt.plot(tr_score)
# plt.plot(te_score)

# plt.show()