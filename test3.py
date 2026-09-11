import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

print(tr_input.shape, te_input.shape)

# 랜덤 포레스트는 트리 기반이라 특성 스케일이 결과에 영향을 주지 않는다.
# test2.py 와 달리 StandardScaler 를 쓰지 않는 이유.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(rf, tr_input, tr_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

rf.fit(tr_input, tr_target)
print(rf.score(tr_input, tr_target))
print(rf.score(te_input, te_target))

# 결정 트리 하나와 특성 중요도 비교. 랜덤 포레스트는 노드마다 특성을 무작위로
# 골라 쓰므로 sugar 쏠림이 줄고 나머지 특성이 조금씩 기회를 더 얻는다.
# (dt 에 max_depth 를 걸어 얕게 만들면 쏠림이 커져 차이가 더 뚜렷해진다.)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(tr_input, tr_target)

print(data.columns.tolist())
print(dt.feature_importances_.round(3))
print(rf.feature_importances_.round(3))

# OOB: 부트스트랩에서 뽑히지 않은 샘플로 매긴 점수. 검증 세트를 따로 떼지 않아도 된다.
rf_oob = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf_oob.fit(tr_input, tr_target)
print(rf_oob.oob_score_)

# import matplotlib.pyplot as plt

# plt.barh(data.columns, rf.feature_importances_)
# plt.show()
